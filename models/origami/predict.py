"""
Command-line scoring interface for ORIGAMI.
"""

from __future__ import annotations

import argparse
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import yaml
from Bio import PDB
from torch_geometric.data import Batch, Data

from data_preprocess import MAX_SASA_DICT, compute_relative_sasa, compute_secondary_structure, ss_to_onehot
from models.origami.datasets import (
    _dihedrals,
    _orientations_interface_aware,
    _process_complex_pdb,
    _sidechains,
)
from models.origami.models import PSRNetwork


def ensure_pyrosetta():
    """Initialise PyRosetta lazily when SASA features are required."""
    try:
        import pyrosetta  # type: ignore
    except ImportError as exc:  # pragma: no cover - explicit error message
        raise RuntimeError(
            "PyRosetta is required. Install the licensed wheel inside this environment."
        ) from exc

    try:
        pyrosetta.init("-mute all")
    except RuntimeError:
        # PyRosetta was already initialised in this process.
        pass
    return pyrosetta


def compute_sasa_map(pdb_path: str) -> Dict[Tuple[str, int], float]:
    """Return relative SASA per residue keyed by (chain_id, residue_number)."""
    pyrosetta = ensure_pyrosetta()

    pose = pyrosetta.pose_from_pdb(pdb_path)
    from pyrosetta.rosetta.core.scoring.sasa import SasaCalc  # type: ignore
    from pyrosetta.rosetta.utility import vector1_double  # type: ignore

    sasa_calc = SasaCalc()
    rsd_sasa = vector1_double()
    rsd_hsasa = vector1_double()
    sasa_calc.calculate(pose, rsd_sasa, rsd_hsasa)

    pdb_info = pose.pdb_info()
    rsa_map: Dict[Tuple[str, int], float] = {}
    for i in range(1, pose.total_residue() + 1):
        chain_id = pdb_info.chain(i)
        res_num = pdb_info.number(i)
        res_name = pose.residue(i).name3()
        abs_sasa = rsd_sasa[i]
        max_sasa = MAX_SASA_DICT.get(res_name)
        rel_sasa = abs_sasa / max_sasa if max_sasa else 0.0
        rsa_map[(chain_id, res_num)] = rel_sasa
    return rsa_map


def collect_residue_indices(pdb_path: str) -> Tuple[List[Tuple[str, List[int]]], List[Tuple[str, int]]]:
    """Collect residues per chain to align DSSP/SASA features."""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)

    chain_residues: List[Tuple[str, List[int]]] = []
    for model in structure:
        for chain in model:
            chain_id = chain.id
            residue_numbers: List[int] = []
            for res in chain:
                if res.id[0] != " ":
                    continue
                try:
                    PDB.Polypeptide.three_to_one(res.get_resname())
                except KeyError:
                    continue
                if any(atom_name not in res for atom_name in ("N", "CA", "C", "O")):
                    continue
                residue_numbers.append(res.id[1])
            if residue_numbers:
                chain_residues.append((chain_id, residue_numbers))
        break

    flat = [(chain_id, res_num) for chain_id, nums in chain_residues for res_num in nums]
    return chain_residues, flat


@lru_cache(maxsize=None)
def load_secondary_structure_map(ss_file: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with open(ss_file, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                ss_str, pdb_path = line.rsplit(" ", 1)
            except ValueError:
                continue
            decoy_id = Path(pdb_path).stem
            mapping[decoy_id] = ss_str
    return mapping


def load_secondary_structure_for_decoy(
    decoy_id: str,
    ss_file: Optional[str],
    data_root: Optional[str],
) -> Optional[str]:
    candidate_files: List[Path] = []
    if ss_file:
        candidate_files.append(Path(ss_file))
    if data_root:
        ss_dir = Path(data_root) / "secondary_structure"
        if ss_dir.exists():
            candidate_files.extend(sorted(ss_dir.glob("*.result")))

    for path in candidate_files:
        if not path.exists():
            continue
        mapping = load_secondary_structure_map(str(path))
        sequence = mapping.get(decoy_id)
        if sequence is not None:
            return sequence
    return None


def load_rsa_values(
    decoy_id: str,
    rsa_dir: Optional[str],
    data_root: Optional[str],
) -> Optional[Dict[int, float]]:
    candidate_dirs: List[Path] = []
    if rsa_dir:
        candidate_dirs.append(Path(rsa_dir))
    if data_root:
        root_path = Path(data_root)
        for child in root_path.iterdir():
            if child.is_dir() and child.name.startswith("rsasa"):
                candidate_dirs.append(child)

    for directory in candidate_dirs:
        if not directory.exists():
            continue
        rsa_path = directory / f"{decoy_id}.txt"
        if rsa_path.exists():
            from models.origami.datasets import parse_rsa_file

            return parse_rsa_file(str(rsa_path))
    return None


def build_interface_graph(
    pdb_path: str,
    ss_file: Optional[str] = None,
    rsa_dir: Optional[str] = None,
    data_root: Optional[str] = None,
) -> Data:
    """Construct a torch-geometric Data object for a single complex."""
    base = _process_complex_pdb(pdb_path, json_file=None, quality_score=0.0)

    if len(base.chain_lengths) != 2:
        raise ValueError(
            f"ORIGAMI expects exactly 2 chains; found {len(base.chain_lengths)} chains in {pdb_path}."
        )

    chain_residues, flat_res = collect_residue_indices(pdb_path)
    if base.num_nodes != len(flat_res):
        raise ValueError(
            "Residue parsing mismatch. Ensure the PDB contains complete backbone atoms."
        )

    decoy_id = Path(pdb_path).stem

    ss_override = load_secondary_structure_for_decoy(decoy_id, ss_file, data_root)
    if ss_override is None:
        ss_sequence, residue_records = compute_secondary_structure(pdb_path)
        ss_map = {
            (res["chain_id"], res["res_num"]): ss
            for res, ss in zip(residue_records, ss_sequence)
        }
        ss_list = [ss_map.get(key, "C") for key in flat_res]
    else:
        ss_list = list(ss_override)
        if len(ss_list) != len(flat_res):
            if len(ss_list) < len(flat_res):
                ss_list.extend(["C"] * (len(flat_res) - len(ss_list)))
            else:
                ss_list = ss_list[: len(flat_res)]
    ss_tensor = torch.tensor(ss_to_onehot(ss_list), dtype=torch.float32)

    rsa_override = load_rsa_values(decoy_id, rsa_dir, data_root)
    rsa_override = load_rsa_values(decoy_id, rsa_dir, data_root)
    if rsa_override is None:
        ensure_pyrosetta()
        sasa_results = compute_relative_sasa(pdb_path)
        if not sasa_results or len(sasa_results) != len(flat_res):
            rsa_map = compute_sasa_map(pdb_path)
            rsa_list = [rsa_map.get(key, 0.0) for key in flat_res]
        else:
            rsa_list = [entry.get("rel_sasa", 0.0) for entry in sasa_results]
    else:
        rsa_list = []
        for chain_id, res_numbers in chain_residues:
            for res_num in res_numbers:
                rsa_list.append(rsa_override.get(res_num, 0.0))
        if len(rsa_list) != len(flat_res):
            rsa_map = compute_sasa_map(pdb_path)
            rsa_list = [rsa_map.get(key, 0.0) for key in flat_res]
    rsa_tensor = torch.tensor(rsa_list, dtype=torch.float32).unsqueeze(-1)

    dihedrals = _dihedrals(base.pos_N, base.pos_CA, base.pos_C)
    sidechains = _sidechains(base.pos_N, base.pos_CA, base.pos_C)
    onehot = F.one_hot(base.seq, num_classes=20).float()
    chain_encoding = torch.zeros(base.num_nodes, 2, dtype=torch.float32)
    chain_encoding[: base.chain_lengths[0], 0] = 1
    chain_encoding[base.chain_lengths[0] :, 1] = 1
    norm_pos = base.norm_pos.unsqueeze(-1)

    interface_indices = torch.nonzero(base.interface_mask, as_tuple=False).squeeze(-1)
    if interface_indices.numel() == 0:
        interface_indices = torch.arange(base.num_nodes, device=base.pos_CA.device)

    orientations = _orientations_interface_aware(
        pos_CA=base.pos_CA[interface_indices],
        chain_indicators=chain_encoding[interface_indices],
    )

    node_s = torch.cat(
        [
            dihedrals[interface_indices],
            onehot[interface_indices],
            chain_encoding[interface_indices],
            norm_pos[interface_indices],
            ss_tensor[interface_indices],
            rsa_tensor[interface_indices],
        ],
        dim=-1,
    )
    node_v = torch.cat(
        [
            orientations,
            sidechains[interface_indices].unsqueeze(1),
        ],
        dim=1,
    )

    interface_data = Data(
        pos_N=base.pos_N[interface_indices],
        pos_CA=base.pos_CA[interface_indices],
        pos_C=base.pos_C[interface_indices],
        pos_O=base.pos_O[interface_indices],
        seq=base.seq[interface_indices],
        num_nodes=interface_indices.size(0),
        node_s=node_s,
        node_v=node_v,
        target_id=base.target_id,
        decoy_id=base.decoy_id,
        chain_indicators=chain_encoding[interface_indices],
        original_indices=interface_indices,
    )
    interface_data.ilddt = torch.tensor(0.0, dtype=torch.float32)
    return interface_data


def load_model(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
    config: Optional[Dict[str, Any]] = None,
) -> PSRNetwork:
    if config is None:
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

    model_cfg = config.get("model", {})
    model = PSRNetwork(
        node_in_dims=tuple(model_cfg.get("node_in_dims", [33, 3])),
        node_hid_dims=tuple(model_cfg.get("node_hid_dims", [128, 32])),
        edge_hid_dims=tuple(model_cfg.get("edge_hid_dims", [64, 16])),
        num_layers=model_cfg.get("num_layers", 6),
        drop_rate=model_cfg.get("drop_rate", 0.15),
        top_k=model_cfg.get("top_k", 40),
        num_rbf=model_cfg.get("num_rbf", 16),
        num_positional_embeddings=model_cfg.get("num_positional_embeddings", 16),
        perceptron_mode=model_cfg.get("perceptron_mode", "svp"),
        conv=model_cfg.get("conv", "gnn"),
        num_heads=model_cfg.get("num_heads", 4),
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def score_complex(
    pdb_path: str,
    checkpoint_path: str,
    config_path: str,
    device: torch.device,
    ss_file: Optional[str] = None,
    rsa_dir: Optional[str] = None,
) -> float:
    config_data: Optional[Dict[str, Any]]
    data_root: Optional[str] = None
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            config_data = yaml.safe_load(handle)
            data_root = config_data.get("data", {}).get("root")
    except FileNotFoundError:
        config_data = None

    data = build_interface_graph(
        pdb_path,
        ss_file=ss_file,
        rsa_dir=rsa_dir,
        data_root=data_root,
    )
    batch = Batch.from_data_list([data]).to(device)
    model = load_model(config_path, checkpoint_path, device, config=config_data)
    with torch.no_grad():
        value = model(batch).item()
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a protein-protein complex using ORIGAMI.",
    )
    parser.add_argument("pdb", help="Path to the input PDB file containing two chains.")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join("pretrained_models", "checkpoints", "best.pt"),
        help="Path to the pretrained checkpoint (.pt).",
    )
    parser.add_argument(
        "--config",
        default=os.path.join("pretrained_models", "dis24.yml"),
        help="YAML configuration used during training.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device identifier (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the prediction score as JSON.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the prediction as JSON instead of a plain float.",
    )
    parser.add_argument(
        "--ss-file",
        help="Optional path to a secondary-structure result file (e.g. SS_casp16.result).",
    )
    parser.add_argument(
        "--rsa-dir",
        help="Optional directory containing per-decoy RSA text files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    score = score_complex(
        pdb_path=args.pdb,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=device,
        ss_file=args.ss_file,
        rsa_dir=args.rsa_dir,
    )

    result = {"pdb": os.path.abspath(args.pdb), "score": score}

    if args.json or args.output:
        payload = json.dumps(result, indent=2 if args.json else None)
        if args.json:
            print(payload)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(payload if payload.endswith("\n") else f"{payload}\n")
    else:
        message = (
            f"Predicted iLDDT for protein complex {os.path.basename(args.pdb)} is: "
            f"{score:.6f}"
        )
        print(message)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)
                handle.write("\n")


if __name__ == "__main__":
    main()

