#!/usr/bin/env python3
"""
Extract protein features from PDB files.
Combines secondary structure (DSSP), relative SASA (PyRosetta), and structural information.
"""

import sys
import os
import argparse
import numpy as np
import torch
import pydssp
import pyrosetta
from Bio import PDB

# ============================================================================
# Secondary Structure Calculation
# ============================================================================

AA_THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

def safe_three_to_one(res_name):
    """Convert three-letter amino acid code to one-letter code."""
    try:
        return PDB.Polypeptide.three_to_one(res_name)
    except (AttributeError, KeyError):
        return AA_THREE_TO_ONE.get(res_name.upper(), None)

def get_all_residues_from_pdb(pdb_path):
    """Extract all valid residues from PDB file."""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    
    all_residues = []
    
    for model in structure:
        for chain in model:
            chain_id = chain.id
            
            for res in chain:
                if res.id[0] != " ":
                    continue
                    
                aa_one = safe_three_to_one(res.get_resname())
                if aa_one is None:
                    continue
                
                missing_backbone = False
                for atom_name in ['N', 'CA', 'C', 'O']:
                    if atom_name not in res:
                        missing_backbone = True
                        break

                if missing_backbone:
                    continue

                all_residues.append({
                    'chain_id': chain_id,
                    'res_num': res.id[1],
                    'res_name': res.get_resname(),
                    'aa_one': aa_one,
                    'residue': res
                })
        
        break
    
    return all_residues

class Protein:
    def __init__(self, bb_atom_coords, residue_mapping=None):
        self.bb_atom_coords = bb_atom_coords
        self.residue_mapping = residue_mapping

    @staticmethod
    def FromValidResidues(valid_residues):
        """Create protein object from valid residues."""
        atom_order = ['N', 'CA', 'C', 'O']
        coords = []
        residue_mapping = []
        
        for res_info in valid_residues:
            res = res_info['residue']
            try:
                res_coords = []
                for atom_name in atom_order:
                    res_coords.append(res[atom_name].get_coord())
                coords.append(res_coords)
                residue_mapping.append((res_info['chain_id'], res_info['res_num']))
            except KeyError:
                continue
        
        return Protein(np.array(coords), residue_mapping)

def compute_secondary_structure(pdb_path):
    """Compute secondary structure using DSSP."""
    all_residues = get_all_residues_from_pdb(pdb_path)
    
    if len(all_residues) == 0:
        raise ValueError(f"No valid residues found in {pdb_path}")
    
    protein = Protein.FromValidResidues(all_residues)
    bb_coords = protein.bb_atom_coords
    residue_mapping = protein.residue_mapping
    
    dssp_ss = None
    if len(bb_coords) > 0:
        try:
            dssp = pydssp.assign(bb_coords, out_type='c3')
            dssp_ss = ''.join(list(dssp)).replace('-', 'C')
        except Exception as e:
            print(f"DSSP failed: {e}")
            dssp_ss = None
    
    dssp_mapping = {}
    if dssp_ss and len(dssp_ss) == len(residue_mapping):
        for i, (chain_id, res_num) in enumerate(residue_mapping):
            dssp_mapping[(chain_id, res_num)] = dssp_ss[i]
    
    final_ss = []
    for res_info in all_residues:
        key = (res_info['chain_id'], res_info['res_num'])
        if key in dssp_mapping:
            final_ss.append(dssp_mapping[key])
        else:
            final_ss.append('C')
    
    return ''.join(final_ss), all_residues

# ============================================================================
# SASA Calculation
# ============================================================================

MAX_SASA_DICT = {
    'ALA': 129, 'ARG': 274, 'ASN': 195, 'ASP': 193, 'CYS': 167,
    'GLN': 223, 'GLU': 225, 'GLY': 104, 'HIS': 224, 'ILE': 197,
    'LEU': 201, 'LYS': 236, 'MET': 224, 'PHE': 240, 'PRO': 159,
    'SER': 155, 'THR': 172, 'TRP': 285, 'TYR': 263, 'VAL': 174
}

def compute_relative_sasa(pdb_path):
    """Compute relative solvent accessible surface area using PyRosetta."""
    try:
        pose = pyrosetta.pose_from_pdb(pdb_path)
    except Exception as e:
        print(f"PyRosetta failed for {pdb_path}: {e}")
        return None
    
    from pyrosetta.rosetta.core.scoring.sasa import SasaCalc
    from pyrosetta.rosetta.utility import vector1_double

    sasa_calc = SasaCalc()
    rsd_sasa = vector1_double()
    rsd_hsasa = vector1_double()
    sasa_calc.calculate(pose, rsd_sasa, rsd_hsasa)

    sasa_results = []
    for i in range(1, pose.total_residue() + 1):
        res_name = pose.residue(i).name3()
        abs_sasa = rsd_sasa[i]
        max_sasa = MAX_SASA_DICT.get(res_name, None)
        if max_sasa:
            rel_sasa = abs_sasa / max_sasa
        else:
            rel_sasa = 0.0
        
        sasa_results.append({
            'res_id': i,
            'res_name': res_name,
            'abs_sasa': abs_sasa,
            'rel_sasa': rel_sasa
        })
    
    return sasa_results

# ============================================================================
# Feature Integration
# ============================================================================

def ss_to_onehot(ss_seq):
    """Convert secondary structure sequence to one-hot encoding (N, 3)."""
    mapping = {'H': [1,0,0], 'C': [0,1,0], 'E': [0,0,1]}
    return np.array([mapping.get(s, [0,0,0]) for s in ss_seq], dtype=np.float32)

def extract_all_features(pdb_path, verbose=False):
    """
    Extract all features from PDB file.
    
    Args:
        pdb_path: Path to PDB file
        verbose: Whether to print detailed information
    
    Returns:
        Dictionary containing all extracted features
    """
    ss_sequence, residues = compute_secondary_structure(pdb_path)
    ss_onehot = ss_to_onehot(list(ss_sequence))
    
    sasa_results = compute_relative_sasa(pdb_path)
    
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    
    chain_data = []
    for model in structure:
        for chain in model:
            chain_id = chain.id
            bb_coords = []
            seq = []
            residue_ids = []
            
            for res in chain:
                if res.id[0] != " ":
                    continue
                
                try:
                    aa_one = PDB.Polypeptide.three_to_one(res.get_resname())
                except KeyError:
                    continue
                
                missing_backbone = False
                for atom_name in ['N', 'CA', 'C', 'O']:
                    if atom_name not in res:
                        missing_backbone = True
                        break

                if not missing_backbone:
                    try:
                        coords = []
                        for atom_name in ['N', 'CA', 'C', 'O']:
                            coords.append(res[atom_name].get_coord())
                        bb_coords.append(coords)
                        seq.append(aa_one)
                        residue_ids.append(res.id[1])
                    except KeyError:
                        continue
            
            if len(bb_coords) > 0:
                chain_data.append({
                    'chain_id': chain_id,
                    'coords': np.array(bb_coords),
                    'sequence': ''.join(seq),
                    'residue_ids': residue_ids
                })
        break
    
    features = {
        'pdb_path': pdb_path,
        'pdb_name': os.path.basename(pdb_path),
        'num_chains': len(chain_data),
        'chains': chain_data,
        'secondary_structure': ss_sequence,
        'ss_onehot': ss_onehot,
        'sasa': sasa_results,
        'total_residues': len(residues)
    }
    
    return features

# ============================================================================
# Batch Processing
# ============================================================================

def process_pdb_directory(pdb_dir, output_dir, verbose=False):
    """
    Process all PDB files in a directory and save features to .pt files.
    
    Args:
        pdb_dir: Directory containing PDB files
        output_dir: Output directory for feature files
        verbose: Whether to print detailed information
    
    Returns:
        Number of successfully processed files
    """
    pdb_files = sorted([f for f in os.listdir(pdb_dir) if f.endswith('.pdb')])
    
    print(f"Found {len(pdb_files)} PDB files")
    print(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Initializing PyRosetta...")
    pyrosetta.init("-mute all")
    
    success_count = 0
    error_count = 0
    
    print("Processing...")
    for i, pdb_file in enumerate(pdb_files, 1):
        pdb_path = os.path.join(pdb_dir, pdb_file)
        output_file = os.path.join(output_dir, f"{pdb_file[:-4]}_features.pt")
        
        try:
            if os.path.exists(output_file):
                print(f"[{i}/{len(pdb_files)}] Skipping {pdb_file} (already exists)")
                success_count += 1
                continue
            
            features = extract_all_features(pdb_path, verbose=verbose)
            
            torch.save(features, output_file)
            
            print(f"[{i}/{len(pdb_files)}] ✓ {pdb_file} -> {os.path.basename(output_file)}")
            success_count += 1
                
        except Exception as e:
            error_count += 1
            print(f"[{i}/{len(pdb_files)}] ✗ {pdb_file}: {e}")
    
    print(f"\n{'='*80}")
    print(f"Completed: {success_count}/{len(pdb_files)} successful, {error_count}/{len(pdb_files)} failed")
    print(f"{'='*80}")
    
    return success_count

# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract protein features from PDB files (secondary structure, SASA, coordinates) and save as .pt files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Process a single PDB file
  python data_preprocess.py -i protein.pdb -o output_dir
  
  # Process all PDB files in a directory
  python data_preprocess.py -i /path/to/pdb_dir -o /path/to/output_dir
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                        help='Input PDB file or directory')
    parser.add_argument('-o', '--output', required=True,
                        help='Output directory')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print detailed information')
    
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        print(f"Processing single PDB file: {args.input}")
        print(f"Output directory: {args.output}")
        
        pyrosetta.init("-mute all")
        features = extract_all_features(args.input, verbose=args.verbose)
        
        os.makedirs(args.output, exist_ok=True)
        pdb_name = os.path.basename(args.input)[:-4]
        output_file = os.path.join(args.output, f"{pdb_name}_features.pt")
        torch.save(features, output_file)
        
        print(f"✓ Features saved to: {output_file}")
        print(f"  - Total residues: {features['total_residues']}")
        print(f"  - Number of chains: {features['num_chains']}")
        
        return features
        
    elif os.path.isdir(args.input):
        success_count = process_pdb_directory(args.input, args.output, verbose=args.verbose)
        return success_count
        
    else:
        print(f"Error: Input path does not exist: {args.input}")
        sys.exit(1)

if __name__ == "__main__":
    main()

