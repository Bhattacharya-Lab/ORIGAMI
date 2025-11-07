import math
import os
import json
from joblib import Parallel, delayed
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, BatchSampler
from torch_geometric.data import Data, Batch
from Bio import PDB
from tqdm.auto import tqdm
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


# --- Helper functions for new features ---
def parse_secondary_structure(ss_file):
    """
    Parse secondary structure file into a dict: {pdb_path: [ss_char, ...]}
    """
    ss_dict = {}
    with open(ss_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            *ss_seq, pdb_path = line.split()
            # The SS sequence may be a single string or space-separated chars
            if len(ss_seq) == 1:
                ss_seq = list(ss_seq[0])
            ss_dict[pdb_path] = ss_seq
    return ss_dict

def parse_rsa_file(rsa_file):
    """
    Parse a single RSA file into a dict: {res_id: rel_sasa}
    """
    rsa_dict = {}
    with open(rsa_file, 'r') as f:
        for line in f:
            if line.startswith('ResID') or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                res_id = int(parts[0])
                rel_sasa = float(parts[3])
                rsa_dict[res_id] = rel_sasa
            except Exception:
                continue
    return rsa_dict

def ss_to_onehot(ss_seq):
    """
    Convert SS sequence (list of 'H', 'E', 'C') to one-hot numpy array (N, 3)
    """
    mapping = {'H': [1,0,0], 'C': [0,1,0], 'E': [0,0,1]}
    return np.array([mapping.get(s, [0,0,0]) for s in ss_seq], dtype=np.float32)

# --- Copy all code from datasets3.py, but add new features in __getitem__ ---
# ... existing code ...
# (Copy all code from datasets3.py up to and including ComplexInterfaceDataset)
# ... existing code ...
def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _process_complex_pdb(pdb_file, json_file):
    
    # Read quality score from JSON
    with open(json_file, 'r') as f:
        data_json = json.load(f)
        quality_score = data_json['ilddt']
        
    # Load structure with Bio.PDB
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    # Initialize chain_data list to store processed chains
    chain_data = []
    
    for model in structure:
        for chain in model:
            chain_id = chain.id
            
            # Extract backbone coordinates
            bb_coords = []
            seq = []
            residue_ids = []
            
            # Count all residues first to check
            all_residues = list(chain.get_residues())
           
            for res in chain:
                # Skip hetero and water atoms
                if res.id[0] != " ":
                    continue
                    
                # Check if this is a standard amino acid
                try:
                    aa_one = PDB.Polypeptide.three_to_one(res.get_resname())
                except KeyError:
                    # Skip non-standard residues
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
                        residue_ids.append(res.id[1])  # Store residue number
                    except KeyError:
                        # This should not happen if we checked correctly
                        print(f"Unexpected error: Skipping residue {res.id} due to missing backbone atoms")
                        continue
            
            if len(bb_coords) == 0:
                raise ValueError(f"No valid residues found in chain {chain_id}")
                
            bb_coords = torch.FloatTensor(bb_coords)    # (L, 4, 3), N, CA, C, O
            pos_N, pos_CA, pos_C, pos_O = torch.unbind(bb_coords, 1)
            seq_str = ''.join(seq)
            seq = torch.tensor([PDB.Polypeptide.one_to_index(aa) for aa in seq], dtype=torch.long)
            
            # Create normalized position indices for this chain
            chain_length = len(seq)
            norm_pos = torch.arange(chain_length, dtype=torch.float32) / max(chain_length - 1, 1)
            
            chain_data.append({
                'chain_id': chain_id,
                'pos_N': pos_N,
                'pos_CA': pos_CA,
                'pos_C': pos_C,
                'pos_O': pos_O,
                'seq': seq,
                'seq_fasta': seq_str,
                'length': len(seq),
                'residue_ids': residue_ids,
                'norm_pos': norm_pos
            })
        
        # We only need to process the first model
        break
    
    # Check if we have exactly two chains (allow homodimers)
    if len(chain_data) != 2:
        raise ValueError(f"Expected exactly 2 chains, but found {len(chain_data)}")
    
    # Allow identical chains if you want to study homodimers
    # Check for unique sequences
    unique_sequences = set()
    for chain in chain_data:
        unique_sequences.add(chain['seq_fasta'])
    
    # Accept both homodimers and heterodimers
    if len(unique_sequences) >= 1 and len(chain_data) == 2:
        # This accepts both cases:
        # - Homodimers: 2 chains with identical sequences (len(unique_sequences) = 1)
        # - Heterodimers: 2 chains with different sequences (len(unique_sequences) = 2)
        pass  # Continue processing
    else:
        raise ValueError(f"Expected exactly 2 chains with valid residues, but found {len(unique_sequences)} unique sequences in {len(chain_data)} chains")
    
    # Identify interface residues
    interface_mask1, interface_mask2 = identify_interface_residues(
        chain_data[0]['pos_CA'], 
        chain_data[1]['pos_CA'],
        distance_threshold=24.0
    )
    
    # Create combined data
    basename = os.path.basename(pdb_file)
    pdb_name = basename[:-4]  # Remove .pdb
    # Special handling for targets starting with H or T
    if pdb_name[0] == 'H':
        if pdb_name.startswith('H1114v2'):
            target_id = pdb_name[:7]  # Take H1114v2
        elif pdb_name.startswith('H1114v1') or pdb_name.startswith('H1114v3'):
            target_id = pdb_name[:5]  # Take H1114
        else:
            target_id = pdb_name[:5]
    elif pdb_name[0] == 'T':
        if pdb_name.startswith('T1176v1'):
            target_id = pdb_name[:7]  # Take T1176v1
        else:
            target_id = pdb_name[:5]  # Take T1176 or other T cases
    else:
        target_id = pdb_name[:4]
    
    data = Data(
        num_nodes=chain_data[0]['length'] + chain_data[1]['length'],
        pos_N=torch.cat([c['pos_N'] for c in chain_data]),
        pos_CA=torch.cat([c['pos_CA'] for c in chain_data]),
        pos_C=torch.cat([c['pos_C'] for c in chain_data]),
        pos_O=torch.cat([c['pos_O'] for c in chain_data]),
        seq=torch.cat([c['seq'] for c in chain_data]),
        seq_fasta=[c['seq_fasta'] for c in chain_data],
        chain_ids=[c['chain_id'] for c in chain_data],
        chain_lengths=[c['length'] for c in chain_data],
        interface_mask=torch.cat([interface_mask1, interface_mask2]),
        target_id=target_id,
        decoy_id=pdb_name,
        norm_pos=torch.cat([c['norm_pos'] for c in chain_data])  # Add normalized position indices
    )
    
    # Add ILDDT score
    data.ilddt = torch.tensor(quality_score, dtype=torch.float)
    
    return data


def identify_interface_residues(pos_CA_chain1, pos_CA_chain2, distance_threshold=24.0):
    """
    Identifies interface residues between two chains based on CA-CA distance.
    
    Args:
        pos_CA_chain1: (N1, 3) tensor of CA positions from chain 1
        pos_CA_chain2: (N2, 3) tensor of CA positions from chain 2
        distance_threshold: maximum distance to consider for interface (default: 24Å)
    
    Returns:
        interface_mask_chain1: (N1,) boolean tensor, True for interface residues in chain 1
        interface_mask_chain2: (N2,) boolean tensor, True for interface residues in chain 2
    """
    device = pos_CA_chain1.device
    
    # Calculate all pairwise distances between chains
    diff = pos_CA_chain1.unsqueeze(1) - pos_CA_chain2.unsqueeze(0)  # (N1, N2, 3)
    distances = torch.norm(diff, dim=2)  # (N1, N2)
    
    # Find residue pairs within threshold
    within_threshold = distances < distance_threshold
    
    # Create interface masks for each chain
    interface_mask_chain1 = torch.zeros(pos_CA_chain1.size(0), dtype=torch.bool, device=device)
    interface_mask_chain2 = torch.zeros(pos_CA_chain2.size(0), dtype=torch.bool, device=device)
    
    # Mark residues as interface if they have any contact with the other chain
    interface_mask_chain1 = within_threshold.any(dim=1)
    interface_mask_chain2 = within_threshold.any(dim=0)
    
    return interface_mask_chain1, interface_mask_chain2


def _dihedrals(pos_N, pos_CA, pos_C, eps=1e-7):
    """Calculate dihedral angles.
    Args:
        pos_N, pos_CA, pos_C: (N, 3)
    Returns:
        Dihedral features (N, 6)
    """
    X = torch.cat([pos_N.view(-1, 1, 3), pos_CA.view(-1, 1, 3), pos_C.view(-1, 1, 3)], dim=1)
    X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = _normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

    D = F.pad(D, [1, 2])
    D = torch.reshape(D, [-1, 3])
    D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
    return D_features


def _orientations_chain_aware(pos_CA, chain_lengths):
    """Calculate orientation vectors while respecting chain boundaries.
    
    Args:
        pos_CA: (N, 3) tensor of CA positions
        chain_lengths: list of chain lengths
        
    Returns:
        Orientation features (N, 2, 3) representing forward and backward directions
    """
    X = pos_CA
    device = X.device
    
    # Initialize with zeros
    forward = torch.zeros_like(X)
    backward = torch.zeros_like(X)
    
    # Calculate for chain 1
    chain1_len = chain_lengths[0]
    if chain1_len > 1:  # Only if chain has multiple residues
        # Forward: vectors from current to next CA (except last residue)
        forward[:chain1_len-1] = _normalize(X[1:chain1_len] - X[:chain1_len-1])
        # Backward: vectors from current to previous CA (except first residue)
        backward[1:chain1_len] = _normalize(X[:chain1_len-1] - X[1:chain1_len])
    
    # Calculate for chain 2
    if len(chain_lengths) > 1 and chain_lengths[1] > 0:
        start_idx = chain1_len
        end_idx = chain1_len + chain_lengths[1]
        if end_idx - start_idx > 1:  # Only if chain has multiple residues
            # Forward: vectors from current to next CA (except last residue)
            forward[start_idx:end_idx-1] = _normalize(X[start_idx+1:end_idx] - X[start_idx:end_idx-1])
            # Backward: vectors from current to previous CA (except first residue)
            backward[start_idx+1:end_idx] = _normalize(X[start_idx:end_idx-1] - X[start_idx+1:end_idx])
    
    # Stack forward and backward into orientation tensor
    orientations = torch.stack([forward, backward], dim=1)
    
    return orientations


def _orientations(pos_CA):
    """Legacy orientation calculation - kept for backward compatibility"""
    X = pos_CA
    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


def _orientations_interface_aware(pos_CA, chain_indicators):
    """Calculate orientation vectors for interface residues while respecting chain boundaries.
    
    Args:
        pos_CA: (N, 3) tensor of CA positions of interface residues
        chain_indicators: (N, 2) tensor indicating which chain each residue belongs to
        
    Returns:
        Orientation features (N, 2, 3) representing forward and backward directions,
        with zeros at chain boundaries
    """
    device = pos_CA.device
    N = pos_CA.size(0)
    
    # Initialize with zeros
    forward = torch.zeros_like(pos_CA)
    backward = torch.zeros_like(pos_CA)
    
    if N > 1:  # Only if we have multiple residues
        # Calculate forward and backward vectors for all residues
        forward_all = _normalize(pos_CA[1:] - pos_CA[:-1])
        backward_all = _normalize(pos_CA[:-1] - pos_CA[1:])
        
        # Find chain transitions
        # chain_indicators is one-hot: first column is 1 for chain A, second column is 1 for chain B
        is_chain_A = chain_indicators[:, 0].bool()
        
        # For each residue (except last), check if it's the last residue of chain A
        is_last_in_chain_A = is_chain_A[:-1] & ~is_chain_A[1:]
        
        # For each residue (except first), check if it's the first residue of chain B
        is_first_in_chain_B = ~is_chain_A[1:] & ~is_chain_A[:-1]
        
        # Forward vectors: zero for last residue in chain A
        forward[:-1] = torch.where(
            is_last_in_chain_A.unsqueeze(1),
            torch.zeros_like(forward_all),
            forward_all
        )
        
        # Backward vectors: zero for first residue in chain B
        backward[1:] = torch.where(
            is_first_in_chain_B.unsqueeze(1),
            torch.zeros_like(backward_all),
            backward_all
        )
    
    # Stack forward and backward into orientation tensor
    orientations = torch.stack([forward, backward], dim=1)
    
    return orientations


def _sidechains(pos_N, pos_CA, pos_C):
    X = torch.cat([pos_N.view(-1, 1, 3), pos_CA.view(-1, 1, 3), pos_C.view(-1, 1, 3)], dim=1)
    n, origin, c = X[:, 0], X[:, 1], X[:, 2]
    c, n = _normalize(c - origin), _normalize(n - origin)
    bisector = _normalize(c + n)
    perp = _normalize(torch.cross(c, n))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec


class ComplexInterfaceDataset(Dataset):
    """Dataset for protein complex interface quality assessment with SS and RSA features."""
    def __init__(self, pdb_dir, ilddt_dir, ss_file, rsa_dir, cache_dir='/home/grads/xinyu0110/OAGNN_dataset2/data_24', preproc_n_jobs=32):
        super().__init__()
        self.pdb_dir = pdb_dir
        self.ilddt_dir = ilddt_dir
        self.ss_file = ss_file
        self.rsa_dir = rsa_dir
        self.cache_path = os.path.join(cache_dir, f'interface_processed_{os.path.basename(pdb_dir)}.pt')
        self.preproc_n_jobs = preproc_n_jobs
        self.dataset = None
        self.target_to_indices = {}
        # Parse SS file once
        self.ss_dict = parse_secondary_structure(ss_file)
        self._load()

    def _load(self):
        if os.path.exists(self.cache_path):
            print(f'[{self.__class__.__name__}] Loading from cache: {self.cache_path}')
            try:
                # Check cache file size
                cache_size = os.path.getsize(self.cache_path) / (1024 * 1024)  # Size in MB
                print(f'[{self.__class__.__name__}] Cache file size: {cache_size:.2f} MB')
                
                # Load the cached dataset
                self.dataset = torch.load(self.cache_path)
                print(f'[{self.__class__.__name__}] ✓ Successfully loaded {len(self.dataset)} structures from cache')
                
            except Exception as e:
                print(f'[{self.__class__.__name__}] ✗ Error loading cache: {e}')
                print(f'[{self.__class__.__name__}] Falling back to processing from scratch...')
                self.dataset = self._process()
        else:
            print(f'[{self.__class__.__name__}] Cache not found at: {self.cache_path}')
            print(f'[{self.__class__.__name__}] Processing dataset from scratch...')
            self.dataset = self._process()
            
        # Build target to indices mapping
        for index in range(len(self.dataset)):
            data = self.dataset[index]
            if data.target_id not in self.target_to_indices:
                self.target_to_indices[data.target_id] = []
            self.target_to_indices[data.target_id].append(index)
            
        print(f'[{self.__class__.__name__}] Dataset ready with {len(self.dataset)} structures from {len(self.target_to_indices)} targets')

    def _process(self):
        print(f'Processing complex dataset from {self.pdb_dir} with ILDDT scores...')
        pdb_files = []
        json_files = []
        
        # Collect all PDB files and their corresponding JSON files
        for pdb_file in os.listdir(self.pdb_dir):
            if pdb_file.endswith('.pdb'):
                pdb_name = pdb_file[:-4]
                json_file = os.path.join(self.ilddt_dir, f"{pdb_name}.json")
                if os.path.exists(json_file):
                    pdb_files.append(os.path.join(self.pdb_dir, pdb_file))
                    json_files.append(json_file)
        
        print(f"Found {len(pdb_files)} PDB files with corresponding JSON files")
        
        # Process files with error handling
        dataset = []
        errors = []
        
        for pdb_file, json_file in tqdm(zip(pdb_files, json_files), desc='Preprocessing', total=len(pdb_files)):
            try:
                data = _process_complex_pdb(pdb_file, json_file)
                dataset.append(data)
            except Exception as e:
                errors.append((pdb_file, str(e)))
                print(f"Error processing {os.path.basename(pdb_file)}: {e}")
        
        # Report errors
        if errors:
            print(f"\nEncountered {len(errors)} errors while processing files:")
            for pdb_file, error in errors:
                print(f"- {os.path.basename(pdb_file)}: {error}")
        
        print(f"Successfully processed {len(dataset)} out of {len(pdb_files)} files")
        
        # Create cache directory if it doesn't exist
        cache_dir = os.path.dirname(self.cache_path)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Save processed dataset to cache
        print(f"Saving processed dataset to cache: {self.cache_path}")
        try:
            torch.save(dataset, self.cache_path)
            print(f"✓ Successfully saved {len(dataset)} processed structures to cache")
            
            # Verify the cache was saved correctly
            cache_size = os.path.getsize(self.cache_path) / (1024 * 1024)  # Size in MB
            print(f"✓ Cache file size: {cache_size:.2f} MB")
            
        except Exception as e:
            print(f"✗ Error saving cache: {e}")
            print(f"  Cache path: {self.cache_path}")
            print(f"  Cache directory exists: {os.path.exists(cache_dir)}")
            print(f"  Cache directory writable: {os.access(cache_dir, os.W_OK)}")
            # Continue without cache if save fails
            
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index].clone()
        dihedrals = _dihedrals(pos_N=data.pos_N, pos_CA=data.pos_CA, pos_C=data.pos_C)
        sidechains = _sidechains(pos_N=data.pos_N, pos_CA=data.pos_CA, pos_C=data.pos_C)
        onehot = F.one_hot(data.seq, 20)
        chain_encoding = torch.zeros(data.num_nodes, 2)
        chain_encoding[:data.chain_lengths[0], 0] = 1
        chain_encoding[data.chain_lengths[0]:, 1] = 1
        norm_pos = data.norm_pos.unsqueeze(-1)
        interface_indices = torch.nonzero(data.interface_mask).squeeze(-1)
        min_interface_residues = 5
        if interface_indices.size(0) < min_interface_residues:
            print(f"Warning: Only {interface_indices.size(0)} interface residues found for {data.decoy_id}, using all residues")
            interface_indices = torch.arange(data.num_nodes, device=data.pos_CA.device)
        orientations = _orientations_interface_aware(
            pos_CA=data.pos_CA[interface_indices],
            chain_indicators=chain_encoding[interface_indices]
        )
        # --- Add secondary structure feature ---
        pdb_path = None
        for k in self.ss_dict:
            if data.decoy_id in k:
                pdb_path = k
                break
        
        if pdb_path is None:
            print(f"Warning: No SS entry for {data.decoy_id}, using all 'X' (unknown)")
            ss_seq = ['X'] * data.num_nodes
        else:
            ss_seq = self.ss_dict[pdb_path]
            
            # Handle SS length mismatch by padding or truncating
            if len(ss_seq) != data.num_nodes:
                print(f"SS length mismatch for {data.decoy_id}: ss_seq={len(ss_seq)}, num_nodes={data.num_nodes}")
                if len(ss_seq) < data.num_nodes:
                    # Pad with 'X' (unknown secondary structure)
                    ss_seq = ss_seq + ['X'] * (data.num_nodes - len(ss_seq))
                    print(f"  Padded SS sequence with {data.num_nodes - len(ss_seq)} 'X' characters")
                else:
                    # Truncate to match protein length
                    ss_seq = ss_seq[:data.num_nodes]
                    print(f"  Truncated SS sequence from {len(self.ss_dict[pdb_path])} to {data.num_nodes}")
        
        # Ensure ss_seq is exactly the right length
        if len(ss_seq) != data.num_nodes:
            print(f"Error: SS sequence length still doesn't match for {data.decoy_id}. Creating fallback.")
            ss_seq = ['X'] * data.num_nodes
        
        ss_onehot = torch.tensor(ss_to_onehot(ss_seq), dtype=torch.float32)
        
        # --- Add RSA feature ---
        rsa_file = os.path.join(self.rsa_dir, f"{data.decoy_id}.txt")
        rsa_dict = parse_rsa_file(rsa_file)
        # Map residue_ids to RSA values
        # Get all residue_ids for this structure
        residue_ids = []
        for c in data.chain_lengths:
            residue_ids.extend(range(1, c+1))
        rsa_vals = torch.tensor([rsa_dict.get(rid, 0.0) for rid in residue_ids], dtype=torch.float32).unsqueeze(-1)
        # --- Combine all features ---
        all_node_s = torch.cat([
            dihedrals[interface_indices],
            onehot[interface_indices],
            chain_encoding[interface_indices],
            norm_pos[interface_indices],
            ss_onehot[interface_indices],
            rsa_vals[interface_indices]
        ], dim=-1)
        all_node_v = torch.cat([orientations, sidechains[interface_indices].unsqueeze(1)], dim=1)
        interface_data = Data(
            pos_N=data.pos_N[interface_indices],
            pos_CA=data.pos_CA[interface_indices],
            pos_C=data.pos_C[interface_indices],
            pos_O=data.pos_O[interface_indices],
            seq=data.seq[interface_indices],
            num_nodes=interface_indices.size(0),
            node_s=all_node_s,
            node_v=all_node_v,
            target_id=data.target_id,
            decoy_id=data.decoy_id,
            chain_indicators=chain_encoding[interface_indices],
            original_indices=interface_indices
        )
        
        # Add ILDDT score
        interface_data.ilddt = data.ilddt
            
        return interface_data


class ComplexPairBatchSampler(BatchSampler):
    """Batch sampler that creates pairs of complex structures for training."""

    def __init__(self, target_to_indices, batch_size=2):
        assert batch_size % 2 == 0
        self.batch_size = batch_size
        self.target_to_indices = target_to_indices
        self.targets = []
        self.counts = []
        for k, v in self.target_to_indices.items():
            if len(v) < 2: continue
            self.targets.append(k)
            self.counts.append(len(v))

        self.counts = np.array(self.counts)
        self.total = self.counts.sum()

    def __iter__(self):
        for _ in range(len(self)):
            targets_selected = np.random.choice(
                self.targets, 
                self.batch_size//2, 
                replace=False, 
                p=(self.counts/self.total)
            )
            idx = []
            for target in targets_selected:
                idx += list(np.random.choice(self.target_to_indices[target], 2, replace=False))
            yield idx

    def __len__(self):
        return self.total // self.batch_size


def interface_focused_collate(batch):
    """Custom collate function for interface data.
    Since the dataset already returns interface-only data, this function
    simply batches the data objects together.
    
    Args:
        batch: List of Data objects to be collated
        
    Returns:
        Batched Data object with interface residues
    """
    # Batch the interface-only data directly
    return Batch.from_data_list(batch)


def min_size_pair_collate(batch):
    """Custom collate function for interface data that handles pairs of structures.
    For pairs of structures from the same target, it ensures they have the same number
    of nodes by truncating to the smaller size.
    
    Args:
        batch: List of Data objects to be collated. Assumed to be in pairs
        from the same target.
        
    Returns:
        Batched Data object with interface residues
    """
    # Process pairs of structures
    processed_batch = []
    
    # Process each pair separately
    for i in range(0, len(batch), 2):
        if i+1 >= len(batch):  # Handle odd number of samples (shouldn't happen)
            processed_batch.append(batch[i])
            continue
            
        # Get the pair of structures
        data1, data2 = batch[i], batch[i+1]
        
        # Verify they are from the same target
        if data1.target_id != data2.target_id:
            print(f"Warning: Structures {data1.decoy_id} and {data2.decoy_id} are not from the same target")
            processed_batch.extend([data1, data2])
            continue
            
        # Find the minimum size between the two structures
        min_size = min(data1.num_nodes, data2.num_nodes)
        
        # Truncate the larger structure if needed
        if data1.num_nodes > min_size:
            # Create a new Data object with only the first min_size nodes
            truncated_data1 = Data(
                pos_N=data1.pos_N[:min_size],
                pos_CA=data1.pos_CA[:min_size],
                pos_C=data1.pos_C[:min_size],
                pos_O=data1.pos_O[:min_size],
                seq=data1.seq[:min_size] if hasattr(data1, 'seq') else None,
                num_nodes=min_size,
                node_s=data1.node_s[:min_size],
                node_v=data1.node_v[:min_size],
                target_id=data1.target_id,
                decoy_id=data1.decoy_id,
                chain_indicators=data1.chain_indicators[:min_size] if hasattr(data1, 'chain_indicators') else None,
                original_indices=data1.original_indices[:min_size] if hasattr(data1, 'original_indices') else None
            )
            # Preserve ILDDT score
            if hasattr(data1, 'ilddt'):
                truncated_data1.ilddt = data1.ilddt
            data1 = truncated_data1
            
        if data2.num_nodes > min_size:
            # Create a new Data object with only the first min_size nodes
            truncated_data2 = Data(
                pos_N=data2.pos_N[:min_size],
                pos_CA=data2.pos_CA[:min_size],
                pos_C=data2.pos_C[:min_size],
                pos_O=data2.pos_O[:min_size],
                seq=data2.seq[:min_size] if hasattr(data2, 'seq') else None,
                num_nodes=min_size,
                node_s=data2.node_s[:min_size],
                node_v=data2.node_v[:min_size],
                target_id=data2.target_id,
                decoy_id=data2.decoy_id,
                chain_indicators=data2.chain_indicators[:min_size] if hasattr(data2, 'chain_indicators') else None,
                original_indices=data2.original_indices[:min_size] if hasattr(data2, 'original_indices') else None
            )
            # Preserve ILDDT score
            if hasattr(data2, 'ilddt'):
                truncated_data2.ilddt = data2.ilddt
            data2 = truncated_data2
            
        processed_batch.extend([data1, data2])
        
    # Batch the processed data
    return Batch.from_data_list(processed_batch)


if __name__ == "__main__":
    # Test script
    base_dir = '/home/grads/xinyu0110/OAGNN_dataset2'
    
    # Test dataset with ILDDT scores
    print("Testing interface dataset with ILDDT scores...")
    ilddt_dataset = ComplexInterfaceDataset(
        pdb_dir=os.path.join(base_dir, 'pdb_casp16_5QA_final'),
        ilddt_dir=os.path.join(base_dir, 'ilddt_casp16_5QA_final'),
        ss_file=os.path.join(base_dir, 'secondary_structure/SS_casp16_5QA_final.result'),
        rsa_dir=os.path.join(base_dir, 'rsasa_casp16_5QA_final')
    )
    print(f"ILDDT dataset size: {len(ilddt_dataset)}")
    
    # Test a sample from the dataset
    if len(ilddt_dataset) > 0:
        sample_idx = 0
        
        # ILDDT sample
        ilddt_sample = ilddt_dataset[sample_idx]
        print(f"\nILDDT sample info:")
        print(f"- Decoy ID: {ilddt_sample.decoy_id}")
        print(f"- Number of interface nodes: {ilddt_sample.num_nodes}")
        print(f"- ILDDT score: {ilddt_sample.ilddt.item():.4f}")
        print(f"- Node features shape: {ilddt_sample.node_s.shape}")
        print(f"- Vector features shape: {ilddt_sample.node_v.shape}")
        
        # Test batching
        print(f"\nTesting batch collation...")
        ilddt_batch = interface_focused_collate([ilddt_sample])
        
        print(f"- Batch has ILDDT scores: {hasattr(ilddt_batch, 'ilddt')}")
        print(f"- Batch ILDDT score: {ilddt_batch.ilddt.item():.4f}")
        
        print(f"\n✓ Interface dataset successfully implemented!")
        print(f"  - Loads ILDDT scores from JSON files")
        print(f"  - Extracts interface residues with secondary structure and RSA features")
        print(f"  - Node features include: dihedrals, one-hot AA, chain encoding, position, SS, RSA")
        print(f"  - Vector features include: orientations and sidechains")




    

