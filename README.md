# ORIGAMI - Protein Structure Quality Assessment

A deep learning framework for protein structure quality assessment combining secondary structure, SASA, and graph neural networks.

## Project Structure

```
ORIGAMI/
├── data_preprocess.py           # PDB feature extraction (SS, SASA, coords)
├── origami_environment.yml      # Conda environment configuration
├── pretrained_models/           # Pretrained model checkpoints
│   ├── checkpoints/
│   │   ├── 110.pt              # Epoch 110 checkpoint
│   │   └── best.pt             # Symlink to best model
│   ├── dis24.yml               # Training configuration
│   └── README.md               # Pretrained model documentation
├── models/
│   └── psr/
│       ├── train_ddp_0719.py   # Main training script (DDP)
│       ├── datasets_24.py       # Dataset loader
│       ├── models.py            # PSR Network model
│       ├── utils.py             # PSR utilities
│       └── configs/             # Training configurations
├── modules/                     # Neural network modules
│   ├── gconv.py                # Graph convolution layers
│   ├── perceptron.py           # Vector perceptron
│   ├── geometric.py            # Geometric operations
│   └── ...
└── utils/                       # Utility functions
    ├── misc.py                 # General utilities
    └── train.py                # Training utilities
```

## Installation

### 1. Create Conda Environment

```bash
conda env create -f origami_environment.yml
conda activate origami
```

### 2. Install PyRosetta (Optional)

PyRosetta is required for SASA calculation. It requires an academic license:

```bash
pip install pyrosetta-2024
```

Or visit: https://www.pyrosetta.org/

## Data Preprocessing

Extract features from PDB files:

```bash
# Process a single PDB file
python data_preprocess.py -i protein.pdb -o output_dir

# Process a directory of PDB files
python data_preprocess.py -i pdb_folder -o features_folder
```

### Output Format

Each PDB generates a `.pt` file containing:
- **Secondary Structure**: DSSP-predicted (H/E/C) + one-hot encoding
- **Relative SASA**: Per-residue solvent accessibility
- **Coordinates**: N, CA, C, O backbone atoms (shape: L×4×3)
- **Sequence**: Amino acid sequence and chain information

## Training

### Basic Training

```bash
python models/psr/train_ddp_0719.py models/psr/configs/0908/your_config.yml
```

### Distributed Training (Multi-GPU)

```bash
torchrun --nproc_per_node=4 models/psr/train_ddp_0719.py models/psr/configs/0908/your_config.yml
```

### Training Options

```bash
python models/psr/train_ddp_0719.py CONFIG_FILE [OPTIONS]

Options:
  --logdir DIR          Log directory
  --tag TAG             Experiment tag
  --device DEVICE       Device (cuda/cpu)
  --resume PATH         Resume from checkpoint
  --test                Run testing only
  --test_dataset NAME   Test dataset: all/test1/test2/test3/casp16_5qa
  --overwrite           Overwrite existing logs
  --debug               Debug mode
```

## Configuration

Training configurations are in `models/psr/configs/`. Example structure:

```yaml
data:
  base_dir: /path/to/data
  train_pdb_dir: pdb_train
  train_ilddt_dir: ilddt_train
  ss_file: secondary_structure/SS_train.result
  rsa_dir: rsasa_train
  
model:
  num_layers: 6
  node_dim_scalar: 256
  node_dim_vector: 64
  edge_dim: 32
  
train:
  batch_size: 4
  num_epochs: 100
  learning_rate: 5.0e-4
  weight_decay: 0.0
  seed: 2020
```

## Features

### Data Processing
- ✅ DSSP secondary structure calculation
- ✅ PyRosetta SASA computation
- ✅ Multi-chain support
- ✅ Interface residue detection
- ✅ Automatic feature caching

### Model Architecture
- ✅ Graph Neural Network (GNN) based
- ✅ Scalar-Vector features
- ✅ Edge features (RBF + positional embeddings)
- ✅ Chain-aware processing
- ✅ Interface-focused prediction

### Training
- ✅ Distributed Data Parallel (DDP)
- ✅ Mixed precision training
- ✅ Gradient clipping
- ✅ TensorBoard logging
- ✅ Checkpoint management
- ✅ Multiple test datasets

## Dependencies

### Core Requirements
- Python 3.8
- PyTorch 1.10.0 + CUDA 11.1
- PyTorch Geometric
- BioPython
- pydssp

### Optional
- PyRosetta (for SASA calculation)
- FreeSASA (alternative to PyRosetta)

See `origami_environment.yml` for complete dependency list.

## Pretrained Models

A pretrained model (epoch 110) is available in `pretrained_models/`:

### Quick Start with Pretrained Model

```bash
# Test on your dataset
python models/psr/train_ddp_0719.py \
    pretrained_models/dis24.yml \
    --test \
    --resume pretrained_models/checkpoints/best.pt \
    --test_dataset your_dataset
```

See `pretrained_models/README.md` for detailed usage instructions.

## Usage Examples

### Example 1: Feature Extraction

```python
import torch
from data_preprocess import extract_all_features
import pyrosetta

pyrosetta.init("-mute all")

# Extract features from a single PDB
features = extract_all_features("protein.pdb")

# Access features
print(f"Residues: {features['total_residues']}")
print(f"Secondary structure: {features['secondary_structure']}")
print(f"Coordinates shape: {features['chains'][0]['coords'].shape}")

# Save features
torch.save(features, "protein_features.pt")
```

### Example 2: Training

```bash
# Train on single GPU
python models/psr/train_ddp_0719.py models/psr/configs/0908/config.yml

# Train on 4 GPUs
torchrun --nproc_per_node=4 models/psr/train_ddp_0719.py models/psr/configs/0908/config.yml

# Resume training
python models/psr/train_ddp_0719.py models/psr/configs/0908/config.yml --resume logs/checkpoint.pt
```

### Example 3: Testing

```bash
# Test on all datasets
python models/psr/train_ddp_0719.py models/psr/configs/0908/config.yml --test --resume best_model.pt

# Test on specific dataset
python models/psr/train_ddp_0719.py models/psr/configs/0908/config.yml --test --test_dataset casp16_5qa --resume best_model.pt
```

## Metrics

The model reports:
- **Spearman Correlation**: Rank correlation with ground truth
- **Kendall's Tau**: Alternative rank correlation
- **Pearson Correlation**: Linear correlation
- **AUC**: Area under ROC curve
- **Loss**: Training/validation loss

## Troubleshooting

### PyRosetta Installation

If PyRosetta is not available, you can:
1. Use FreeSASA as an alternative: `pip install freesasa`
2. Skip SASA features (modify `data_preprocess.py`)
3. Use pre-computed SASA values

### CUDA Out of Memory

- Reduce `batch_size` in config
- Use gradient accumulation
- Enable mixed precision training
- Reduce model size (`num_layers`, `node_dim_scalar`)

### Data Loading Issues

Ensure your data directory structure matches:
```
data/
├── pdb_train/          # Training PDB files
├── ilddt_train/        # ILDDT JSON files
├── rsasa_train/        # SASA text files
└── secondary_structure/
    └── SS_train.result # Secondary structure file
```

## Citation

If you use this code, please cite:
```
[Add citation here]
```

## License

[Add license information]

## Contact

[Add contact information]

