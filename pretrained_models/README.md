# Pretrained Models

This directory contains pretrained ORIGAMI models ready for inference or fine-tuning.

## Available Models

### Model: dis24_epoch110

**Location**: `checkpoints/110.pt` (also available as `checkpoints/best.pt`)

**Training Details**:
- **Training Date**: September 25, 2025
- **Epoch**: 110
- **Configuration**: `dis24.yml`
- **Source**: `/home/grads/xinyu0110/0925_replicate_0719_log/dis24_2025_09_25__12_59_40_replicate_0719_training/`
- **Model Size**: 52 MB

**Configuration File**: `dis24.yml`

## Usage

### Load Pretrained Model for Inference

```python
import torch
from models.psr.models import PSRNetwork

# Load checkpoint
checkpoint = torch.load('pretrained_models/checkpoints/best.pt', map_location='cuda')

# Create model
model = PSRNetwork(
    num_layers=6,
    node_dim_scalar=256,
    node_dim_vector=64,
    edge_dim=32
)

# Load weights
model.load_state_dict(checkpoint['model'])
model.eval()

print(f"Loaded model from epoch {checkpoint['epoch']}")
```

### Resume Training

```bash
python models/psr/train_ddp_0719.py \
    pretrained_models/dis24.yml \
    --resume pretrained_models/checkpoints/best.pt
```

### Test on Your Data

```bash
python models/psr/train_ddp_0719.py \
    pretrained_models/dis24.yml \
    --test \
    --resume pretrained_models/checkpoints/best.pt \
    --test_dataset your_dataset
```

## Model Architecture

The pretrained model uses PSR Network with:
- **Layers**: 6 graph convolution layers
- **Node Features**: Scalar (256-dim) + Vector (64-dim)
- **Edge Features**: 32-dim (RBF + positional embeddings)
- **Input Features**: 
  - Secondary Structure (DSSP, 3-dim one-hot)
  - Relative SASA (1-dim)
  - Dihedral angles (6-dim)
  - Amino acid type (20-dim one-hot)
  - Chain encoding (2-dim)
  - Position encoding (1-dim)

## Configuration

See `dis24.yml` for the complete training configuration including:
- Data paths
- Model hyperparameters
- Training settings (learning rate, batch size, etc.)
- Optimizer and scheduler configuration

## Performance

To evaluate the model performance on your dataset:

```bash
python models/psr/train_ddp_0719.py \
    pretrained_models/dis24.yml \
    --test \
    --resume pretrained_models/checkpoints/best.pt \
    --test_dataset all
```

This will report:
- Spearman correlation
- Kendall's tau
- Pearson correlation
- AUC (Area Under Curve)
- Loss metrics

## Fine-tuning

To fine-tune the model on your own dataset:

1. Prepare your data following the ORIGAMI data format
2. Create a new config file or modify `dis24.yml`
3. Run training with the `--resume` flag:

```bash
python models/psr/train_ddp_0719.py \
    your_config.yml \
    --resume pretrained_models/checkpoints/best.pt
```

The model will load the pretrained weights and continue training on your data.

## Quick Inference Example

```python
import torch
from models.psr.models import PSRNetwork
from models.psr.datasets_24 import ComplexInterfaceDataset
from torch_geometric.data import DataLoader

# Load model
checkpoint = torch.load('pretrained_models/checkpoints/best.pt')
model = PSRNetwork(
    num_layers=6,
    node_dim_scalar=256, 
    node_dim_vector=64,
    edge_dim=32
).cuda()
model.load_state_dict(checkpoint['model'])
model.eval()

# Load your data
dataset = ComplexInterfaceDataset(
    pdb_dir='your_pdb_dir',
    ilddt_dir='your_ilddt_dir',
    ss_file='your_ss_file',
    rsa_dir='your_rsa_dir'
)

loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Run inference
predictions = []
with torch.no_grad():
    for batch in loader:
        batch = batch.cuda()
        pred = model(batch)
        predictions.append(pred.cpu())

print(f"Generated {len(predictions)} predictions")
```

## Checkpoint Contents

The checkpoint file contains:
- `model`: Model state dictionary
- `optimizer`: Optimizer state
- `scheduler`: Learning rate scheduler state
- `epoch`: Training epoch number
- `config`: Training configuration

## Notes

- The model expects input features in the format described in `data_preprocess.py`
- Make sure to preprocess your PDB files using the data preprocessing pipeline
- For best results, use the same feature extraction process as during training

## Citation

If you use this pretrained model, please cite:
```
[Add citation information]
```

## Contact

For questions or issues with the pretrained model, please contact:
[Add contact information]

