# Quick Start Guide

Get started with ORIGAMI in 5 minutes!

## 🚀 Super Quick Start with Pretrained Model

If you just want to test the pretrained model on your data:

```bash
# 1. Setup environment
conda env create -f origami_environment.yml
conda activate origami

# 2. Prepare your data (extract features)
python data_preprocess.py -i your_pdb_dir/ -o features/

# 3. Test with pretrained model
python models/psr/train_ddp_0719.py \
    pretrained_models/dis24.yml \
    --test \
    --resume pretrained_models/checkpoints/best.pt
```

That's it! The pretrained model will evaluate your structures.

---

## Full Guide: Training from Scratch

## 1. Setup Environment (5 min)

```bash
# Clone/navigate to the project
cd /home/grads/xinyu0110/ORIGAMI

# Create and activate conda environment
conda env create -f origami_environment.yml
conda activate origami

# Optional: Install PyRosetta for SASA calculation
pip install pyrosetta-2024
```

## 2. Prepare Your Data (10 min)

### Option A: Extract Features from PDB Files

```bash
# Process a single PDB
python data_preprocess.py -i your_protein.pdb -o features/

# Process a directory
python data_preprocess.py -i pdb_directory/ -o features/
```

### Option B: Use Pre-extracted Features

If you already have `.pt` feature files, skip to step 3.

## 3. Configure Training

Edit a config file in `models/psr/configs/` or create your own:

```yaml
data:
  base_dir: /path/to/your/data
  train_pdb_dir: pdb_train
  train_ilddt_dir: ilddt_train
  ss_file: secondary_structure/SS_train.result
  rsa_dir: rsasa_train

model:
  num_layers: 6
  node_dim_scalar: 256
  node_dim_vector: 64

train:
  batch_size: 4
  num_epochs: 100
  learning_rate: 5.0e-4
  seed: 2020
```

## 4. Train the Model

### Single GPU

```bash
python models/psr/train_ddp_0719.py models/psr/configs/your_config.yml
```

### Multi-GPU (Recommended)

```bash
# 4 GPUs
torchrun --nproc_per_node=4 models/psr/train_ddp_0719.py models/psr/configs/your_config.yml

# 8 GPUs
torchrun --nproc_per_node=8 models/psr/train_ddp_0719.py models/psr/configs/your_config.yml
```

## 5. Monitor Training

Training logs are saved in the `logs/` directory:

```bash
# View TensorBoard
tensorboard --logdir logs/

# Check training progress
tail -f logs/YOUR_EXPERIMENT/train.log
```

## 6. Test the Model

```bash
# Test on all datasets
python models/psr/train_ddp_0719.py your_config.yml --test --resume logs/YOUR_EXPERIMENT/checkpoints/best.pt

# Test on specific dataset
python models/psr/train_ddp_0719.py your_config.yml --test --test_dataset casp16_5qa --resume logs/YOUR_EXPERIMENT/checkpoints/best.pt
```

## Common Commands

### Data Processing

```bash
# Extract features with verbose output
python data_preprocess.py -i pdb_dir/ -o features/ -v

# Process only specific PDB files
python data_preprocess.py -i specific_protein.pdb -o output/
```

### Training

```bash
# Train with custom log directory
python models/psr/train_ddp_0719.py config.yml --logdir my_experiments/

# Resume from checkpoint
python models/psr/train_ddp_0719.py config.yml --resume logs/exp1/checkpoints/100.pt

# Debug mode (single epoch, no saving)
python models/psr/train_ddp_0719.py config.yml --debug
```

### Testing

```bash
# Test all datasets
python models/psr/train_ddp_0719.py config.yml --test --test_dataset all --resume best.pt

# Test individual datasets
python models/psr/train_ddp_0719.py config.yml --test --test_dataset test1 --resume best.pt
python models/psr/train_ddp_0719.py config.yml --test --test_dataset casp16_5qa --resume best.pt
```

## Directory Structure After Setup

```
ORIGAMI/
├── data/                       # Your data directory
│   ├── pdb_train/
│   ├── ilddt_train/
│   ├── rsasa_train/
│   └── secondary_structure/
├── features/                   # Extracted features (.pt files)
├── logs/                       # Training logs and checkpoints
│   └── experiment_name/
│       ├── train.log
│       ├── config.yml
│       └── checkpoints/
└── models/psr/configs/        # Configuration files
```

## Troubleshooting

### Issue: CUDA out of memory

**Solution**: Reduce batch size in config file:
```yaml
train:
  batch_size: 2  # Reduce from 4 to 2
```

### Issue: PyRosetta not installed

**Solution**: Either install PyRosetta or use FreeSASA:
```bash
pip install freesasa
```

### Issue: Cannot find data files

**Solution**: Check paths in config file match your data structure:
```bash
ls -la data/pdb_train/
ls -la data/ilddt_train/
```

### Issue: Import errors

**Solution**: Make sure you're in the ORIGAMI directory and environment is activated:
```bash
cd /home/grads/xinyu0110/ORIGAMI
conda activate origami
```

## Next Steps

1. ✅ Read the full [README.md](README.md) for detailed documentation
2. ✅ Explore example configs in `models/psr/configs/`
3. ✅ Modify hyperparameters for your use case
4. ✅ Monitor training with TensorBoard
5. ✅ Evaluate on your test sets

## Need Help?

- Check [README.md](README.md) for detailed documentation
- Review example configs in `models/psr/configs/`
- Examine the training logs in `logs/`

Happy training! 🚀

