# ORIGAMI


>Orientation-Aware Graph Neural Network for Assessing Multimeric Interfaces of Protein Complex Structures



## Installation

### With Anaconda
1. **Clone the repository**
   ```bash
   git clone https://github.com/Bhattacharya-Lab/ORIGAMI.git
   cd ORIGAMI
   ```
   Replace `<your-user>` with the GitHub organisation or username that will host ORIGAMI.

2. **Set up the environment**
   ```bash
   conda env create -f origami_environment.yml
   conda activate origami
   ```
   The YAML ships with CUDA-enabled builds; feel free to slim it down for CPU-only deployments.

3. **Install PyRosetta (optional, for SASA)**
   ```bash
   pip install pyrosetta-2024
   ```
   PyRosetta requires an academic licence. See the troubleshooting section for alternatives if it is unavailable.

4. **Install the command-line tool**
   ```bash
   pip install -e .
   ```
   Editable mode exposes the `origami-preprocess` entry point and keeps local edits live.



## Usage

### Feature extraction
```bash
# Single PDB
origami-preprocess -i protein.pdb -o features/protein

# Directory of PDBs
origami-preprocess -i path/to/pdb_dir -o features/corpus
```
Each `.pt` file contains the residue sequence, backbone coordinates, DSSP string + one-hot encoding, and relative SASA values.

### Inference helper
```bash
python data_preprocess.py -i protein.pdb -o features/protein --verbose
```
Running the script directly prints progress, writes features to `<output>/<pdb_name>_features.pt`, and reports residue counts per chain.

## Pretrained Models

### Quick evaluation
```bash
python models/psr/train_ddp_0719.py \
    pretrained_models/dis24.yml \
    --test \
    --test_dataset test1 \
    --resume pretrained_models/checkpoints/best.pt
```
- `--test_dataset` accepts `all`, `test1`, `test2`, `test3`, `casp16_5qa` or any custom split defined in the config.
- Metrics (Spearman, Kendall, Pearson, AUC, MAE) stream to stdout and TensorBoard.

### Evaluate on a custom dataset
1. **Organise decoys**
   - Place PDBs under `data/my_dataset/pdb_custom/`.
   - Filenames should be unique per target (e.g. `T0001TS001_1.pdb`).

2. **Provide ground-truth JSON**
   - For each decoy create `data/my_dataset/ilddt_custom/<decoy>.json` with keys `target`, `decoy`, `ilddt`.
   - Stub values are acceptable if you only require predictions.

3. **Generate DSSP + SASA inputs**
   ```bash
   origami-preprocess -i data/my_dataset/pdb_custom -o features/my_dataset
   ```
   - Collect DSSP strings into `secondary_structure/SS_custom.result` with `"<SS> <relative/path/to/pdb>"` per line.
   - Convert per-residue SASA to `rsasa_custom/<decoy>.txt` (one float per residue).

4. **Match expected layout**
   ```
   data/my_dataset/
   ├── pdb_custom/
   ├── ilddt_custom/
   ├── rsasa_custom/
   └── secondary_structure/
       └── SS_custom.result
   ```

5. **Configure evaluation**
   - Copy `pretrained_models/dis24.yml` to `configs/my_custom_eval.yml`.
   - Update `data.root` and the `test1_*` paths to point at your folders.

6. **Run scoring**
   ```bash
   python models/psr/train_ddp_0719.py \
       configs/my_custom_eval.yml \
       --test \
       --test_dataset test1 \
       --resume pretrained_models/checkpoints/best.pt \
       --logdir runs/my_dataset_eval
   ```
   - Predictions land in `runs/my_dataset_eval/test1_predictions.csv`.
   - Add `--device cpu` when GPUs are unavailable (expect slower runs).

7. **Interpret outputs**
   - Sort by `pred` to rank decoys per target.
   - TensorBoard (if enabled) stores correlation curves and loss metrics.

For more background, see `pretrained_models/README.md`.

## Training

### Single-GPU run
```bash
python models/psr/train_ddp_0719.py configs/dis24.yml --logdir runs/dis24_baseline
```

### Distributed (multi-GPU)
```bash
torchrun --nproc_per_node=4 models/psr/train_ddp_0719.py \
    configs/dis24.yml --logdir runs/dis24_ddp
```

### Useful flags
```
--resume PATH          Resume from checkpoint (.pt)
--overwrite            Reuse logdir when resuming
--test                 Skip training, run evaluation only
--tag STRING           Append experiment tag to logdir
--device cpu           Force CPU execution
```

Configuration examples live under `models/psr/configs/`. A minimal template:
```yaml
data:
  root: /path/to/dataset
  train_pdb_dir: pdb_train
  train_ilddt_dir: ilddt_train
  val_pdb_dir: pdb_val
  val_ilddt_dir: ilddt_val
  train_batch_size: 8
  val_batch_size: 8

model:
  num_layers: 6
  node_hid_dims: [128, 32]
  edge_hid_dims: [64, 16]

train:
  seed: 42
  max_epochs: 200
  optimizer:
    type: adam
    lr: 1.0e-5
```

## Dataset Preparation

1. **Raw structures**: PDB files per decoy with continuous residue numbering per chain.
2. **Labels**: JSON files containing interface lDDT or your target metric.
3. **Secondary structure**: Aggregate DSSP output in `secondary_structure/SS_<split>.result` (one entry per decoy).
4. **SASA text files**: Relative SASA per residue stored as `<decoy>.txt`.
5. **Cache**: Processed datasets are cached under `/home/grads/xinyu0110/OAGNN_dataset2/data_24` by default; override via the dataset constructor if needed.

Refer to the curated CASP datasets in `origami_datasets/` for full examples.

## Testing & QA
- `pytest tests/` – smoke tests for importability (PyRosetta skipped automatically).
- `python -m torch.distributed.run --nproc_per_node=2 ... --test` – quickly validate distributed inference.
- `tensorboard --logdir runs` – inspect training curves and per-target correlations.

## Project Structure
```
ORIGAMI/
├── data_preprocess.py           # Feature extraction CLI/utility
├── origami_environment.yml      # Conda environment definition
├── pretrained_models/           # Checkpoints + evaluation configs
├── models/                      # Training scripts, datasets, utilities
├── modules/                     # Neural network building blocks
├── utils/                       # Shared helpers (logging, schedulers)
└── tests/                       # Smoke tests
```

## Troubleshooting
- **PyRosetta unavailable**: Install FreeSASA (`pip install freesasa`) and adjust the preprocessing script to skip PyRosetta-dependent features.
- **CUDA OOM**: Lower `train_batch_size`, reduce the number of layers, or enable mixed precision (`torch.cuda.amp`).
- **Slow preprocessing**: Use the cached dataset files produced in `/data_24` or lower `preproc_n_jobs` when compute resources are limited.
- **Missing DSSP entries**: Ensure the DSSP result file contains an entry per decoy with matching basename.

## Citation
If you use ORIGAMI in your research, please cite:
```

```

## License
```
MIT License

Copyright (c) 2025 ORIGAMI Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Contact
For questions or collaboration requests, reach out to the maintainers:
- Xinyu Wang — Virginia Tech — `xinyu0110@vt.edu`


