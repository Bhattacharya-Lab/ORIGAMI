# ORIGAMI


Orientation-Aware Graph Neural Network for Assessing Multimeric Interfaces of Protein Complex Structures



## Installation

The project requires several ML-specific libraries, it's easier to setup with Anaconda:
1. **Clone the repository**
   ```bash
   git clone https://github.com/Bhattacharya-Lab/ORIGAMI.git
   cd ORIGAMI
   ```
  
2. **Set up the environment**
   ```bash
   conda env create -f origami_environment.yml
   conda activate origami
   ```

3. **Install**
   ```bash
   pip install -e .
   ```
  


## Usage
We provide a command-line interface for ORIGAMI that can easily be used to score protein-protein complexes. The command-line interface can be used as follows:


```bash
$ origami-score -h
usage: origami-score [-h] [--checkpoint CHECKPOINT] [--config CONFIG] [--device DEVICE] [--output OUTPUT] [--json] pdb

Score a protein-protein complex using ORIGAMI.

positional arguments:
  pdb                   Path to the input PDB file containing two chains.

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint CHECKPOINT
                        Path to the pretrained checkpoint (.pt).
  --config CONFIG       YAML configuration used during training.
  --device DEVICE       Torch device identifier (e.g. cuda, cuda:0, cpu).
  --output OUTPUT       Optional path to write the prediction score as JSON.
  --json                Print the prediction as JSON instead of a plain float.
```

Example, score the H1245TS028_4.pdb complex (in ORIGAMI/example)
```bash
$ conda activate origami
(origami) $ origami-score ORIGAMI/example/H1245TS028_4.pdb
```


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


