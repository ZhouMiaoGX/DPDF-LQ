# DPDF-LQ

Code and supplementary material for our EMNLP 2025 paper *Dual-Path Dynamic Fusion with Learnable Query for Multimodal Sentiment Analysis* (DPDF-LQ).  
The paper can be found [here]().  
All source code is planned to be released after the EMNLP 2025 conference (scheduled for November 9).


---

## Environment

- Python 3.10 (Ubuntu 22.04)
- PyTorch 2.1.0
- CUDA 12.1

Install dependencies:
```bash
pip install -r requirements.txt
```


## Usage

### 1. Clone this repository
```bash
git clone https://github.com/your-username/DPDF-LQ.git
cd DPDF-LQ
```

### 2. Download the datasets
Download the CMU-MOSI and CMU-MOSEI datasets, and place them under the `datasets/` directory.  
(Refer to the original dataset instructions if needed.)

### 3. Run training (example on MOSI)
```bash
python train_acc2.py --config_file configs/mosi.yaml --gpu_id 0 --seed 1
```

### 4. View training logs
Training logs will be saved automatically. You can visualize them using TensorBoard:
```bash
tensorboard --logdir runs
```


## Notes

- The default configuration uses BERT-base for fair comparison with baselines and to reduce computation cost.
- The full source code will be released after EMNLP 2025 (November 9).
- Please refer to the paper [here]() for detailed descriptions of the model architecture and experiments.