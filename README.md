# DPDF-LQ

Code and supplementary material for our EMNLP 2025 paper *Dual-Path Dynamic Fusion with Learnable Query for Multimodal Sentiment Analysis* (DPDF-LQ).  
The paper can be found [here](https://aclanthology.org/2025.emnlp-main.571/).  
All source code is planned to be released after the EMNLP 2025 conference (scheduled for November 9).
![DPDFLQArchitecture](https://github.com/user-attachments/assets/c3f99501-dc68-4786-9514-ca627d002258)


---

## Environment

- Python 3.10 (Ubuntu 22.04)
- PyTorch 2.1.0
- CUDA 12.1


## Usage

### 1. Clone this repository and install dependencies
```bash
git clone https://github.com/ZhouMiaoGX/DPDF-LQ.git
cd DPDF-LQ
pip install -r requirements.txt
```

### 2. Download the datasets
Download the **aligned versions** of the CMU-MOSI and CMU-MOSEI datasets, and place them under the `datasets/` directory.  
(Refer to the original dataset instructions if needed.)

### 3. Run training (example on MOSI)
```bash
python train.py --config_file configs/mosi.yaml --gpu_id 0
```

### 4. View training logs
Training logs will be saved automatically. You can visualize them using TensorBoard:
```bash
tensorboard --logdir runs
```


## Notes

- The default configuration uses BERT-base for fair comparison with baselines and to reduce computation cost.
- The full source code will be released after EMNLP 2025 (November 9).
- Please refer to the paper [here](https://aclanthology.org/2025.emnlp-main.571/) for detailed descriptions of the model architecture and experiments.
