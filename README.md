# DetEC: Deep Learning Model for Enzyme Function Prediction

DetEC is a deep learning model designed for predicting enzyme functions from protein sequences and structural information. This repository contains the model implementation, training scripts, and evaluation tools.
![figer1](https://github.com/user-attachments/assets/ba5d67e5-28a6-44b1-8897-88cdf7fd1c42)

## Table of Contents
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Model Download](#model-download)
- [Contact](#contact)



## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/JunjunWang111/DetEC.git
   cd DetEC
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

The dataset files are not included in this repository. Please download them from the Releases section of this repository:

[Download Dataset from Releases](https://github.com/JunjunWang111/DetEC/releases)  # Download from GitHub Releases
After downloading, place the dataset files in the `data/` directory:
- `split100.csv` (Training set)
- `New-392.csv` (Test set)
- `Price-149.csv` (Test set)
- `Temporal-Val.csv` (Validation set)

## Training

To train the model, run the training script:

```bash
python train.py
```

The training process will use `split100.csv` as the training set and `Temporal-Val.csv` as the validation set. The trained model will be saved in the `checkpoints/` directory.

## Evaluation

To evaluate the model on the test sets, run the evaluation script:

```bash
python simple_evaluate.py
```

This will evaluate the model on both `New-392.csv` and `Price-149.csv` test sets and generate an evaluation report.

## Results

The model achieves the following performance on the test sets:

| Dataset   | Precision | Recall  | F1 Score | Accuracy |
|-----------|-----------|---------|----------|----------|
| New-392   | 0.6561    | 0.6588  | 0.6647   | 0.8805   |
| Price-149 | 0.6379    | 0.6118  | 0.5832   | 0.8643   |

## Model Download

Pretrained models are available for download from the Releases section of this repository:

[Download Pretrained Model from Releases](https://github.com/JunjunWang111/DetEC/releases)  # Download from GitHub Releases

After downloading, place the model file in the `checkpoints/` directory.

## Contact

For questions or issues, please contact:

- GitHub: [JunjunWang111](https://github.com/JunjunWang111)
- Email: 2504462064@qq.com
