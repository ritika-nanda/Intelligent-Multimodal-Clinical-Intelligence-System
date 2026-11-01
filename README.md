# Intelligent-Multimodal-Clinical-Intelligence-System

A lightweight multimodal deep learning framework that combines **chest X-ray analysis**, **heart disease prediction**, and **clinical text classification** into a single patient risk assessment with an automated clinical report.

> **Disclaimer:** This is a research prototype built on public, de-identified datasets. It is **not** a medical device and must not be used for real clinical decisions.

## Overview

Most healthcare AI systems analyze only one type of data. This project processes three modalities in one pipeline and fuses their outputs:

| Module | Modality | Model | Task |
|---|---|---|---|
| 1 | Chest X-ray images | MobileNetV2 (transfer learning) | Pneumonia detection |
| 2 | Structured clinical data | Residual neural network | Heart disease prediction |
| 3 | Clinical transcriptions | CNN + Bidirectional LSTM | Medical specialty classification (10 classes) |
| Fusion | All of the above | Late-fusion aggregator | Low / Moderate / High risk + report |

![Architecture](figures/architecture.png)

## Key Features

- Unified pipeline for image, tabular, and text data
- Lightweight models that run without a dedicated GPU
- Corrected stratified 80:20 validation split for the chest X-ray data (the original validation set had only ~8 images per class)
- Engineered interaction features for heart disease prediction (13 → 18 features)
- Class-imbalance handling: focal loss and threshold optimization (X-ray), class weighting (heart), class capping and minority boosting (NLP)
- Late-fusion risk scoring with automated clinical report generation

## Results

| Module | Accuracy | AUC-ROC | Notes |
|---|---|---|---|
| MobileNetV2 (Pneumonia) | 85.1% | 0.9426 | Optimal threshold 0.386; pneumonia recall 0.94 |
| ResNet NN (Heart Disease) | 95.78% | 0.9885 | Train/val accuracy gap 0.01 |
| Bi-LSTM (Specialty, 10 classes) | 40.7% | N/A | Limited by class imbalance and overlapping terminology |

See the paper for full per-class metrics.

## Datasets

Datasets are **not** included in this repository. Download them from the original sources:

| Dataset | Samples | Link |
|---|---|---|
| Chest X-Ray Images (Pneumonia) | 5,856 | https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia |
| Heart Disease (UCI-derived, 1,025 records) | 1,025 | https://archive.ics.uci.edu/dataset/45/heart+disease |
| MTSamples (clinical transcriptions) | 3,630 (after filtering) | https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions |

Place them as follows:

```
data/
├── chest_xray/        # train/ val/ test/ folders
├── heart.csv
└── mtsamples.csv
```

## Repository Structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── Multimodal_clinical_system.ipynb
├── src/                # optional: xray.py, heart.py, nlp.py, fusion.py
├── models/             # trained weights (or link below)
├── figures/            # architecture, ROC curves, confusion matrices
└── docs/               # presentation slides
```

## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

1. Download the datasets and place them in `data/` (see above).
2. Open the notebook:
   ```bash
   jupyter notebook notebooks/Multimodal_clinical_system.ipynb
   ```
   or upload it to Google Colab.
3. Run the cells in order: preprocessing and EDA, training for each module, then fusion and report generation.

## Methodology Summary

- **Module 1:** images resized to 96×96×3 and normalized with MobileNetV2 preprocessing. Two-stage training (frozen feature extractor, then fine-tuning) with binary focal loss.
- **Module 2:** RobustScaler, five engineered interaction features, residual blocks with batch normalization and dropout, binary cross-entropy.
- **Module 3:** Keras tokenizer (vocab 12,000), sequences padded to 250 tokens, embedding → Conv → MaxPool → BiLSTM → dense output.
- **Fusion:** weighted combination of the three module scores. The weights are manually defined (see the paper, Fig. 1). Output is Low / Moderate / High risk plus a generated report.

## Limitations

- The three datasets come from different sources, so they do not represent the same patients.
- The NLP module has comparatively low accuracy.
- Fusion weights are hand-set, not learned.
- Validated on benchmark data only, not real patient data.
- No real-time deployment or clinical UI yet.

## Future Work

- Replace the BiLSTM with ClinicalBERT or BioBERT
- Learned fusion (meta-learning)
- Add ECG and pathology report modalities
- Explainability with Grad-CAM, SHAP, and attention visualization
- Web dashboard (Streamlit / Gradio / FastAPI)
- Federated learning for privacy-preserving training

## Authors

- Dr. Abhijit Shit (Supervisor), Centre for AI&ML, ITER, Siksha 'O' Anusandhan (Deemed to be University), Bhubaneswar
- Aditya Kumar
- Nandana Devi
- Ritika Nanda
- Barsha Vijayeta Das

## Citation

If you use this work, please cite:

```bibtex
@misc{multimodal_clinical_2026,
  title  = {Intelligent Multimodal Clinical Intelligence System},
  author = {Shit, Abhijit and Kumar, Aditya and Devi, Nandana and Nanda, Ritika and Das, Barsha Vijayeta},
  year   = {2026},
  note   = {Siksha 'O' Anusandhan University, Bhubaneswar}
}
```

## Acknowledgments

Thanks to the Centre for Artificial Intelligence and Machine Learning, ITER, SOA University, for resources and guidance.
