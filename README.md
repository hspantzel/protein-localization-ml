# Protein Localization ML

Predicting human protein subcellular localization from amino-acid sequences using classical machine learning and pretrained protein language models.

## Project Goal

Proteins perform different functions depending partly on where they are located within a cell. The goal of this project is to predict one or more subcellular locations from a protein's amino-acid sequence.

This project will compare two approaches:

1. **Baseline model:** amino-acid composition and other sequence-derived features with classical machine learning.
2. **Advanced model:** pretrained ESM-2 protein embeddings with a neural-network classifier.

Because a protein may occur in more than one cellular compartment, this will be treated as a multi-label classification problem.

## Planned Pipeline

Protein sequence  
→ preprocessing  
→ sequence features / ESM-2 embeddings  
→ classifier  
→ predicted cellular locations

Example classes may include:

- Nucleus
- Cytoplasm
- Mitochondria
- Plasma membrane
- Endoplasmic reticulum
- Golgi apparatus

## Evaluation

Models will be compared using:

- Micro F1 score
- Macro F1 score
- Precision
- Recall
- Per-class performance
- Precision-recall curves

## Repository Structure

```text
protein-localization-ml/
├── data/
├── figures/
├── models/
├── notebooks/
├── src/
├── README.md
├── requirements.txt
└── .gitignore