# Importance Sampling — Reproduction Code

This repository contains the code used to run the experiments for the paper **"Evaluating Model Performance and Fairness without Labeled Representative Data"**.  
It provides all scripts needed to reproduce the numerical results, including data preprocessing, experiment execution, and evaluation.

The goal of this repository is to make all analyses transparent, reproducible, and easy to run.

---

## Overview

This codebase provides:

- Implementation of the importance sampling approach described in the paper  
- Scripts to run all experiments 
- Examples of preprocessing pipelines for text datasets  
- Instructions for reproducing the figures and tables in the paper  

---

## Installation

Clone the repository:

```bash
git clone https://github.com/anon68414/importance-sampling.git
cd importance-sampling
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data access

The preprocessed datasets used in this study are available for download from Google Drive:
[Download link](https://drive.google.com/drive/folders/1j-iNSmT1vs3kPHDdMNjHf0QKnSs3fyZZ?usp=share_link) 

After downloading, place the files in `./data/processed/` before running the notebook.

## Reproducing Preprocessing 

An examples of a preprocessing scripts are included in 
```bash 
src/preprocessing
```
If you want to reproduce the preprocessing yourself instead of downloading, run:
```bash
python src/preprocessing/preprocess_sentiment.py
```

Note:
The full preprocessing pipeline—especially for the computer vision datasets—consists of many steps and model calls.
We provide a high-level description below.

### Preprocessing Summary - Computer Vision Data 
The computer vision preprocessing pipeline includes face detection and cropping, image quality and contrast assessment, head pose estimation, skin color estimation, age and gender prediction, and identity embedding extraction. These explicit features are complemented by model-internal representations extracted from the final hidden layer of the models used in the experiments.

A detailed explanation of the explicit and model-internal features can be found in **Section 3.2.1** and **Section 3.2.2**of the paper.

### Preprocessing Summary - Text Data 
For the text datasets, we apply a lighter preprocessing pipeline that focuses on extracting task-relevant representations. We generate predicted labels and extract model-internal representations from the final hidden layer of the language model.

## Running Experiments 
To reproduce the experiments from the paper: 
```bash
notebooks/01_analysis_simulations.ipynb
```
Running the experiments will produce the figures and tables from the paper. 

## Citation 
