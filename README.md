Live Demo: https://huggingface.co/spaces/Adinzlotyint/ESGAnalyze
Note: For optimal performance, it is advised to clone this repo and run app.py ("app" folder) locally on a high-end GPU.

This repository is a part of my master's thesis research project. Its purpose was to train and evaluate a multi-label classification model capable of analyzing Polish corporate sustainability reports based on selected ESRS standards.

## Problem description
Multi-label classification of long, unstructured documents in Polish language, under conditions of strong class imbalance and low data volume.

## Project Overview

This project focuses on fine-tuning the `sdadas/polish-longformer-base-4096` model for the specialized task of analyzing non-financial report content. It utilizes a dataset of 393 Polish corporate reports and their corresponding expert annotations.

## Implemented Features

- **Chunks tokenization**: Tokenization using a "sliding window" technique to manage long documents (4096 tokens).
- **Multi-Label Stratification**: The dataset is split using iterative stratification (`scikit-multilearn`) to preserve the distribution of label combinations across the train, validation, and test sets.
- **Threshold Optimization**: The evaluation strategy includes a step to find F1-score-maximizing decision thresholds for each category individually using the validation set.
- **Polish Language Model**: The project leverages a model pre-trained specifically on the Polish language to achieve better performance on local documents.