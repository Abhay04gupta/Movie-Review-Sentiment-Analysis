# Sentiment Analysis Model

This repository contains models for sentiment analysis, including both traditional machine learning and deep learning approaches. The project involves preprocessing text data, training models to classify sentiment as positive or negative, and evaluating their performance.

## Table of Contents

- [Overview](#overview)
- [Model Details](#model-details)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)

## Overview

This project aims to analyze the sentiment of text data by leveraging machine learning and deep learning models. The pipeline involves:
1. Preprocessing text data to clean and prepare it for model training.
2. Training a Logistic Regression model and an LSTM model to classify sentiment.
3. Evaluating the performance of the trained models on test data.

## Model Details

- **Logistic Regression Model:** Trained using TF-IDF vectorization for feature extraction.
- **LSTM Model:** Deep learning model trained on sequential data to capture context and sentiment.
- **Frameworks:** Scikit-learn for Logistic Regression, TensorFlow/Keras for LSTM.

## Dataset

The models are trained and evaluated on a dataset containing text labeled with positive or negative sentiment. The dataset is preprocessed to remove noise, tokenize text, and apply lemmatization.

## Dependencies

Ensure you have the following dependencies installed:

- Python 3.8+
- Scikit-learn
- TensorFlow 2.4+
- NLTK
- numpy
- pandas
- matplotlib
- seaborn

You can install these dependencies using:

```bash
pip install -r requirements.txt

