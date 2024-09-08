# Sentiment Analysis Model

This repository contains models for sentiment analysis, including both traditional machine learning and deep learning approaches. The project involves preprocessing text data, training models to classify sentiment as positive or negative, and evaluating their performance.

## Table of Contents

- [Overview](#overview)
- [Model Details](#model-details)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Training](#training)
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

## Training

The training process involved fine-tuning a pre-trained Hugging Face model on the English-Hindi translation dataset.

### Steps:
1. **Data Preprocessing:**
   - Clean and preprocess the text data (tokenization, lemmatization).
   - Split the data into training and testing sets.
   
2. **Training Logistic Regression:**
   - Use TF-IDF vectorization to convert text data into numerical features.
   - Train a Logistic Regression model using the processed data.

3. **Training LSTM:**
   - Prepare sequential data for the LSTM model.
   - Train the LSTM model to classify sentiment based on the text sequences.
   
4. **Model Saving:**
   - Save the trained models and vectorizers for later use.

## Results 
The models achieved satisfactory results in classifying sentiment with notable accuracy of 88.25% with Logistics Regression and TFIDF embedding. The Logistic Regression model performed well with TF-IDF features, while the LSTM model demonstrated robust performance by capturing sequential patterns in the data.


