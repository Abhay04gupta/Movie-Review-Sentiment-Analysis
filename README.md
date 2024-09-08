
# Sentiment Analysis

This project focuses on building a sentiment analysis model using both traditional machine learning and deep learning approaches. The dataset contains text data labeled with positive or negative sentiments. The project includes text preprocessing, model training (using Logistic Regression and LSTM), and model evaluation.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Preprocessing](#preprocessing)
- [Training](#training)
  - [Logistic Regression](#logistic-regression)
  - [LSTM](#lstm)
- [Evaluation](#evaluation)
- [Results](#results)
- [Requirements](#requirements)
- [License](#license)

## Project Structure

```plaintext
├── data
│   ├── train.csv         # Raw training data
│   ├── test.csv          # Raw testing data
│   ├── preprocessed.csv  # Preprocessed training data
├── models
│   ├── logistic_regression.pkl  # Logistic Regression model
│   ├── lstm_model.h5            # LSTM model
│   ├── tfidf_vectorizer.pkl     # TFIDF vectorizer
│   ├── tokenizer.pkl            # Tokenizer for LSTM
├── src
│   ├── preprocess.py     # Preprocessing script
│   ├── train.py          # Training script
│   ├── evaluate.py       # Evaluation script
├── requirements.txt      # Required packages
├── README.md             # Project README
