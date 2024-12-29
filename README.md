# Stock Price Movement Prediction Using Financial News Headlines

This project implements a prediction of rise/fall of stocks based on news headlines using machine and deep learning techniques.

## Files

- `preprocess.py`: This file contains the preprocessing of the data - the headlines are fitted to dates.
- `extracting_features_updated.ipynb`: This notebook reads the preprocessed data. It then generates embeddings and sentiment analysis columns for the data.
- `model.py`: This file contains the Neural Network training and prediction process.

## Data
We use financial news headlines data from the following Kaggle source: https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests/data?select=raw_partner_headlines.csv

### Prerequisites

	•	Required Python libraries: pandas, yfinance, numpy, transformers, torch

### Running the Code

	1.	Download the data from kaggle.
	2.	Run preprocess.py
 	3.	Run extracting_features_updated.ipynb
  	4.	Run model.py
