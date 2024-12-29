import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from transformers import pipeline
from datetime import datetime

# Load the pre-trained LSTM model
model = load_model('big_model.h5')

# Define parameters for simulation
initial_capital = 20000  # Starting amount in dollars
capital = initial_capital

# Initialize the embeddings model
embedder = pipeline('feature-extraction', model='distilbert-base-uncased', tokenizer='distilbert-base-uncased')

# Function to fetch stock data
def fetch_stock_data():
    stock_data = pd.read_csv('filtered_df_small_amount.csv')
    return stock_data

# Function to get embeddings for titles
def get_embeddings(text):
    if isinstance(text, str):  # Ensure the input is a string
        embedding = embedder(text, truncation=True, max_length=50)
        return np.mean(embedding[0], axis=0)
    else:
        return np.zeros(768)


data = fetch_stock_data()

# Preprocess the data
data['title_embedding'] = data['headline'].apply(lambda x: get_embeddings(str(x)))
data['price_change'] = (data['Close'] > data['Open']).astype(int)

# Simulation function
def simulate_trading(data, model, capital):
    daily_trades = []  # Track daily trade outcomes
    data['investment'] = 0  # Track investments

    for idx, row in data.iterrows():
        # Get the headline embedding and reshape for model input
        if row['title_embedding'] is not None:
            embedding = np.reshape(np.array(row['title_embedding']), (1, 1, 768))
            prediction = model.predict(embedding)
            prob_increase = prediction[0][1]  # Probability of price increase
            # The investment the will maximize the expectancy
            investment_per_stock = capital * (2 * prob_increase - 1)
            capital -= investment_per_stock  # Deduct investment
            capital += investment_per_stock * (row['Close'] / row['Open'])  # Adjust for price movement

            data.at[idx, 'investment'] = investment_per_stock

            daily_trades.append(capital)

    return daily_trades, capital

# Run the simulation
trading_results, final_capital = simulate_trading(data, model, capital)

# Output results
print(f"Initial capital: ${initial_capital}")
print(f"Final capital after trading: ${final_capital}")
data['daily_capital'] = trading_results
print("Capital over time:")
print(data[['Date', 'daily_capital']])

# Save the results to a CSV file
data[['Date', 'daily_capital']].to_csv('trading_simulation_results.csv', index=False)
