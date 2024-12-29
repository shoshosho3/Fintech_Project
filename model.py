import pandas as pd
import numpy as np

def parse_combined_column(s):
    if s == '':
        return np.zeros(768)
    else:
        return np.array(eval(s))
    

df = pd.read_csv('final_data_with_features_2.csv', converters={
    'bert': parse_combined_column,
    'auto': parse_combined_column,
    'roberta': parse_combined_column
})

# Function to filter rows based on first non-NaN headline
def filter_after_first_non_nan(group):
    first_non_nan_index = group['headline'].first_valid_index()
    if first_non_nan_index is not None:
        return group.loc[first_non_nan_index:]
    else:
        return pd.DataFrame(columns=group.columns)

# Apply the function to each group
filtered_df = df.groupby('stock').apply(filter_after_first_non_nan).reset_index(drop=True)

print(filtered_df)
def change_sentiment_label(x):
    if x == 'POSITIVE':
        return 1
    elif x == 'NEGATIVE':
        return -1
    else:
        return 0

def change_sentiment_score(x):
    if -1<=x<=1 :
        return x
    else:
        return 0
    
filtered_df['sentiment_label'] = filtered_df['sentiment_label'].apply(change_sentiment_label)
filtered_df['sentiment_score'] = filtered_df['sentiment_score'].apply(change_sentiment_score)
# set sentiment score to be the sentiment score times the sentiment label
filtered_df['sentiment_score'] = filtered_df['sentiment_score'] * filtered_df['sentiment_label']

# Shift the 'sentiment_score' and 'sentiment_label' columns down by one row
filtered_df['sentiment_score'] = filtered_df['sentiment_score'].shift(1).fillna(0)
filtered_df['sentiment_label'] = filtered_df['sentiment_label'].shift(1).fillna(0)

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Assuming 'df' is your DataFrame with 'stock', 'Open', and 'Close' columns

# Function to apply MinMaxScaler to each group independently
def scale_prices(group):
    scaler_close = MinMaxScaler(feature_range=(0, 1))
    # scaler_open = MinMaxScaler(feature_range=(0, 1))
    
    # Fit and transform 'Close' prices
    group['normalized_closing_price'] = scaler_close.fit_transform(group['Close'].values.reshape(-1, 1))
    
    # only transform open prices
    group['normalized_opening_price'] = scaler_close.transform(group['Open'].values.reshape(-1, 1))
    
    return group

# Apply the scaler independently to each stock
filtered_df_same_scaler = filtered_df.groupby('stock').apply(scale_prices)
avgo_based_df = filtered_df_same_scaler[filtered_df_same_scaler['Date'] <= '2020-06-03']
avgo_based_df = avgo_based_df[avgo_based_df['Date'] >= '2016-10-31']
import pandas as pd
import numpy as np

# Filter AVGO data
avgo_df = avgo_based_df[avgo_based_df['stock'] == 'AVGO']

# Get the list of other stocks
other_stocks = avgo_based_df['stock'].unique()
other_stocks = other_stocks[other_stocks != 'AVGO']

# Initialize columns for vectors of other stocks
columns = {
    'normalized_closing_price': [],
    'normalized_opening_price': [],
    'sentiment_label': [],
    'sentiment_score': []
}

# Function to aggregate data into vectors for each row corresponding to AVGO stock
def aggregate_vectors(date):
    vectors = {key: [] for key in columns}
    for stock in other_stocks:
        stock_data = avgo_based_df[(avgo_based_df['stock'] == stock) & (avgo_based_df['Date'] == date)]
        if not stock_data.empty:
            vectors['normalized_closing_price'].append(stock_data['normalized_closing_price'].values[0])
            vectors['normalized_opening_price'].append(stock_data['normalized_opening_price'].values[0])
            vectors['sentiment_label'].append(stock_data['sentiment_label'].values[0])
            vectors['sentiment_score'].append(stock_data['sentiment_score'].values[0])
        else:
            # Append NaN or a specific value if data is missing for this stock/date
            # TODO: Handle missing data more appropriately
            vectors['normalized_closing_price'].append(0)
            vectors['normalized_opening_price'].append(0)
            vectors['sentiment_label'].append(0)
            vectors['sentiment_score'].append(0)
    return vectors

# Loop through AVGO dates and aggregate vectors for each date
for date in avgo_df['Date']:
    vectors = aggregate_vectors(date)
    for key in columns:
        columns[key].append(vectors[key])

# Add the vectors as columns to the AVGO dataframe
avgo_df['other_stocks_normalized_closing_price'] = columns['normalized_closing_price']
avgo_df['other_stocks_normalized_opening_price'] = columns['normalized_opening_price']
avgo_df['other_stocks_sentiment_label'] = columns['sentiment_label']
avgo_df['other_stocks_sentiment_score'] = columns['sentiment_score']

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Define Dataset class with date handling
class StockDataset(Dataset):
    def __init__(self, df):
        self.avgo_features = df[['normalized_opening_price', 'sentiment_label', 'sentiment_score']].values
        self.other_features = np.concatenate([
            # np.stack(df['other_stocks_normalized_closing_price'].values),
            np.stack(df['other_stocks_normalized_opening_price'].values),
            np.stack(df['other_stocks_sentiment_label'].values),
            np.stack(df['other_stocks_sentiment_score'].values)
        ], axis=1)
        self.targets = df['normalized_closing_price'].values

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.avgo_features[idx], dtype=torch.float32), 
            torch.tensor(self.other_features[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )

# Define the neural network model
class StockPredictor(nn.Module):
    def __init__(self, avgo_input_size, other_input_size, hidden_size):
        super(StockPredictor, self).__init__()
        
        # AVGO-specific branch
        self.avgo_branch = nn.Sequential(
            nn.Linear(avgo_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Other stocks branch
        self.other_branch = nn.Sequential(
            nn.Linear(other_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Fusion layer combining AVGO and other stocks
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # Output layer for closing price prediction
        )

    def forward(self, avgo_features, other_features):
        # Forward pass for AVGO
        avgo_out = self.avgo_branch(avgo_features)
        
        # Forward pass for other stocks
        other_out = self.other_branch(other_features)
        
        # Concatenate outputs from both branches
        combined = torch.cat((avgo_out, other_out), dim=1)
        
        # Final prediction
        return self.fusion(combined).squeeze()

# Split data into training and testing sets
AVGO_train = avgo_df[avgo_df['Date'] < '2020-02-10']
AVGO_test = avgo_df[avgo_df['Date'] >= '2020-02-10']

# Prepare the datasets
train_dataset = StockDataset(AVGO_train)
test_dataset = StockDataset(AVGO_test)

# Define DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model
avgo_input_size = 3  # AVGO features: opening price, sentiment label, sentiment score
other_input_size = train_dataset.other_features.shape[1]  # Combined other stocks features
hidden_size = 64
model = StockPredictor(avgo_input_size, other_input_size, hidden_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for avgo_features, other_features, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(avgo_features, other_features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}')

# Evaluation on test data
model.eval()
correct_direction = 0
total = 0
all_actual_directions = []
all_predicted_directions = []

# Prepare previous day's closing prices for direction comparison
previous_closing_prices = AVGO_test['normalized_closing_price'].shift(1).fillna(method='bfill').values

with torch.no_grad():
    for avgo_features, other_features, targets in test_loader:
        outputs = model(avgo_features, other_features)
        predicted_prices = outputs.numpy()

        # Compare directions: 1 if price increased, 0 if decreased
        predicted_direction = (predicted_prices > previous_closing_prices[:len(predicted_prices)]).astype(int)
        actual_direction = (targets.numpy() > previous_closing_prices[:len(targets)]).astype(int)

        all_predicted_directions.extend(predicted_direction)
        all_actual_directions.extend(actual_direction)

        correct_direction += np.sum(predicted_direction == actual_direction)
        total += len(actual_direction)

# Calculate accuracy and F1 score
accuracy = accuracy_score(all_actual_directions, all_predicted_directions)
f1 = f1_score(all_actual_directions, all_predicted_directions)

print(f"Accuracy of predicting the direction of AVGO stock: {accuracy:.4f}")
print(f"F1 Score of predicting the direction of AVGO stock: {f1:.4f}")

intc_based_df = filtered_df_same_scaler[filtered_df_same_scaler['Date'] <= '2020-06-03']
intc_based_df = intc_based_df[intc_based_df['Date'] >= '2018-10-25']

import pandas as pd
import numpy as np

# Filter INTC data
intc_df = intc_based_df[intc_based_df['stock'] == 'INTC']

# Get the list of other stocks
other_stocks = intc_based_df['stock'].unique()
other_stocks = other_stocks[other_stocks != 'INTC']

# Initialize columns for vectors of other stocks
columns = {
    'normalized_closing_price': [],
    'normalized_opening_price': [],
    'sentiment_label': [],
    'sentiment_score': []
}

# Function to aggregate data into vectors for each row corresponding to INTC stock
def aggregate_vectors(date):
    vectors = {key: [] for key in columns}
    for stock in other_stocks:
        stock_data = intc_based_df[(intc_based_df['stock'] == stock) & (intc_based_df['Date'] == date)]
        if not stock_data.empty:
            vectors['normalized_closing_price'].append(stock_data['normalized_closing_price'].values[0])
            vectors['normalized_opening_price'].append(stock_data['normalized_opening_price'].values[0])
            vectors['sentiment_label'].append(stock_data['sentiment_label'].values[0])
            vectors['sentiment_score'].append(stock_data['sentiment_score'].values[0])
        else:
            # Append NaN or a specific value if data is missing for this stock/date
            # TODO: Handle missing data more appropriately
            vectors['normalized_closing_price'].append(0)
            vectors['normalized_opening_price'].append(0)
            vectors['sentiment_label'].append(0)
            vectors['sentiment_score'].append(0)
    return vectors

# Loop through INTC dates and aggregate vectors for each date
for date in intc_df['Date']:
    vectors = aggregate_vectors(date)
    for key in columns:
        columns[key].append(vectors[key])

# Add the vectors as columns to the INTC dataframe
intc_df['other_stocks_normalized_closing_price'] = columns['normalized_closing_price']
intc_df['other_stocks_normalized_opening_price'] = columns['normalized_opening_price']
intc_df['other_stocks_sentiment_label'] = columns['sentiment_label']
intc_df['other_stocks_sentiment_score'] = columns['sentiment_score']

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Define Dataset class with date handling
class StockDataset(Dataset):
    def __init__(self, df):
        self.intc_features = df[['normalized_opening_price', 'sentiment_label', 'sentiment_score']].values
        self.other_features = np.concatenate([
            # np.stack(df['other_stocks_normalized_closing_price'].values),
            np.stack(df['other_stocks_normalized_opening_price'].values),
            np.stack(df['other_stocks_sentiment_label'].values),
            np.stack(df['other_stocks_sentiment_score'].values)
        ], axis=1)
        self.targets = df['normalized_closing_price'].values

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.intc_features[idx], dtype=torch.float32), 
            torch.tensor(self.other_features[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )

# Define the neural network model
class StockPredictor(nn.Module):
    def __init__(self, intc_input_size, other_input_size, hidden_size):
        super(StockPredictor, self).__init__()
        
        # INTC-specific branch
        self.intc_branch = nn.Sequential(
            nn.Linear(intc_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Other stocks branch
        self.other_branch = nn.Sequential(
            nn.Linear(other_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Fusion layer combining INTC and other stocks
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # Output layer for closing price prediction
        )

    def forward(self, intc_features, other_features):
        # Forward pass for INTC
        intc_out = self.intc_branch(intc_features)
        
        # Forward pass for other stocks
        other_out = self.other_branch(other_features)
        
        # Concatenate outputs from both branches
        combined = torch.cat((intc_out, other_out), dim=1)
        
        # Final prediction
        return self.fusion(combined).squeeze()

# Split data into training and testing sets
INTC_train = intc_df[intc_df['Date'] < '2020-02-10']
INTC_test = intc_df[intc_df['Date'] >= '2020-02-10']

# Prepare the datasets
train_dataset = StockDataset(INTC_train)
test_dataset = StockDataset(INTC_test)

# Define DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model
intc_input_size = 3  # INTC features: opening price, sentiment label, sentiment score
other_input_size = train_dataset.other_features.shape[1]  # Combined other stocks features
hidden_size = 64
model = StockPredictor(intc_input_size, other_input_size, hidden_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for intc_features, other_features, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(intc_features, other_features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}')

# Evaluation on test data
model.eval()
correct_direction = 0
total = 0
all_actual_directions = []
all_predicted_directions = []

# Prepare previous day's closing prices for direction comparison
previous_closing_prices = INTC_test['normalized_closing_price'].shift(1).fillna(method='bfill').values

with torch.no_grad():
    for intc_features, other_features, targets in test_loader:
        outputs = model(intc_features, other_features)
        predicted_prices = outputs.numpy()

        # Compare directions: 1 if price increased, 0 if decreased
        predicted_direction = (predicted_prices > previous_closing_prices[:len(predicted_prices)]).astype(int)
        actual_direction = (targets.numpy() > previous_closing_prices[:len(targets)]).astype(int)

        all_predicted_directions.extend(predicted_direction)
        all_actual_directions.extend(actual_direction)

        correct_direction += np.sum(predicted_direction == actual_direction)
        total += len(actual_direction)

# Calculate accuracy and F1 score
accuracy = accuracy_score(all_actual_directions, all_predicted_directions)
f1 = f1_score(all_actual_directions, all_predicted_directions)

print(f"Accuracy of predicting the direction of INTC stock: {accuracy:.4f}")
print(f"F1 Score of predicting the direction of INTC stock: {f1:.4f}")