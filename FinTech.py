import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from transformers import pipeline

data = pd.read_csv("filtered_df_part1.csv")

# Load pre-trained model for embeddings (distilbert for speed)
embedder = pipeline('feature-extraction', model='distilbert-base-uncased', tokenizer='distilbert-base-uncased')

# Create a binary price change column: 1 if closing price is higher than opening price, 0 otherwise
data['price_change'] = (data['Close'] > data['Open']).astype(int)

def get_embeddings(text):
    if isinstance(text, str):  # Ensure the input is a string
        embedding = embedder(text, truncation=True, max_length=50)
        return np.mean(embedding[0], axis=0)
    else:
        return np.zeros(768)

data['title_embedding'] = data['headline'].apply(lambda x: get_embeddings(str(x)))

X = np.stack(data['title_embedding'].values)
y = to_categorical(data['price_change'].values)

X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(1, 768), return_sequences=False))
model.add(Dropout(0.8))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
model.save('big_model.h5')
# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
y_test_classes = np.argmax(y_test, axis=1)  # Convert one-hot to class labels for comparison

# Evaluate predictions
accuracy = accuracy_score(y_test_classes, y_pred_classes)
print("Accuracy:", accuracy)
print("f1 score: ", f1_score(y_test_classes, y_pred_classes))
print("Classification Report:\n", classification_report(y_test_classes, y_pred_classes))
