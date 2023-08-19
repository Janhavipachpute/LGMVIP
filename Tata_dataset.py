import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load your dataset
# Replace 'your_data.csv' with your actual dataset file
data = pd.read_csv("C:/Users/Lenovo/Downloads/NSE-TATAGLOBAL.csv")

# Preprocess the data
data = data[['Date', 'Close']]  # Assuming you have a 'Date' and 'Close' column
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Normalize the data
scaler = MinMaxScaler()
data['Close'] = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Create training data
sequence_length = 10  # Adjust as needed
sequences = []
targets = []

for i in range(len(data) - sequence_length):
    sequences.append(data['Close'].values[i:i + sequence_length])
    targets.append(data['Close'].values[i + sequence_length])

X = np.array(sequences)
y = np.array(targets)

# Reshape the input data for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# Build the model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=50, batch_size=32)

# Forecasting
last_sequence = data['Close'].values[-sequence_length:]
last_sequence = last_sequence.reshape(1, sequence_length, 1)

predicted_value = model.predict(last_sequence)
predicted_value = scaler.inverse_transform(predicted_value)

print("Predicted value:", predicted_value)