import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from prometheus_api_client import PrometheusConnect
from math import sqrt
import datetime

# Configuration for Prometheus
PROMETHEUS_URL = 'http://localhost:9090'
PROMETHEUS_ACCESS_TOKEN = ''  # Your access token if needed

# Initialize Prometheus connection
prom = PrometheusConnect(url=PROMETHEUS_URL, disable_ssl=True)

# Define the range query parameters for the last 7 days with 5-minute intervals
end_time = datetime.datetime.now()
start_time = end_time - datetime.timedelta(days=7)
step = '3000'

# Prometheus query to fetch CPU usage
query = 'rate(container_cpu_usage_seconds_total{pod=~"opensearch-master-0"}[5m])'

# Fetch the data over the last 7 days with 5-minute intervals
data = prom.custom_query_range(query=query, start_time=start_time, end_time=end_time, step=step)

if data:
    # Process the data
    timestamps = [datetime.datetime.fromtimestamp(float(item[0])) for item in data[0]['values']]
    cpu_usage = [float(item[1]) for item in data[0]['values']]

    df = pd.DataFrame(data={'timestamp': timestamps, 'cpu_usage': cpu_usage})
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_cpu_usage = scaler.fit_transform(df['cpu_usage'].values.reshape(-1, 1))
    look_back = 3
    
    # Convert an array of values into a dataset matrix
    def create_dataset(dataset, look_back=3):  # Increased look-back period
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    train_size = int(len(scaled_cpu_usage) * 0.67)
    test_size = len(scaled_cpu_usage) - train_size
    train, test = scaled_cpu_usage[0:train_size, :], scaled_cpu_usage[train_size:len(scaled_cpu_usage), :]

    trainX, trainY = create_dataset(train)
    testX, testY = create_dataset(test)

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, input_shape=(1, look_back), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(25, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=Adam(0.001))

    # Fit the model with early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    history = model.fit(trainX, trainY, epochs=1000, batch_size=1, verbose=2, validation_data=(testX, testY), callbacks=[early_stop])

    # Predictions for training and testing dataset
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # Invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # Calculate and print RMSE and MAE
    trainScore = sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    testScore = sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    print('Test Score: %.2f RMSE' % (testScore))

    # Visualization
    plt.figure(figsize=(15, 6))
    plt.plot(df['timestamp'], df['cpu_usage'], label='Original Data')
    plt.plot(df['timestamp'], np.pad(trainPredict.ravel(), (look_back, len(df) - len(trainPredict) - look_back), 'constant', constant_values=np.nan), label='Train Prediction')
    plt.plot(df['timestamp'], np.pad(testPredict.ravel(), (len(trainPredict) + 2*look_back + 1, len(df) - len(testPredict) - len(trainPredict) - 2*look_back - 1), 'constant', constant_values=np.nan), label='Test Prediction')
    plt.title('CPU Usage Prediction')
    plt.xlabel('Timestamp')
    plt.ylabel('CPU Usage')
    plt.legend()
    plt.show()
else:
    print("No data returned from Prometheus or unexpected data format.")
