from prometheus_api_client import PrometheusConnect
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import datetime
import matplotlib.pyplot as plt

# Configuration for Prometheus
PROMETHEUS_URL = 'http://localhost:9090'
PROMETHEUS_ACCESS_TOKEN = ''

headers = {"Authorization": f"Bearer {PROMETHEUS_ACCESS_TOKEN}"} if PROMETHEUS_ACCESS_TOKEN else {}

prom = PrometheusConnect(url=PROMETHEUS_URL, disable_ssl=True, headers=headers)

end_time = datetime.datetime.now()
start_time = end_time - datetime.timedelta(days=7)
step = '30000'  # 5 minutes in seconds, as a string

query = 'rate(container_cpu_usage_seconds_total{pod=~"prometheus-monitor-kube-prometheus-st-prometheus-0"}[5m])'

data = prom.custom_query_range(query=query, start_time=start_time, end_time=end_time, step=step)

if data:
    # Process the data
    timestamps = [datetime.datetime.fromtimestamp(float(item[0])) for item in data[0]['values']]
    cpu_usage = [float(item[1]) for item in data[0]['values']]

    df = pd.DataFrame(data={'timestamp': timestamps, 'cpu_usage': cpu_usage})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['time_delta'] = (df['timestamp'] - df['timestamp'].min()) / np.timedelta64(1, 'D')

    # Split data into features and target
    X = df[['time_delta']]
    y = df['cpu_usage']

    # Linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predicting future CPU usage
    max_time_delta = df['time_delta'].max()
    future_days = 7
    future_time_deltas = [max_time_delta + i for i in range(1, future_days + 1)]
    future_predictions = model.predict(np.array(future_time_deltas).reshape(-1, 1))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df['timestamp'], df['cpu_usage'], color='blue', label='Historical CPU Usage', linestyle='-', marker='o')
    plt.plot(df['timestamp'], model.predict(X), color='red', label='Model Fit', linestyle='-')
    future_dates = [df['timestamp'].max() + datetime.timedelta(days=i) for i in range(1, future_days + 1)]
    plt.plot(future_dates, future_predictions, color='green', label='Predicted Future CPU Usage', linestyle='--', marker='x')
    plt.title("Nginx Pod CPU Usage Forecast")
    plt.xlabel("Timestamp")
    plt.ylabel("CPU Usage")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No data returned from Prometheus or unexpected data format.")
