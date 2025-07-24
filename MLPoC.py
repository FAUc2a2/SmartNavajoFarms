import pandas as pd
from datetime import datetime, timedelta
import requests
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Define the time window in days
timeWindow = 7

# Set the forecast window in days (used for plot label)
forecastWindow = 1 # This variable needs to be defined if used in the plot label

# --- InfluxDB Data Retrieval and Preprocessing ---

# Adjust the time range for the InfluxDB query
end_time_query = datetime.now()
start_time_query = end_time_query - timedelta(days=timeWindow)

# Query data with the adjusted time range
table = client.query(
    query=f"SELECT * FROM '4103' WHERE time >= '{start_time_query.isoformat()}Z'",
    language="sql"
)

# Convert to Pandas DataFrame and preprocess
df = table.to_pandas()
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values(by=['sensorId', 'time'])

# Calculate the overall moving average
window_size = 30
df = df.sort_values(by='time')
df['moving_avg'] = df['value'].rolling(window=window_size).mean()

# --- Fetch Rainfall Data with Adjustable Time Range ---

# Weather API details
weather_api_key = 'YNDQ2KZLNYVX48KHABGUZP67K'
location = 'Boca Raton,Florida'

# Define the date range for rainfall data
start_date_str = (datetime.now() - timedelta(days=timeWindow)).strftime('%Y-%m-%d')
end_date_str = datetime.now().strftime('%Y-%m-%d')

# Construct and fetch historical rainfall data
weather_api_url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{start_date_str}/{end_date_str}?unitGroup=us&include=days&key={weather_api_key}&contentType=json"

rainfall_df = pd.DataFrame()
try:
    weather_response = requests.get(weather_api_url)
    weather_response.raise_for_status()
    weather_data = weather_response.json()

    rainfall_data = []
    rainfall_times = []
    if 'days' in weather_data:
        for day_data in weather_data['days']:
            day_time = datetime.fromtimestamp(day_data['datetimeEpoch'])
            rainfall_inches = day_data.get('precip', 0.0)
            rainfall_data.append(rainfall_inches)
            rainfall_times.append(day_time)

    rainfall_df = pd.DataFrame({'time': rainfall_times, 'rainfall_inches': rainfall_data})
    rainfall_df['time'] = pd.to_datetime(rainfall_df['time'])

except requests.exceptions.RequestException as e:
    print(f"Error fetching historical weather data: {e}")
    print("Historical rainfall data will not be included in training.")

# --- Merge Sensor and Rainfall Data ---

# Merge sensor data with rainfall data
if not rainfall_df.empty:
    df['date'] = df['time'].dt.date
    rainfall_df['date'] = rainfall_df['time'].dt.date
    df_merged = pd.merge(df, rainfall_df[['date', 'rainfall_inches']], on='date', how='left')
    df_merged['rainfall_inches'] = df_merged['rainfall_inches'].fillna(0)
    df_merged = df_merged.drop('date', axis=1)
else:
    df_merged = df.copy()
    df_merged['rainfall_inches'] = 0

# --- Prepare Data for TensorFlow Model with Multiple Inputs ---

# Scale the data and create sequences for the LSTM model
df_merged.dropna(subset=['moving_avg'], inplace=True)
data = df_merged[['moving_avg', 'rainfall_inches']].values

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

def create_dataset_multi_feature(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 60
X, Y = create_dataset_multi_feature(scaled_data, look_back)

# Build and train the LSTM model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(look_back, X.shape[2])))
model.add(tf.keras.layers.LSTM(50))
model.add(tf.keras.layers.Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, Y, epochs=10, batch_size=1, verbose=2)

# --- Make Predictions for the Future including Forecasted Rainfall ---

# Fetch weather forecast data
forecast_api_url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/next24hours?unitGroup=us&include=hours&key={weather_api_key}&contentType=json"

forecasted_rainfall_df = pd.DataFrame({'time': [], 'rainfall_inches': []})
try:
    forecast_response = requests.get(forecast_api_url)
    forecast_response.raise_for_status()
    forecast_data = forecast_response.json()

    forecasted_rainfall_data = []
    forecasted_rainfall_times = []
    if 'days' in forecast_data and len(forecast_data['days']) > 0 and 'hours' in forecast_data['days'][0]:
        for hour_data in forecast_data['days'][0]['hours']:
            hour_time = datetime.fromtimestamp(hour_data['datetimeEpoch'])
            rainfall_inches = hour_data.get('precip', 0.0)
            forecasted_rainfall_data.append(rainfall_inches)
            forecasted_rainfall_times.append(hour_time)

    forecasted_rainfall_df = pd.DataFrame({'time': forecasted_rainfall_times, 'rainfall_inches': forecasted_rainfall_data})
    forecasted_rainfall_df['time'] = pd.to_datetime(forecasted_rainfall_df['time'])

except requests.exceptions.RequestException as e:
    print(f"Error fetching weather forecast data: {e}")
    print("Forecasted rainfall data will not be included in future predictions.")

# Generate future predictions
last_data_points_scaled = scaled_data[-look_back:].reshape(1, look_back, scaled_data.shape[1])
current_batch_scaled = last_data_points_scaled

forecast_steps = 12
forecast = []
forecast_time_steps = []
last_actual_time = df_merged['time'].max()
num_forecast_steps_possible = min(forecast_steps, len(forecasted_rainfall_df))

for i in range(num_forecast_steps_possible):
    next_forecast_time = last_actual_time + timedelta(hours=i+1)
    forecast_time_steps.append(next_forecast_time)

    if not forecasted_rainfall_df.empty:
        closest_rainfall_idx = (forecasted_rainfall_df['time'] - next_forecast_time).abs().idxmin()
        scaled_rainfall_for_this_step = scaler.transform([[0, forecasted_rainfall_df.loc[closest_rainfall_idx]['rainfall_inches']]])[0, 1]
    else:
        scaled_rainfall_for_this_step = 0.0

    predicted_scaled_moving_avg = model.predict(current_batch_scaled)[0][0]
    forecast.append(predicted_scaled_moving_avg)

    if (i + 1) < len(forecasted_rainfall_df):
        next_forecast_time_step = last_actual_time + timedelta(hours=i+2)
        closest_rainfall_idx_next = (forecasted_rainfall_df['time'] - next_forecast_time_step).abs().idxmin()
        scaled_rainfall_for_next_step = scaler.transform([[0, forecasted_rainfall_df.loc[closest_rainfall_idx_next]['rainfall_inches']]])[0, 1]
        next_input_point = np.array([[predicted_scaled_moving_avg, scaled_rainfall_for_next_step]])
    else:
        print(f"Warning: Not enough rainfall forecast data to include rainfall beyond step {i}. Assuming 0 rainfall for future predictions.")
        scaled_rainfall_for_next_step = scaler.transform([[0, 0]])[0, 1]
        next_input_point = np.array([[predicted_scaled_moving_avg, scaled_rainfall_for_next_step]])

    current_batch_scaled = np.append(current_batch_scaled[:, 1:, :], [next_input_point], axis=1)

forecast = scaler.inverse_transform(np.concatenate((np.array(forecast).reshape(-1, 1), np.zeros((len(forecast), 1))), axis=1))[:, 0]

# --- Plotting ---

# Create a figure and axes for the combined plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Define the start time for plotting
end_time_plot = datetime.now()
start_time_plot = end_time_plot - timedelta(days=timeWindow)

# Filter data for plotting
df_plot = df_merged[df_merged['time'] >= start_time_plot].copy()
sensor_ids_plot = df[df['time'] >= start_time_plot]['sensorId'].unique()

# Plot sensor data, moving average, and forecasted moving average
for sensor_id in sensor_ids_plot:
    sensor_df_plot = df[df['sensorId'] == sensor_id].copy()
    sensor_df_plot = sensor_df_plot[sensor_df_plot['time'] >= start_time_plot]
    ax1.plot(sensor_df_plot['time'], sensor_df_plot['value'], label=f'Sensor ID: {sensor_id}', alpha=0.5)

df_ma_plot = df_merged[df_merged['time'] >= start_time_plot].copy()
ax1.plot(df_ma_plot['time'], df_ma_plot['moving_avg'], label=f'Overall Moving Average (Window: {window_size})', color='black', linestyle='--')

if len(forecast_time_steps) > 0:
    ax1.plot(forecast_time_steps, forecast, label=f'{forecastWindow}-day Forecast', color='red', linestyle='-')

ax1.set_xlabel('Time')
ax1.set_ylabel('Sensor Value / Moving Average', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_title(f'Sensor Data (Last {timeWindow} Days), Moving Average, Forecast, and Rainfall')
ax1.legend(loc='upper left')

# Create and plot rainfall data on a second y-axis
ax2 = ax1.twinx()

rainfall_plot_df = rainfall_df[(rainfall_df['time'] >= start_time_plot) & (rainfall_df['time'] <= forecast_time_steps[-1] if len(forecast_time_steps) > 0 else end_time_plot)].copy()

if not rainfall_plot_df.empty:
    ax2.plot(rainfall_plot_df['time'], rainfall_plot_df['rainfall_inches'], label='Rainfall (inches)', color='green', linestyle='-', alpha=0.7)

ax2.set_ylabel('Rainfall (inches)', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.legend(loc='upper right')

# Set x-axis limits and display plot
if len(forecast_time_steps) > 0:
    ax1.set_xlim(start_time_plot, forecast_time_steps[-1])
    ax2.set_xlim(start_time_plot, forecast_time_steps[-1])
else:
    ax1.set_xlim(start_time_plot, end_time_plot)
    ax2.set_xlim(start_time_plot, end_time_plot)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
