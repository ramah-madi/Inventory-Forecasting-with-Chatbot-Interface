# Importing the necessary libraries
from io import BytesIO, StringIO
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile
from pmdarima import auto_arima
from scipy.signal import find_peaks
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import acf

app = FastAPI()

# Statistical Models
@app.post("/api/statsForecast/")
async def forecast(file: UploadFile = File(...), start: str = Form(...), end: str = Form(...), date_column: str = Form(...), target_column: str = Form(...)):
    def preprocess(data: pd.DataFrame, date_column: str, value_column: str) -> pd.Series:
        # Convert the date column to datetime and set it as the index
        data[date_column] = pd.to_datetime(data[date_column])
        data.set_index(date_column, inplace=True)
        
        # Preprocess the data
        series = data[value_column]
        return series

    def model_auto_arima(data: pd.Series, periods: int):
        # Use auto_arima to find the best ARIMA parameters
        model = auto_arima(data, seasonal=False, stepwise=True, trace=True)
        
        # Forecast for the specified number of periods
        forecast = model.predict(n_periods=periods)
        return forecast

    def model_holt_winters(data: pd.Series, periods: int):
        # Calculate autocorrelation function
        lags = 50  # Set the maximum lag to search for seasonal period
        acf_vals = acf(data, fft=True, nlags=lags)

        # Find peaks in the ACF
        peaks, _ = find_peaks(acf_vals)

        # Find the first peak (excluding the first lag which is always 1)
        first_peak = peaks[0] if len(peaks) > 0 else None

        # Estimate the seasonal period
        seasonal_period = first_peak if first_peak is not None else np.argmax(acf_vals[1:]) + 1

        # Use the estimated seasonal period in the model
        model = ExponentialSmoothing(data, seasonal='add', seasonal_periods=seasonal_period)
        model_fit = model.fit()
        
        # Forecast for the specified number of periods
        forecast = model_fit.forecast(steps=periods)
        return forecast
    
    # Read the uploaded file content
    contents = await file.read()
    data = pd.read_csv(BytesIO(contents), delimiter=',')

    # Preprocess the data
    series = preprocess(data, date_column, target_column)

    # Split data into train and test sets
    train_size = int(len(series) * 0.9)
    train, test = series[:train_size], series[train_size:]

    # Perform forecasting on the test data to calculate accuracy
    arima_forecast_test = model_auto_arima(train, len(test))
    arima_mae_test = mean_absolute_error(test, arima_forecast_test)
    arima_mse_test = mean_squared_error(test, arima_forecast_test)

    hw_forecast_test = model_holt_winters(train, len(test))
    hw_mae_test = mean_absolute_error(test, hw_forecast_test)
    hw_mse_test = mean_squared_error(test, hw_forecast_test)

    # Determine the number of steps for future forecasting based on period type
    forecast_duration = (pd.to_datetime(end) - pd.to_datetime(start)).days
    future_steps = forecast_duration

    # Perform future forecasting
    arima_forecast_future = model_auto_arima(series, future_steps)
    hw_forecast_future = model_holt_winters(series, future_steps)

    # Prepare response
    response = {
        "Test_Accuracy": {
            "Auto_ARIMA": {
                "MAE": arima_mae_test,
                "MSE": arima_mse_test
            },
            "Holt_Winters": {
                "MAE": hw_mae_test,
                "MSE": hw_mse_test
            }
        },
        "Future_Forecast": {
            "Auto_ARIMA": arima_forecast_future.tolist(),
            "Holt_Winters": hw_forecast_future.tolist()
        }
    }

    return response

# Machine learning models 
@app.post("/api/mlForecast/")
async def ml_forecast(file: UploadFile = File(...), start: str = Form(...), end: str = Form(...), date_column: str = Form(...), target_column: str = Form(...)):

    def preprocess(data: pd.DataFrame, date_column: str, value_column: str) -> pd.Series:
        # Convert the date column to datetime
        data[date_column] = pd.to_datetime(data[date_column])
        
        # Set the index to the date column
        data.set_index(date_column, inplace=True)
        
        # Extract date features
        data['year'] = data.index.year
        data['month'] = data.index.month
        data['day'] = data.index.day
        data['dayofweek'] = data.index.dayofweek
        data['weekofyear'] = data.index.isocalendar().week  # Get week number using isocalendar().wee
        data['quarter'] = data.index.quarter
        
        # Preprocess the data
        series = data[value_column]

        return series

    # Read the uploaded file content
    contents = await file.read()

    # Check the type of content and use the appropriate IO class
    if isinstance(contents, str):
        data = pd.read_csv(StringIO(contents), delimiter=',')
    else:
        data = pd.read_csv(BytesIO(contents), delimiter=',')

    # Preprocess the data
    series = preprocess(data, date_column, target_column)

    # Create a time-based split for train and test sets
    train_size = int(len(series) * 0.9)
    train, test = series[:train_size], series[train_size:]

    def create_lagged_features(series, lag=1):
        df = pd.DataFrame(series)
        columns = [df.shift(i) for i in range(1, lag + 1)]
        columns.append(df)
        df = pd.concat(columns, axis=1)
        df.dropna(inplace=True)
        return df

    lag = 1  # Number of lagged features
    train_lagged = create_lagged_features(train, lag)
    test_lagged = create_lagged_features(test, lag)

    X_train = train_lagged.iloc[:, :-1].values
    y_train = train_lagged.iloc[:, -1].values
    X_test = test_lagged.iloc[:, :-1].values
    y_test = test_lagged.iloc[:, -1].values

    # Train GradientBoosting model
    gb_model = GradientBoostingRegressor()
    gb_model.fit(X_train, y_train)

    # Perform forecasting on the test data to calculate accuracy
    gb_forecast_test = gb_model.predict(X_test)
    gb_mae_test = mean_absolute_error(y_test, gb_forecast_test)
    gb_mse_test = mean_squared_error(y_test, gb_forecast_test)

    # Perform future forecasting
    forecast_duration = (pd.to_datetime(end) - pd.to_datetime(start)).days
    last_values = train[-lag:].values
    gb_forecast_future = []

    for _ in range(forecast_duration):
        next_value = gb_model.predict(last_values.reshape(1, -1))[0]
        gb_forecast_future.append(next_value)
        last_values = np.roll(last_values, -1)
        last_values[-1] = next_value

    # Train MLP model
    mlp_model = MLPRegressor()
    mlp_model.fit(np.arange(len(train)).reshape(-1, 1), train.values)

    # Perform forecasting on the test data to calculate accuracy
    mlp_forecast_test = mlp_model.predict(np.arange(len(train), len(train) + len(test)).reshape(-1, 1))
    mlp_mae_test = mean_absolute_error(test.values, mlp_forecast_test)
    mlp_mse_test = mean_squared_error(test.values, mlp_forecast_test)

    # Perform future forecasting for MLP
    mlp_forecast_future = mlp_model.predict(np.arange(len(series), len(series) + forecast_duration).reshape(-1, 1))

    # Prepare response
    response = {
        "Test_Accuracy": {
            "GradientBoosting": {
                "MAE": gb_mae_test,
                "MSE": gb_mse_test
            },
            "MLPRegressor": {
                "MAE": mlp_mae_test,
                "MSE": mlp_mse_test
            }
        },
        "Future_Forecast": {
            "GradientBoosting": gb_forecast_future,
            "MLPRegressor": mlp_forecast_future.tolist()
        }
    }

    return response