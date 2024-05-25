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

@app.get("/")
def hello():
    return {"message": "Hello to forecast API"}

def handle_missing_values(data):
    # Check if there are any missing values in the data
    if data.isnull().values.any():
        # Fill missing values using forward fill
        data = data.fillna(method='ffill')
    return data

def convert_target_to_numeric(data, target_column):
    data[target_column] = pd.to_numeric(data[target_column], errors='coerce')
    data = data.dropna(subset=[target_column])
    return data

# Statistical Models
@app.post("/api/statsForecast/")
async def forecast(file: UploadFile = File(...), start: str = Form(...), end: str = Form(...), date_column: str = Form(...), target_column: str = Form(...)):
    def preprocess(data: pd.DataFrame, date_column: str, value_column: str) -> pd.Series:
        data = handle_missing_values(data)
        data = convert_target_to_numeric(data, value_column)
        data[date_column] = pd.to_datetime(data[date_column])
        data.set_index(date_column, inplace=True)
        series = data[value_column]
        return series

    def model_auto_arima(data: pd.Series, periods: int):
        model = auto_arima(data, seasonal=False, stepwise=True, trace=True)
        forecast = model.predict(n_periods=periods)
        return forecast

    def model_holt_winters(data: pd.Series, periods: int):
        lags = 50
        acf_vals = acf(data, fft=True, nlags=lags)
        peaks, _ = find_peaks(acf_vals)
        first_peak = peaks[0] if len(peaks) > 0 else None
        seasonal_period = first_peak if first_peak is not None else np.argmax(acf_vals[1:]) + 1
        model = ExponentialSmoothing(data, seasonal='add', seasonal_periods=seasonal_period)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=periods)
        return forecast
    
    contents = await file.read()
    data = pd.read_csv(BytesIO(contents), delimiter=',')
    series = preprocess(data, date_column, target_column)
    train_size = int(len(series) * 0.9)
    train, test = series[:train_size], series[train_size:]

    arima_forecast_test = model_auto_arima(train, len(test))
    arima_mae_test = mean_absolute_error(test, arima_forecast_test)
    arima_mse_test = mean_squared_error(test, arima_forecast_test)

    hw_forecast_test = model_holt_winters(train, len(test))
    hw_mae_test = mean_absolute_error(test, hw_forecast_test)
    hw_mse_test = mean_squared_error(test, hw_forecast_test)

    forecast_duration = (pd.to_datetime(end) - pd.to_datetime(start)).days
    future_steps = forecast_duration

    arima_forecast_future = model_auto_arima(series, future_steps)
    hw_forecast_future = model_holt_winters(series, future_steps)

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

    def preprocess(data: pd.DataFrame, date_column: str, value_column: str) -> pd.DataFrame:
        data = handle_missing_values(data)
        data = convert_target_to_numeric(data, value_column)
        data[date_column] = pd.to_datetime(data[date_column])
        data.set_index(date_column, inplace=True)
        data['year'] = data.index.year
        data['month'] = data.index.month
        data['day'] = data.index.day
        data['dayofweek'] = data.index.dayofweek
        data['weekofyear'] = data.index.isocalendar().week
        data['quarter'] = data.index.quarter
        return data

    contents = await file.read()

    if isinstance(contents, str):
        data = pd.read_csv(StringIO(contents), delimiter=',')
    else:
        data = pd.read_csv(BytesIO(contents), delimiter=',')

    data = preprocess(data, date_column, target_column)
    series = data[target_column]
    train_size = int(len(series) * 0.9)
    train, test = series[:train_size], series[train_size:]

    def create_lagged_features(series, lag=1):
        df = pd.DataFrame(series)
        columns = [df.shift(i) for i in range(1, lag + 1)]
        columns.append(df)
        df = pd.concat(columns, axis=1)
        df.dropna(inplace=True)
        return df

    lag = 1
    train_lagged = create_lagged_features(train, lag)
    test_lagged = create_lagged_features(test, lag)

    X_train = train_lagged.iloc[:, :-1].values
    y_train = train_lagged.iloc[:, -1].values
    X_test = test_lagged.iloc[:, :-1].values
    y_test = test_lagged.iloc[:, -1].values

    gb_model = GradientBoostingRegressor()
    gb_model.fit(X_train, y_train)

    gb_forecast_test = gb_model.predict(X_test)
    gb_mae_test = mean_absolute_error(y_test, gb_forecast_test)
    gb_mse_test = mean_squared_error(y_test, gb_forecast_test)

    forecast_duration = (pd.to_datetime(end) - pd.to_datetime(start)).days
    last_values = X_test[-1].reshape(1, -1)
    gb_forecast_future = []

    for _ in range(forecast_duration):
        next_value = gb_model.predict(last_values)[0]
        gb_forecast_future.append(next_value)
        last_values = np.roll(last_values, -1)
        last_values[0, -1] = next_value

    mlp_model = MLPRegressor(
        hidden_layer_sizes=(100,),
        activation='relu',
        solver='adam',
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=200,
        alpha=0.0001,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.1
    )
    mlp_model.fit(np.arange(len(train)).reshape(-1, 1), train.values)
    mlp_forecast_test = mlp_model.predict(np.arange(len(train), len(train) + len(test)).reshape(-1, 1))
    mlp_mae_test = mean_absolute_error(test.values, mlp_forecast_test)
    mlp_mse_test = mean_squared_error(test.values, mlp_forecast_test)

    mlp_forecast_future = mlp_model.predict(np.arange(len(series), len(series) + forecast_duration).reshape(-1, 1))

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