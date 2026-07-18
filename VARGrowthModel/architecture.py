import numpy as np

class VectorAutoregressionModel:
    def __init__(self, lags: int = 5):
        self.lags = lags
        self.coefficients = None

    def fit(self, data: np.ndarray) -> None:
        n_samples, n_features = data.shape
        lags = self.lags
        if n_samples <= lags:
            raise ValueError("Not enough data to fit VAR model with specified lags")

        z_list = []
        x_list = []
        for t in range(lags, n_samples):
            z_list.append(data[t])
            row = [1.0]
            for lag in range(1, lags + 1):
                row.extend(data[t - lag])
            x_list.append(row)

        z = np.array(z_list)
        x = np.array(x_list)
        self.coefficients = np.linalg.pinv(x.T @ x) @ x.T @ z

    def forecast(self, history: np.ndarray, steps: int) -> np.ndarray:
        lags = self.lags
        forecasts = []
        current_history = list(history[-lags:])

        for _ in range(steps):
            row = [1.0]
            for lag in range(1, lags + 1):
                row.extend(current_history[-lag])
            pred = np.array(row) @ self.coefficients
            forecasts.append(pred)
            current_history.append(pred)

        return np.array(forecasts)
