import numpy as np

class VectorAutoregressionModel:
    def __init__(self, lags: int = 5, alpha: float = 1.0):
        self.lags = lags
        self.alpha = alpha
        self.coefficients = None
        self.means = None
        self.stds = None

    def fit(self, data: np.ndarray) -> None:
        n_samples, n_features = data.shape
        lags = self.lags
        if n_samples <= lags:
            raise ValueError("Not enough data to fit VAR model with specified lags")

        self.means = np.mean(data, axis=0)
        self.stds = np.std(data, axis=0)
        self.stds[self.stds == 0.0] = 1.0

        std_data = (data - self.means) / self.stds

        z_list = []
        x_list = []
        for t in range(lags, n_samples):
            z_list.append(std_data[t])
            row = [1.0]
            for lag in range(1, lags + 1):
                row.extend(std_data[t - lag])
            x_list.append(row)

        z = np.array(z_list)
        x = np.array(x_list)

        n_col = x.shape[1]
        identity = np.eye(n_col)
        identity[0, 0] = 0.0

        self.coefficients = np.linalg.pinv(x.T @ x + self.alpha * identity) @ x.T @ z

    def forecast(self, history: np.ndarray, steps: int) -> np.ndarray:
        lags = self.lags
        std_history = list((history - self.means) / self.stds)
        forecasts_std = []

        for _ in range(steps):
            row = [1.0]
            for lag in range(1, lags + 1):
                row.extend(std_history[-lag])
            pred_std = np.array(row) @ self.coefficients
            forecasts_std.append(pred_std)
            std_history.append(pred_std)

        forecasts_std = np.array(forecasts_std)
        forecasts = forecasts_std * self.stds + self.means
        return forecasts
