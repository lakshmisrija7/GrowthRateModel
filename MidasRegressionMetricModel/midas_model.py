import numpy as np
import pandas as pd

class MidasRegression:
    def __init__(self, lag_length: int = 60):
        self.lag_length = lag_length
        self.beta0 = 0.0
        self.beta1 = 0.0
        self.theta = 0.1

    def fit(self, daily_returns: pd.Series, sparse_quarterly: pd.Series) -> None:
        idx_report = sparse_quarterly.dropna().index
        if len(idx_report) < 3:
            self.beta0 = float(sparse_quarterly.mean()) if not sparse_quarterly.empty else 5.0
            self.beta1 = 0.0
            return

        returns_np = daily_returns.to_numpy()
        dates_np = daily_returns.index.to_numpy()
        report_dates = idx_report.to_numpy()

        best_mse = float("inf")
        best_params = (0.0, 0.0, 0.1)

        theta_grid = np.linspace(0.001, 0.5, 50)
        
        y_train = []
        for d in report_dates:
            y_train.append(sparse_quarterly.loc[d])
        y_train = np.array(y_train)

        for theta in theta_grid:
            x_train = []
            for d in report_dates:
                matches = np.where(dates_np == d)[0]
                if len(matches) == 0:
                    x_train.append(0.0)
                    continue
                t_idx = matches[0]
                if t_idx < self.lag_length:
                    x_train.append(0.0)
                    continue
                
                weights = np.exp(-theta * np.arange(1, self.lag_length + 1))
                weights /= np.sum(weights)
                lagged_returns = returns_np[t_idx - self.lag_length : t_idx]
                x_val = np.sum(weights * lagged_returns[::-1])
                x_train.append(x_val)
            
            x_train = np.array(x_train)
            
            x_mat = np.column_stack((np.ones_like(x_train), x_train))
            try:
                beta = np.linalg.pinv(x_mat.T @ x_mat) @ x_mat.T @ y_train
                preds = x_mat @ beta
                mse = np.mean((y_train - preds) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    best_params = (beta[0], beta[1], theta)
            except Exception:
                continue

        self.beta0, self.beta1, self.theta = best_params

    def predict(self, daily_returns: pd.Series) -> pd.Series:
        returns_np = daily_returns.to_numpy()
        n = len(daily_returns)
        
        predictions = np.zeros(n)
        weights = np.exp(-self.theta * np.arange(1, self.lag_length + 1))
        weights /= np.sum(weights)

        for t_idx in range(n):
            if t_idx < self.lag_length:
                predictions[t_idx] = self.beta0
                continue
            
            lagged_returns = returns_np[t_idx - self.lag_length : t_idx]
            x_val = np.sum(weights * lagged_returns[::-1])
            predictions[t_idx] = self.beta0 + self.beta1 * x_val

        return pd.Series(predictions, index=daily_returns.index)
