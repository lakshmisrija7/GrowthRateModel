import numpy as np
import pandas as pd

class KalmanDFM:
    def __init__(self, phi: float = 0.95, q: float = 0.1, h_r: float = 1.0, h_f: float = 0.5):
        self.phi = phi
        self.q = q
        self.h_r = h_r
        self.h_f = h_f
        self.lambda_r = 0.1
        self.lambda_f = 1.0
        self.f_mean = 5.0

    def fit_and_filter(self, daily_returns: pd.Series, sparse_quarterly: pd.Series) -> pd.Series:
        idx_report = sparse_quarterly.dropna().index
        if len(idx_report) >= 3:
            self.f_mean = float(sparse_quarterly.mean())
            y_r = []
            y_f = []
            for d in idx_report:
                if d in daily_returns.index:
                    y_r.append(daily_returns.loc[d])
                    y_f.append(sparse_quarterly.loc[d] - self.f_mean)
            
            if len(y_r) >= 3:
                y_r = np.array(y_r)
                y_f = np.array(y_f)
                denom = np.sum(y_f ** 2)
                if denom > 0:
                    self.lambda_r = np.sum(y_r * y_f) / denom
                self.h_r = max(float(np.var(daily_returns)), 0.01)
                self.h_f = max(float(np.var(sparse_quarterly.dropna())), 0.01)

        returns_np = daily_returns.to_numpy()
        dates_np = daily_returns.index.to_numpy()
        n = len(daily_returns)

        f_state = np.zeros(n)
        p_state = np.zeros(n)
        
        f_val = 0.0
        p_val = 1.0

        for t in range(n):
            f_pred = self.phi * f_val
            p_pred = (self.phi ** 2) * p_val + self.q

            date_t = dates_np[t]
            val_f = sparse_quarterly.loc[date_t] if date_t in sparse_quarterly.index else np.nan
            
            r_val = returns_np[t]

            if not np.isnan(val_f):
                y = np.array([[r_val], [val_f - self.f_mean]])
                C = np.array([[self.lambda_r], [self.lambda_f]])
                R_cov = np.array([[self.h_r, 0.0], [0.0, self.h_f]])
                
                S = C @ np.array([[p_pred]]) @ C.T + R_cov
                S_inv = np.linalg.inv(S)
                K = p_pred * C.T @ S_inv
                
                innov = y - C * f_pred
                f_val = f_pred + float(K @ innov)
                p_val = float((1.0 - K @ C) * p_pred)
            else:
                y = r_val
                C = self.lambda_r
                R_cov = self.h_r
                
                S = (C ** 2) * p_pred + R_cov
                K = (p_pred * C) / S
                innov = y - C * f_pred
                f_val = f_pred + K * innov
                p_val = (1.0 - K * C) * p_pred

            f_state[t] = f_val
            p_state[t] = p_val

        predictions = self.f_mean + self.lambda_f * f_state
        predictions = np.clip(predictions, 0.0, 10.0)
        return pd.Series(predictions, index=daily_returns.index)
