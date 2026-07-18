import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from .logger import get_logger
from .exceptions import ValuationError

logger = get_logger(__name__)

class ARIMAModelArchitecture:
    def __init__(self, p: int = 1, d: int = 0, q: int = 0):
        self.p = p
        self.d = d
        self.q = q
        self.model_fit = None
        self.fitted_features = []
        self.train_len = 0

    def fit_model(self, endog: pd.Series, exog_full: pd.DataFrame, feature_cols: list):
        self.train_len = len(endog)
        exog = exog_full[feature_cols].fillna(0.0)
        exog = exog.loc[:, exog.nunique() > 1]
        
        if exog.empty:
            exog = None
            self.fitted_features = []
        else:
            self.fitted_features = list(exog.columns)
            logger.info(f"Filtered training features: {self.fitted_features}")

        try:
            model = ARIMA(endog, order=(self.p, self.d, self.q), exog=exog)
            self.model_fit = model.fit()
            logger.info("ARIMA model fitted successfully")
            return self.model_fit
        except Exception as e:
            logger.error(f"Failed to fit ARIMA: {e}")
            raise ValuationError(f"Model fitting failed: {e}") from e

    def predict_in_sample(self, endog: pd.Series, exog_full: pd.DataFrame) -> pd.Series:
        """
        Walk-forward 5-year forecast at each training date.
        At each date t, trains on data[0..t-1] and forecasts 1260 steps (5 years) ahead.
        The mean of those 1260 forecasts is the predicted 5-year growth rate at t —
        matching the same horizon as the actual growth target.
        Subsamples every STEP_DAYS and interpolates for efficiency.
        """
        if self.model_fit is None:
            raise ValuationError("Model has not been trained yet")

        n = len(endog)
        FORECAST_HORIZON = 1260   # 5 years in trading days
        STEP = 5                  # refit every 5 days, then interpolate
        min_train = max(self.p + self.q + 2, 10)

        exog = exog_full[self.fitted_features].fillna(0.0) if self.fitted_features else None

        sample_indices = []
        sample_preds = []

        for t in range(min_train, n, STEP):
            try:
                train_endog = endog.iloc[:t]
                train_exog = exog.iloc[:t] if exog is not None else None

                if exog is not None:
                    last_row = exog.iloc[t - 1]
                    exog_forecast = pd.DataFrame(
                        [last_row.values] * FORECAST_HORIZON,
                        columns=exog.columns
                    )
                    exog_forecast.index = range(t, t + FORECAST_HORIZON)
                else:
                    exog_forecast = None

                step_model = ARIMA(train_endog, order=(self.p, self.d, self.q), exog=train_exog)
                step_fit = step_model.fit()
                forecast = step_fit.forecast(steps=FORECAST_HORIZON, exog=exog_forecast)
                pred = float(forecast.mean())
            except Exception:
                pred = float(endog.iloc[:t].mean())

            sample_indices.append(t)
            sample_preds.append(pred)

        # Interpolate sample predictions across all n indices
        sparse = pd.Series(sample_preds, index=sample_indices)
        full = sparse.reindex(range(n)).interpolate(method="linear").ffill().bfill()
        if sample_preds:
            full.iloc[:min_train] = sample_preds[0]

        logger.info(f"Walk-forward 5-year forecast complete: {n} steps ({len(sample_indices)} refits at step={STEP})")
        return pd.Series(full.values, index=endog.index)

    def forecast_out_of_sample(self, steps: int, last_exog: pd.Series) -> pd.Series:
        if self.model_fit is None:
            raise ValuationError("Model has not been trained yet")
            
        exog_forecast = None
        if self.fitted_features:
            exog_forecast = pd.DataFrame([last_exog[self.fitted_features]] * steps)
            exog_forecast.index = range(self.train_len, self.train_len + steps)
            
        try:
            forecast = self.model_fit.forecast(steps=steps, exog=exog_forecast)
            return forecast
        except Exception as e:
            logger.error(f"Forecasting failed: {e}")
            raise ValuationError(f"Forecasting failed: {e}") from e
