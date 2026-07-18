import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from .logger import get_logger
from .exceptions import ValuationError

logger = get_logger(__name__)

class ARIMAGrowthModelPredictor:
    def __init__(self, p: int = 1, d: int = 0, q: int = 0):
        self.p = p
        self.d = d
        self.q = q
        self.model_fit = None
        self.fitted_features = []

    def train(self, df: pd.DataFrame, feature_cols: list) -> tuple:
        if df.empty:
            raise ValuationError("Training dataframe is empty")
        if len(df) < 5:
            raise ValuationError(f"Not enough data points to train model (have {len(df)}, need at least 5)")

        logger.info(f"Training ARIMA({self.p},{self.d},{self.q}) model with {len(df)} rows")
        
        df[feature_cols] = df[feature_cols].fillna(0.0)
        
        endog = df["growth"]
        exog_full = df[feature_cols]
        exog = exog_full.loc[:, exog_full.nunique() > 1]
        
        if exog.empty:
            exog = None
            self.fitted_features = []
        else:
            self.fitted_features = list(exog.columns)
            logger.info(f"Filtered features with variance: {self.fitted_features}")

        try:
            model = ARIMA(endog, order=(self.p, self.d, self.q), exog=exog)
            self.model_fit = model.fit()
            logger.info("ARIMA model fitted successfully")
            
            predictions = self.model_fit.predict(start=0, end=len(df)-1, exog=exog)
            
            errors = endog - predictions
            mse = np.mean(errors ** 2)
            mae = np.mean(np.abs(errors))
            
            logger.info(f"Training error - MSE: {mse:.6f}, MAE: {mae:.6f}")
            return float(mse), float(mae)
        except Exception as e:
            logger.error(f"Failed to fit ARIMA model: {e}")
            raise ValuationError(f"Model fitting failed: {e}") from e

    def predict(self, exog: pd.DataFrame) -> pd.Series:
        if self.model_fit is None:
            raise ValuationError("Model has not been trained yet")
        
        exog = exog.fillna(0.0)
        if self.fitted_features:
            exog = exog[self.fitted_features]
        else:
            exog = None
        
        try:
            forecast = self.model_fit.forecast(steps=1 if exog is None else len(exog), exog=exog)
            return forecast
        except Exception as e:
            logger.error(f"Forecasting failed: {e}")
            raise ValuationError(f"Forecasting failed: {e}") from e
