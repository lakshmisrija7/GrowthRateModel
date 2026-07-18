import numpy as np
import pandas as pd
from .architecture import VectorAutoregressionModel

class VARTrainer:
    def __init__(self, lags: int = 5):
        self.lags = lags

    def prepare_data(self, ohlcv_list: list) -> pd.DataFrame:
        df = pd.DataFrame([
            {
                "open": item.open,
                "high": item.high,
                "low": item.low,
                "close": item.close,
                "volume": item.volume
            }
            for item in ohlcv_list
        ])
        
        df_growth = df.pct_change().dropna()
        df_growth = df_growth.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return df_growth

    def train(self, ohlcv_list: list) -> VectorAutoregressionModel:
        df_growth = self.prepare_data(ohlcv_list)
        model = VectorAutoregressionModel(lags=self.lags)
        model.fit(df_growth.to_numpy())
        return model
