import numpy as np
import pandas as pd
from .architecture import VectorAutoregressionModel

class VARTrainer:
    def __init__(self, lags: int = 5):
        self.lags = lags

    def prepare_data(self, ohlcv_list: list, scores_list: list = None) -> pd.DataFrame:
        ohlcv_data = []
        for item in ohlcv_list:
            dt = pd.to_datetime(item.time, unit="ms").date()
            ohlcv_data.append({
                "date": dt,
                "open": item.open,
                "high": item.high,
                "low": item.low,
                "close": item.close,
                "volume": item.volume
            })
        df_ohlcv = pd.DataFrame(ohlcv_data).set_index("date")

        if scores_list:
            scores_data = []
            for s in scores_list:
                ts = s.get("timestamp")
                if ts:
                    dt = pd.to_datetime(ts, unit="ms").date()
                    scores_data.append({
                        "date": dt,
                        "overallScore": s.get("overallScore"),
                        "technicalScore": s.get("technicalScore"),
                        "fundamentalScore": s.get("fundamentalScore"),
                        "sentimentScore": s.get("sentimentScore"),
                        "industryScore": s.get("industryScore"),
                        "riskScore": s.get("riskScore")
                    })
            if scores_data:
                df_scores = pd.DataFrame(scores_data)
                df_scores = df_scores.replace('-', np.nan)
                score_cols = ["overallScore", "technicalScore", "fundamentalScore", "sentimentScore", "industryScore", "riskScore"]
                for col in score_cols:
                    if col in df_scores.columns:
                        df_scores[col] = pd.to_numeric(df_scores[col], errors='coerce')
                df_scores = df_scores.groupby("date").mean()
            else:
                df_scores = pd.DataFrame(columns=["overallScore", "technicalScore", "fundamentalScore", "sentimentScore", "industryScore", "riskScore"])
        else:
            df_scores = pd.DataFrame(columns=["overallScore", "technicalScore", "fundamentalScore", "sentimentScore", "industryScore", "riskScore"])

        df_price_growth = df_ohlcv.pct_change().dropna()
        df_price_growth = df_price_growth.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        df_merged = df_price_growth.join(df_scores, how="left")
        df_merged = df_merged.ffill().bfill()

        score_cols = ["overallScore", "technicalScore", "fundamentalScore", "sentimentScore", "industryScore", "riskScore"]
        for col in score_cols:
            if col not in df_merged.columns:
                df_merged[col] = 5.0
            else:
                df_merged[col] = df_merged[col].fillna(5.0)

        return df_merged

    def train(self, ohlcv_list: list, scores_list: list = None) -> VectorAutoregressionModel:
        df_growth = self.prepare_data(ohlcv_list, scores_list)
        model = VectorAutoregressionModel(lags=self.lags)
        model.fit(df_growth.to_numpy())
        return model
