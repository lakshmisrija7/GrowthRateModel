import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from .architecture import ARIMAModelArchitecture
from .logger import get_logger
from .exceptions import ValuationError
from IntrinsicValueModels import DiscountedCashFlowModel

logger = get_logger(__name__)

MAX_FORWARD_TOLERANCE_DAYS = 30

class ARIMATester:
    def __init__(self, model_arch: ARIMAModelArchitecture):
        self.model_arch = model_arch

    async def test_and_evaluate(self, df: pd.DataFrame, feature_cols: list, loader) -> dict:
        logger.info("Evaluating ARIMA model performance")
        endog = df["growth"]
        predictions = self.model_arch.predict_in_sample(endog, df)
        
        errors = endog - predictions
        mse = float(np.mean(errors ** 2))
        mae = float(np.mean(np.abs(errors)))
        
        last_exog = df[feature_cols].iloc[-1]
        forecast_steps = 504
        logger.info(f"Forecasting {forecast_steps} daily steps (2 years) into the future")
        forecast_daily = self.model_arch.forecast_out_of_sample(forecast_steps, last_exog)
        
        annual_growth_rates = []
        for i in range(2):
            segment_mean = float(forecast_daily.iloc[i * 252:(i + 1) * 252].mean())
            annual_growth_rates.append(segment_mean)
        logger.info(f"Calculated 2-year forecasted annual growth rates: {annual_growth_rates}")

        dcf = DiscountedCashFlowModel(initial_fcf=150.0, discount_rate=0.09)
        intrinsic_val = dcf.calculate_valuation(
            growth_rates=annual_growth_rates,
            forecast_years=2,
            terminal_growth_rate=0.03,
            net_debt=200.0,
            shares_outstanding=10.0
        )
        logger.info(f"Computed Intrinsic Value per share from forecasted growth rates: ${intrinsic_val:.2f}")

        df_insample = pd.DataFrame({
            "date": pd.to_datetime(df["date"].values),
            "actual_growth": endog.values,
            "predicted_growth": predictions.values
        })

        max_date = df["date"].max()
        max_dt = datetime.datetime.combine(max_date, datetime.time.min, tzinfo=datetime.timezone.utc)
        to_date_ms = int(max_dt.timestamp() * 1000)
        ms_2y = 2 * 365 * 24 * 60 * 60 * 1000
        future_from_ms = to_date_ms
        future_to_ms = to_date_ms + ms_2y
        future_2y_fwd_to_ms = future_to_ms + ms_2y

        logger.info(f"Fetching future base window {future_from_ms} to {future_to_ms} and forward window {future_to_ms} to {future_2y_fwd_to_ms} for {loader.symbol}")
        ohlcv_future_base = await loader._fetch_ohlcv_data(loader.symbol, future_from_ms, future_to_ms, real_time=False)
        ohlcv_future_fwd = await loader._fetch_ohlcv_data(loader.symbol, future_to_ms, future_2y_fwd_to_ms, real_time=False)

        daily_closes_future_base = {}
        for item in ohlcv_future_base:
            dt = datetime.datetime.fromtimestamp(item["time"] / 1000, datetime.timezone.utc).date()
            daily_closes_future_base[dt] = item["close"]

        daily_closes_future_fwd = {}
        for item in ohlcv_future_fwd:
            dt = datetime.datetime.fromtimestamp(item["time"] / 1000, datetime.timezone.utc).date()
            daily_closes_future_fwd[dt] = item["close"]

        dates_fwd = sorted(daily_closes_future_fwd.keys())

        forecast_list = list(forecast_daily)
        out_dates = []
        out_actual = []
        out_predicted = []

        for i, (d, close_d) in enumerate(sorted(daily_closes_future_base.items())):
            d_2y_fwd_target = d + datetime.timedelta(days=2 * 365)
            actual_growth = None
            if dates_fwd:
                closest_d = min(dates_fwd, key=lambda x: abs((x - d_2y_fwd_target).days))
                if abs((closest_d - d_2y_fwd_target).days) <= MAX_FORWARD_TOLERANCE_DAYS:
                    actual_growth = (daily_closes_future_fwd[closest_d] - close_d) / close_d

            out_dates.append(pd.Timestamp(d))
            out_actual.append(actual_growth)
            out_predicted.append(forecast_list[i] if i < len(forecast_list) else None)

        df_outsample = pd.DataFrame({
            "date": out_dates,
            "actual_growth": out_actual,
            "predicted_growth": out_predicted
        })

        valid_actuals = df_outsample.dropna(subset=["actual_growth"])
        logger.info(f"Out-of-sample window: {len(df_outsample)} total dates, {len(valid_actuals)} with valid actual growth")

        self._save_results_plot(df_insample, df_outsample, symbol=loader.symbol)

        return {
            "mse": mse,
            "mae": mae,
            "forecasted_growth_rates": annual_growth_rates,
            "intrinsic_value": float(intrinsic_val)
        }

    def _save_results_plot(self, df_insample: pd.DataFrame, df_outsample: pd.DataFrame, symbol: str = "STOCK"):
        results_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(results_dir, exist_ok=True)
        plot_path = os.path.join(results_dir, f"{symbol}_valuation_forecast.png")

        fig, ax = plt.subplots(figsize=(16, 6))

        ax.plot(
            df_insample["date"], df_insample["actual_growth"],
            label="Actual Growth Rate", color="steelblue", alpha=0.8, linewidth=1.2
        )
        ax.plot(
            df_insample["date"], df_insample["predicted_growth"],
            label="Predicted Growth Rate", color="tomato", linestyle="--", alpha=0.85, linewidth=1.2
        )

        df_out_valid_actual = df_outsample.dropna(subset=["actual_growth"])
        if not df_out_valid_actual.empty:
            ax.plot(
                df_out_valid_actual["date"], df_out_valid_actual["actual_growth"],
                color="steelblue", alpha=0.8, linewidth=1.2
            )

        df_out_pred = df_outsample.dropna(subset=["predicted_growth"])
        if not df_out_pred.empty:
            ax.plot(
                df_out_pred["date"], df_out_pred["predicted_growth"],
                color="tomato", linestyle="--", alpha=0.85, linewidth=1.2
            )

        boundary_date = df_insample["date"].max()
        ax.axvline(
            x=boundary_date, color="dimgray", linestyle=":", linewidth=1.5,
            label=f"Training End ({boundary_date.date()})"
        )

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        fig.autofmt_xdate(rotation=45)

        ax.set_title(f"ARIMA: Actual vs Predicted 2-Year Forward Growth Rate ({symbol}, from 2015)")
        ax.set_xlabel("Date")
        ax.set_ylabel("2-Year Forward Growth Rate")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved visualization results plot to: {plot_path}")
