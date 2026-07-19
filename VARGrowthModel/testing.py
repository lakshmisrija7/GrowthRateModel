import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from .training import VARTrainer
from .architecture import VectorAutoregressionModel
from IntrinsicValueModels import DividendDiscountModel, DiscountedCashFlowModel

REFIT_EVERY = 20

class VARTester:
    def __init__(self, trainer: VARTrainer = None):
        self.trainer = trainer or VARTrainer()

    def _walk_forward_predictions(self, df_growth, ohlcv_list, lags, alpha, close_idx):
        n = len(df_growth)
        data_np = df_growth.to_numpy()
        actual_prices = [item.close for item in ohlcv_list]

        current_model = None
        wf_dates_idx = []
        wf_predicted = []

        for t in range(lags, n):
            if t % REFIT_EVERY == 0 or current_model is None:
                temp_data = data_np[:t]
                if len(temp_data) > lags:
                    current_model = VectorAutoregressionModel(lags=lags, alpha=alpha)
                    current_model.fit(temp_data)

            if current_model is None:
                continue

            past = data_np[:t]
            std_past = (past - current_model.means) / current_model.stds
            row = [1.0]
            for lag in range(1, current_model.lags + 1):
                row.extend(std_past[-lag])
            pred_std = np.array(row) @ current_model.coefficients
            pred_growth = pred_std * current_model.stds + current_model.means
            pred_g = float(np.clip(pred_growth[close_idx], -0.15, 0.15))
            pred_price = actual_prices[t - 1] * (1.0 + pred_g)

            wf_dates_idx.append(t)
            wf_predicted.append(pred_price)

        return wf_dates_idx, wf_predicted

    def evaluate(self, ohlcv_list: list, scores_list: list = None, fundamental_list: list = None, results_dir: str = None, client = None) -> dict:
        if not ohlcv_list or len(ohlcv_list) <= self.trainer.lags:
            raise ValueError("No data or insufficient data available for testing")

        df_growth = self.trainer.prepare_data(ohlcv_list, scores_list, fundamental_list)
        n_samples = len(df_growth)
        train_size = int(n_samples * 0.8)

        train_np = df_growth.to_numpy()[:train_size]
        test_np = df_growth.to_numpy()[train_size:]

        lags_train, alpha_train = self.trainer.tune_hyperparameters(train_np)
        model = VectorAutoregressionModel(lags=lags_train, alpha=alpha_train)
        model.fit(train_np)

        test_steps = len(test_np)
        forecasts = model.forecast(train_np, steps=test_steps)

        close_idx = list(df_growth.columns).index("close")
        actual_close_growth = test_np[:, close_idx]
        pred_close_growth = forecasts[:, close_idx]

        mse = np.mean((actual_close_growth - pred_close_growth) ** 2)
        mae = np.mean(np.abs(actual_close_growth - pred_close_growth))

        lags_full, alpha_full = self.trainer.tune_hyperparameters(df_growth.to_numpy())
        full_model = VectorAutoregressionModel(lags=lags_full, alpha=alpha_full)
        full_model.fit(df_growth.to_numpy())

        symbol = ohlcv_list[0].symbol if ohlcv_list else "VAR"

        future_ohlcv = []
        if client:
            last_item = ohlcv_list[-1]
            future_from_ms = str(last_item.time)
            future_to_ms = str(int(last_item.time) + 2 * 365 * 24 * 60 * 60 * 1000)
            try:
                future_response = client.fetch(symbol, future_from_ms, future_to_ms)
                if future_response and future_response.data:
                    future_ohlcv = future_response.data
            except Exception:
                pass

        if results_dir:
            price_comp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "price_comparison_results")
            os.makedirs(price_comp_dir, exist_ok=True)

            actual_prices = [item.close for item in ohlcv_list]
            actual_dates = [pd.to_datetime(item.time, unit="ms") for item in ohlcv_list]

            wf_idx, wf_predicted = self._walk_forward_predictions(
                df_growth, ohlcv_list, lags_full, alpha_full, close_idx
            )
            wf_actual = [actual_prices[i] for i in wf_idx]
            wf_dates = [actual_dates[i] for i in wf_idx]

            boundary_date = actual_dates[-1]

            if future_ohlcv:
                outsample_dates = [pd.to_datetime(item.time, unit="ms") for item in future_ohlcv]
                outsample_actual = [item.close for item in future_ohlcv]
                steps = len(future_ohlcv)
            else:
                outsample_dates = pd.date_range(
                    start=actual_dates[-1] + pd.offsets.BDay(), periods=504, freq="B"
                )
                outsample_actual = [actual_prices[-1]] * len(outsample_dates)
                steps = 504

            future_forecasts = full_model.forecast(df_growth.to_numpy(), steps=steps)

            log_returns_hist = np.diff(np.log(np.maximum(actual_prices, 1e-6)))
            daily_log_std = float(np.std(log_returns_hist)) if len(log_returns_hist) > 1 else 0.01

            rng = np.random.default_rng(42)
            outsample_pred = []
            log_current = np.log(max(actual_prices[-1], 1e-6))
            for pred_g in future_forecasts[:, close_idx]:
                clipped_g = float(np.clip(pred_g, -0.05, 0.05))
                noise = rng.normal(0, daily_log_std)
                log_current = log_current + np.log(1.0 + clipped_g) + noise
                outsample_pred.append(np.exp(log_current))

            insample_dates = wf_dates
            insample_actual = wf_actual
            insample_pred = wf_predicted

            fig, ax = plt.subplots(figsize=(16, 6))

            ax.plot(insample_dates, insample_actual,
                    color="steelblue", alpha=0.8, linewidth=1.2, label="Actual Stock Price")
            ax.plot(insample_dates, insample_pred,
                    color="tomato", linestyle="--", alpha=0.85, linewidth=1.2, label="Predicted Stock Price")

            if list(outsample_dates):
                ax.plot(outsample_dates, outsample_actual,
                        color="steelblue", alpha=0.8, linewidth=1.2)
                ax.plot(outsample_dates, outsample_pred,
                        color="tomato", linestyle="--", alpha=0.85, linewidth=1.2)

            ax.axvline(x=boundary_date, color="dimgray", linestyle=":", linewidth=1.5,
                       label=f"Training End ({boundary_date.date()})")

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            fig.autofmt_xdate(rotation=45)

            ax.set_title(f"VAR: Actual vs Predicted 2-Year Forward Stock Price ({symbol}, from 2015)")
            ax.set_xlabel("Date")
            ax.set_ylabel("Stock Price (USD)")
            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.5)

            plt.tight_layout()
            plt.savefig(os.path.join(price_comp_dir, f"{symbol}_price_comparison.png"), dpi=300, bbox_inches="tight")
            plt.close()

        future_forecasts_full = full_model.forecast(df_growth.to_numpy(), steps=504)
        annual_growths = []
        for year in range(2):
            start_idx = year * 252
            end_idx = (year + 1) * 252
            year_forecasts = future_forecasts_full[start_idx:end_idx, close_idx]
            annual_g = np.prod(1.0 + year_forecasts) - 1.0
            annual_growths.append(annual_g)

        avg_growth = float(np.mean(annual_growths))

        ddm = DividendDiscountModel(last_dividend=2.00, discount_rate=0.08)
        dcf = DiscountedCashFlowModel(initial_fcf=150.0, discount_rate=0.09)

        ddm_g = min(avg_growth, 0.07)
        dcf_rates = [min(g, 0.08) for g in annual_growths]

        ddm_val = ddm.calculate_constant_growth(growth_rate=ddm_g)
        dcf_val = dcf.calculate_valuation(
            growth_rates=dcf_rates,
            forecast_years=2,
            terminal_growth_rate=0.03,
            net_debt=200.0,
            shares_outstanding=10.0
        )

        return {
            "mse": mse,
            "mae": mae,
            "annual_growths": annual_growths,
            "ddm_intrinsic_value": ddm_val,
            "dcf_intrinsic_value": dcf_val,
            "daily_forecasts": future_forecasts_full[:10, close_idx].tolist()
        }
