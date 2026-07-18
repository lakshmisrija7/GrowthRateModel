import os
import numpy as np
import matplotlib.pyplot as plt
from .training import VARTrainer
from .architecture import VectorAutoregressionModel
from IntrinsicValueModels import DividendDiscountModel, DiscountedCashFlowModel

class VARTester:
    def __init__(self, trainer: VARTrainer = None):
        self.trainer = trainer or VARTrainer()

    def evaluate(self, ohlcv_list: list, scores_list: list = None, results_dir: str = None) -> dict:
        if not ohlcv_list or len(ohlcv_list) <= self.trainer.lags:
            raise ValueError("No data or insufficient data available for testing")

        df_growth = self.trainer.prepare_data(ohlcv_list, scores_list)
        n_samples = len(df_growth)
        train_size = int(n_samples * 0.8)
        
        train_np = df_growth.to_numpy()[:train_size]
        test_np = df_growth.to_numpy()[train_size:]

        model = VectorAutoregressionModel(lags=self.trainer.lags)
        model.fit(train_np)
        
        test_steps = len(test_np)
        forecasts = model.forecast(train_np, steps=test_steps)

        close_idx = list(df_growth.columns).index("close")
        actual_close_growth = test_np[:, close_idx]
        pred_close_growth = forecasts[:, close_idx]

        mse = np.mean((actual_close_growth - pred_close_growth) ** 2)
        mae = np.mean(np.abs(actual_close_growth - pred_close_growth))

        full_model = VectorAutoregressionModel(lags=self.trainer.lags)
        full_model.fit(df_growth.to_numpy())

        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
            import pandas as pd
            import matplotlib.dates as mdates

            symbol = ohlcv_list[0].symbol if ohlcv_list else "VAR"
            actual_prices = [item.close for item in ohlcv_list]
            actual_dates = [pd.to_datetime(item.time, unit="ms") for item in ohlcv_list]

            lags = full_model.lags
            predicted_prices = list(actual_prices)
            for t in range(lags, len(df_growth)):
                row = [1.0]
                for lag in range(1, lags + 1):
                    row.extend(df_growth.to_numpy()[t - lag])
                pred_growth = np.array(row) @ full_model.coefficients
                pred_close_g = pred_growth[close_idx]
                predicted_prices[t + 1] = ohlcv_list[t].close * (1.0 + pred_close_g)

            future_forecasts = full_model.forecast(df_growth.to_numpy(), steps=1260)
            current_price = predicted_prices[-1]
            for pred_g in future_forecasts[:, close_idx]:
                next_price = current_price * (1.0 + pred_g)
                predicted_prices.append(next_price)
                current_price = next_price

            last_date = actual_dates[-1]
            future_dates = pd.date_range(start=last_date + pd.offsets.BDay(), periods=1260, freq="B")
            predicted_dates = actual_dates + list(future_dates)

            plt.figure(figsize=(16, 6))
            plt.plot(actual_dates, actual_prices, label="Actual Growth (Daily)", color="#3b4cc0", linewidth=1.0)
            plt.plot(predicted_dates, predicted_prices, label="In-Sample Predicted Growth", color="#ff3b30", linestyle="--", linewidth=1.0)
            
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            plt.gca().xaxis.set_major_locator(mdates.YearLocator())
            plt.gcf().autofmt_xdate()

            plt.title(f"VAR Actual vs Predicted Growth Rate - {symbol}", fontsize=12)
            plt.xlabel("Timeline", fontsize=10)
            plt.ylabel("Growth Rate", fontsize=10)
            plt.legend(loc="upper left")
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"{symbol}_forecast.png"), dpi=300)
            plt.close()

        future_forecasts = full_model.forecast(df_growth.to_numpy(), steps=1260)
        
        annual_growths = []
        for year in range(5):
            start_idx = year * 252
            end_idx = (year + 1) * 252
            year_forecasts = future_forecasts[start_idx:end_idx, close_idx]
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
            forecast_years=5,
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
            "daily_forecasts": future_forecasts[:10, close_idx].tolist()
        }
