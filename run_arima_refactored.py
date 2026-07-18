import asyncio
from ARIMAGrowthModel import ARIMADataLoader, ARIMAModelArchitecture, ARIMATrainer, ARIMATester
from ARIMAGrowthModel.logger import get_logger

logger = get_logger("runner")

SYMBOLS = ["UBER", "NVDA", "APP", "LLY", "AVGO"]

# Training window: Jan 2015 – Jan 2020
FROM_DATE = 1420070400000
TO_DATE   = 1577836800000

FEATURE_COLS = [
    "tech_trend", "tech_momentum", "tech_volatility", "tech_rsi", "tech_adx",
    "sec_pe", "sec_roe", "sec_npm", "sec_sector_pe", "sec_sector_roe",
    "sent_score", "sent_confidence"
]
FUND_COLS = ["fund_cashHealth", "fund_leverage", "fund_liquidity", "fund_profitability"]


async def run_symbol(symbol: str):
    logger.info(f"\n{'='*60}")
    logger.info(f"Running ARIMA for symbol: {symbol}")
    logger.info(f"{'='*60}")

    loader = ARIMADataLoader(symbol=symbol)
    try:
        df = await loader.load_and_align_data(FROM_DATE, TO_DATE)
        logger.info(f"[{symbol}] Loaded aligned dataset. Final symbol: {loader.symbol}. Rows: {len(df)}")

        feature_cols = list(FEATURE_COLS)
        for col in FUND_COLS:
            if col in df.columns:
                feature_cols.append(col)

        model_arch = ARIMAModelArchitecture(p=1, d=0, q=0)

        trainer = ARIMATrainer(model_arch)
        trainer.train_model(df, feature_cols)

        tester = ARIMATester(model_arch)
        report = await tester.test_and_evaluate(df, feature_cols, loader)

        logger.info(f"[{symbol}] === ARIMA Report ===")
        logger.info(f"[{symbol}] In-Sample MSE: {report['mse']:.6f}")
        logger.info(f"[{symbol}] In-Sample MAE: {report['mae']:.6f}")
        logger.info(f"[{symbol}] Forecasted Annual Growth Rates (5 Yrs): {report['forecasted_growth_rates']}")
        logger.info(f"[{symbol}] DCF Intrinsic Value: ${report['intrinsic_value']:.2f}")
        logger.info(f"[{symbol}] Plot saved to results/{symbol}_valuation_forecast.png")

    except Exception as e:
        logger.error(f"[{symbol}] Execution failed: {e}", exc_info=True)


async def main():
    for symbol in SYMBOLS:
        await run_symbol(symbol)
    logger.info("\nAll symbols processed.")


if __name__ == "__main__":
    asyncio.run(main())
