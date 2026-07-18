import asyncio
import datetime
import pandas as pd
from ARIMAGrowthModel.data_loader import ARIMADataLoader
from ARIMAGrowthModel.predictor import ARIMAGrowthModelPredictor
from ARIMAGrowthModel.logger import get_logger

logger = get_logger("runner")

async def main():
    symbol = "NVDA"
    company_name = "NVDA"
    
    to_date = 1784419199000
    from_date = to_date - (30 * 24 * 60 * 60 * 1000)
    
    loader = ARIMADataLoader(symbol, company_name)
    try:
        df = await loader.load_and_align_data(from_date, to_date)
        logger.info(f"Aligned dataset successfully. Rows count: {len(df)}")
        
        feature_cols = [
            "tech_trend", "tech_momentum", "tech_volatility", "tech_rsi", "tech_adx",
            "sec_pe", "sec_roe", "sec_npm", "sec_sector_pe", "sec_sector_roe",
            "sent_score", "sent_confidence"
        ]
        
        fund_cols = ["fund_cashHealth", "fund_leverage", "fund_liquidity", "fund_profitability"]
        for col in fund_cols:
            if col in df.columns:
                feature_cols.append(col)
                
        logger.info(f"Features list: {feature_cols}")
        
        predictor = ARIMAGrowthModelPredictor(p=1, d=0, q=0)
        mse, mae = predictor.train(df, feature_cols)
        
        logger.info(f"Model coefficients: {predictor.model_fit.params.to_dict()}")
        
        df["predicted_growth"] = predictor.model_fit.predict(start=0, end=len(df)-1, exog=df[feature_cols])
        logger.info("Actual Growth vs Predicted Growth:")
        for idx, row in df.tail(5).iterrows():
            logger.info(f"  Date: {row['date']}, Close: ${row['close_curr']:.2f}, 5y Ago Close: ${row['close_5y']:.2f}, Actual Growth: {row['growth']:.4f}, Predicted Growth: {row['predicted_growth']:.4f}")
            
    except Exception as e:
        logger.error(f"Failed execution: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
