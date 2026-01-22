import asyncio
import sys
from FundamentalAnalysis.client import FundamentalWebSocketClient
from FundamentalAnalysis.config import WS_URL, API_KEY, DEFAULT_FROM_DATE, DEFAULT_TO_DATE
from FundamentalAnalysis.logger import get_logger

logger = get_logger("runner")

async def main():
    symbol = "NVDA"
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
        
    client = FundamentalWebSocketClient(WS_URL, api_key=API_KEY)
    try:
        await client.connect()
        response = await client.fetch_fundamentals(symbol, DEFAULT_FROM_DATE, DEFAULT_TO_DATE)
        periods = response.get_periods()
        logger.info(f"Found periods: {periods}")
        for period in periods:
            logger.info(f"=== Period: {period} ===")
            scores = response.get_scores(period)
            logger.info(f"Scores for {period}:")
            for k, v in scores.items():
                logger.info(f"  {k}: {v}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
