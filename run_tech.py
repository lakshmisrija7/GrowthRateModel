import asyncio
import sys
from TechnicalAnalysis.client import TechnicalWebSocketClient
from TechnicalAnalysis.config import WS_URL, API_KEY, DEFAULT_FROM_DATE, DEFAULT_TO_DATE
from TechnicalAnalysis.logger import get_logger

logger = get_logger("runner")

async def main():
    symbol = "NVDA"
    from_date = DEFAULT_FROM_DATE
    to_date = DEFAULT_TO_DATE
    
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
    if len(sys.argv) > 3:
        try:
            from_date = int(sys.argv[2])
            to_date = int(sys.argv[3])
        except ValueError:
            pass
    elif symbol == "NVDA":
        from_date = None
        to_date = None
        
    client = TechnicalWebSocketClient(WS_URL, api_key=API_KEY)
    try:
        await client.connect()
        response = await client.fetch_technical_analysis(symbol, from_date, to_date)
        mapped_entries = response.get_mapped_entries()
        logger.info(f"Retrieved {len(mapped_entries)} technical analysis entries for {symbol}:")
        for entry in mapped_entries:
            logger.info(f"--- Time: {entry.get('time')} ---")
            for k, v in entry.items():
                if k not in ("symbol", "time"):
                    logger.info(f"  {k}: {v}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
