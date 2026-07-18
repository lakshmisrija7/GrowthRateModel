import sys
import os
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def fetch_overall_analysis_async(symbol: str, from_date: str, to_date: str) -> list:
    from OverallAnalysis.client import OverallAnalysisClient
    client = OverallAnalysisClient()
    try:
        await client.connect()
        response = await client.fetch_overall_analysis(symbol, from_date, to_date)
        return response.get_all_scores()
    finally:
        await client.disconnect()

def fetch_overall_analysis(symbol: str, from_date: str, to_date: str) -> list:
    try:
        return asyncio.run(fetch_overall_analysis_async(symbol, from_date, to_date))
    except RuntimeError:
        def run_in_new_loop():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(fetch_overall_analysis_async(symbol, from_date, to_date))
            finally:
                new_loop.close()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_in_new_loop)
            return future.result()

def main():
    symbols = ["UBER", "NVDA", "APP", "LLY", "AVGO"]
    from_date = "1577836800000"
    to_date = "1735689599000"

    logging.getLogger("IntrinsicValueModels.ddm").setLevel(logging.WARNING)
    logging.getLogger("IntrinsicValueModels.dcf").setLevel(logging.WARNING)

    from VARGrowthModel.client import VARWebSocketClient
    from VARGrowthModel.testing import VARTester

    client = VARWebSocketClient()
    tester = VARTester()

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VARGrowthModel", "results2")

    for symbol in symbols:
        print(f"\nFetching OHLCV data for {symbol}...")
        response = client.fetch(symbol, from_date, to_date)
        ohlcv_list = response.data

        if not ohlcv_list:
            print(f"Error: No OHLCV data retrieved for {symbol} in the requested range.")
            continue

        print(f"Fetching overall analysis scores for {symbol}...")
        scores_list = fetch_overall_analysis(symbol, from_date, to_date)

        print(f"Successfully retrieved {len(ohlcv_list)} price points and {len(scores_list)} score points for {symbol}.")
        print(f"Running training and testing for {symbol}...")
        metrics = tester.evaluate(ohlcv_list, scores_list, results_dir=results_dir)

        print("\n" + "="*50)
        print(f"VAR MODEL TEST REPORT (WITH SCORES) - {symbol}")
        print("="*50)
        print(f"Target Symbol:           {symbol}")
        print(f"Mean Squared Error:      {metrics['mse']:.8f}")
        print(f"Mean Absolute Error:     {metrics['mae']:.8f}")
        print("="*50)
        print("DAILY FORECASTED GROWTH RATE PREDICTIONS (FIRST 10 DAYS)")
        print("="*50)
        for day, g in enumerate(metrics['daily_forecasts'], 1):
            print(f"Day {day:02d}:                  {g*100:+.6f}%")
        print("="*50)
        print("5-YEAR GROWTH RATE PREDICTIONS (ANNUAL)")
        print("="*50)
        for year, g in enumerate(metrics['annual_growths'], 1):
            print(f"Year {year}:                 {g*100:+.0f}%")
        print("="*50)
        print("INTRINSIC VALUE CALCULATIONS")
        print("="*50)
        print(f"Dividend Discount Model: ${metrics['ddm_intrinsic_value']:.0f}")
        print(f"Discounted Cash Flow:    ${metrics['dcf_intrinsic_value']:.0f}")
        print("="*50)
        print(f"Forecast plot saved to:  {results_dir}/{symbol}_forecast.png")
        print("="*50 + "\n")

if __name__ == "__main__":
    main()
