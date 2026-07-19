


import asyncio
import os
import json
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from OverallAnalysis.client import OverallAnalysisClient
from OverallAnalysis.logger import get_logger

logger = get_logger("overall_runner")

SYMBOLS = ["UBER", "NVDA", "APP", "LLY", "AVGO"]

FROM_DATE = "977219570000"
TO_DATE   = "1766137976000"

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "OverallAnalysis", "results")

SCORE_FIELDS = [
    ("overallScore",     "Overall",     "#4C72B0"),
    ("technicalScore",   "Technical",   "#55A868"),
    ("fundamentalScore", "Fundamental", "#C44E52"),
    ("sentimentScore",   "Sentiment",   "#8172B2"),
    ("sectorScore",    "sector",    "#CCB974"),
]


def save_score_bar_plot(symbol: str, summary: dict):

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_path = os.path.join(RESULTS_DIR, f"{symbol}_overall_analysis.png")

    labels, values, colors = [], [], []
    for field, label, color in SCORE_FIELDS:
        val = summary.get(field)
        if val is None:
            continue
        try:
            values.append(float(val))
            labels.append(label)
            colors.append(color)
        except (ValueError, TypeError):
            logger.warning(f"[{symbol}] Skipping non-numeric score for '{field}': {val!r}")

    if not values:
        logger.warning(f"[{symbol}] No score values to plot — skipping PNG")
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.12,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold", color="#222222"
        )

    action = summary.get("action", "N/A")
    ts = summary.get("timestamp")
    date_str = datetime.datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d") if ts else "N/A"

    ax.set_title(
        f"Overall Analysis — {symbol}  |  Action: {action}  |  As of: {date_str}",
        fontsize=13, fontweight="bold", pad=14
    )
    ax.set_ylabel("Score (0–10)", fontsize=12)
    ax.set_ylim(0, 10.5)
    ax.axhline(y=5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="Neutral (5.0)")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"[{symbol}] Bar chart saved to {plot_path}")


def save_timeseries_plot(symbol: str, all_scores: list):

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_path = os.path.join(RESULTS_DIR, f"{symbol}_overall_timeseries.png")

    dated = [(s["timestamp"], s["overallScore"]) for s in all_scores if s.get("timestamp") and s.get("overallScore") is not None]
    if not dated:
        logger.warning(f"[{symbol}] No time-series data to plot")
        return

    dates = [datetime.datetime.fromtimestamp(t / 1000) for t, _ in dated]
    scores = [v for _, v in dated]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, scores, color="#4C72B0", linewidth=1.5, label="Overall Score")
    ax.axhline(y=5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="Neutral (5.0)")
    ax.fill_between(dates, scores, 5, where=[s >= 5 for s in scores], alpha=0.15, color="#55A868", label="Above neutral")
    ax.fill_between(dates, scores, 5, where=[s < 5 for s in scores], alpha=0.15, color="#C44E52", label="Below neutral")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate(rotation=45)

    ax.set_title(f"Overall Score Over Time — {symbol}", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Overall Score (0–10)", fontsize=11)
    ax.set_ylim(0, 10.5)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"[{symbol}] Time-series chart saved to {plot_path}")


def save_raw_json(symbol: str, all_scores: list):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, f"{symbol}_overall_analysis.json")
    with open(json_path, "w") as f:
        json.dump(all_scores, f, indent=2)
    logger.info(f"[{symbol}] Raw data saved to {json_path}")


async def run_symbol(symbol: str):
    logger.info(f"\n{'='*60}")
    logger.info(f"Fetching overall analysis for: {symbol}")
    logger.info(f"{'='*60}")

    client = OverallAnalysisClient()
    try:
        await client.connect()
        response = await client.fetch_overall_analysis(symbol, FROM_DATE, TO_DATE)

        entries = response.get_entries()
        logger.info(f"[{symbol}] Received {len(entries)} entries")

        all_scores = response.get_all_scores()
        save_raw_json(symbol, all_scores)

        summary = response.get_summary()
        if summary:
            logger.info(f"[{symbol}] Latest: action={summary.get('action')}, overallScore={summary.get('overallScore')}")
            save_score_bar_plot(symbol, summary)
            save_timeseries_plot(symbol, all_scores)
        else:
            logger.warning(f"[{symbol}] No summary data available")

    except Exception as e:
        logger.error(f"[{symbol}] Failed: {e}", exc_info=True)
    finally:
        await client.disconnect()


async def main():
    for symbol in SYMBOLS:
        await run_symbol(symbol)
    logger.info("\nAll symbols processed. Results in OverallAnalysis/results/")


if __name__ == "__main__":
    asyncio.run(main())
