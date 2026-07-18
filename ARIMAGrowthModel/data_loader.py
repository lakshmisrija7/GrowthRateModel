import asyncio
import datetime
import json
import pandas as pd
import websockets
from FundamentalAnalysis.client import FundamentalWebSocketClient
from TechnicalAnalysis.client import TechnicalWebSocketClient
from SectorAnalysis.SectorAnalysisFetcher import SectorAnalysisFetcher
from SentimentAnalysis.SentimentAnalysisFetcher import SentimentAnalysisFetcher
from OverallAnalysis.client import OverallAnalysisClient
from .config import WS_URL, API_KEY, DEFAULT_SYMBOL, FALLBACK_SYMBOL
from .logger import get_logger
from .exceptions import ResponseError

logger = get_logger(__name__)

class ARIMADataLoader:
    def __init__(self, symbol: str = DEFAULT_SYMBOL, api_key: str = API_KEY):
        self.symbol = symbol
        self.company_name = symbol
        self.api_key = api_key
        
    async def _fetch_ohlcv_data(self, symbol: str, from_date: str, to_date: str, real_time: bool = True) -> list:
        headers = {}
        if self.api_key:
            headers["X-API-KEY"] = self.api_key
            headers["X_API_KEY"] = self.api_key
        try:
            async with websockets.connect(WS_URL, additional_headers=headers, max_size=None, ping_interval=None) as ws:
                req = {
                    "type": "ohlcvData",
                    "action": "GET_OHLCV_DATA",
                    "filter": {
                        "symbol": symbol,
                        "realTime": real_time,
                        "dateRange": {
                            "fromDate": str(from_date),
                            "toDate": str(to_date)
                        }
                    }
                }
                await ws.send(json.dumps(req))
                resp = await ws.recv()
                data = json.loads(resp)
                return data.get("data", [])
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
            return []

    async def load_and_align_data(self, from_date: int, to_date: int) -> pd.DataFrame:
        logger.info(f"Attempting to load data for symbol {self.symbol} from {from_date} to {to_date}")
        
        ohlcv_current = await self._fetch_ohlcv_data(self.symbol, from_date, to_date)
        if not ohlcv_current:
            logger.warning(f"No OHLCV data returned for symbol {self.symbol}. Falling back to {FALLBACK_SYMBOL}!")
            self.symbol = FALLBACK_SYMBOL
            self.company_name = FALLBACK_SYMBOL
            ohlcv_current = await self._fetch_ohlcv_data(self.symbol, from_date, to_date)
            if not ohlcv_current:
                raise ResponseError("No OHLCV data found for either primary or fallback symbols.")
            
        ms_5y = 5 * 365 * 24 * 60 * 60 * 1000
        from_date_5y_fwd = from_date + ms_5y
        to_date_5y_fwd = to_date + ms_5y
        ohlcv_5y_forward = await self._fetch_ohlcv_data(self.symbol, from_date_5y_fwd, to_date_5y_fwd, real_time=False)
        
        daily_closes_curr = {}
        for item in ohlcv_current:
            dt = datetime.datetime.fromtimestamp(item["time"]/1000, datetime.timezone.utc).date()
            daily_closes_curr[dt] = item["close"]
            
        daily_closes_5y_fwd = {}
        for item in ohlcv_5y_forward:
            dt = datetime.datetime.fromtimestamp(item["time"]/1000, datetime.timezone.utc).date()
            daily_closes_5y_fwd[dt] = item["close"]
            
        logger.info(f"Loaded {len(daily_closes_curr)} current closes and {len(daily_closes_5y_fwd)} 5y-forward closes for {self.symbol}")

        wide_from = 1600000000000
        wide_to = 1900000000000

        tech_client = TechnicalWebSocketClient(WS_URL, api_key=self.api_key)
        await tech_client.connect()
        try:
            tech_resp = await tech_client.fetch_technical_analysis(self.symbol, wide_from, wide_to)
            tech_entries = tech_resp.get_entries()
        finally:
            await tech_client.disconnect()
            
        tech_list = []
        for entry in tech_entries:
            dt = datetime.datetime.fromtimestamp(entry["time"]/1000, datetime.timezone.utc).date()
            tech_list.append({
                "date": dt,
                "tech_trend": entry.get("trend"),
                "tech_momentum": entry.get("momentum"),
                "tech_volatility": entry.get("volatility"),
                "tech_rsi": entry.get("rsiScore"),
                "tech_adx": entry.get("adxScore"),
                "tech_bollinger": entry.get("bollingerWidth")
            })
        df_tech = pd.DataFrame(tech_list)
        if df_tech.empty:
            raise ResponseError("No Technical analysis data fetched")
            
        fund_client = FundamentalWebSocketClient(WS_URL, api_key=self.api_key)
        await fund_client.connect()
        try:
            fund_resp = await fund_client.fetch_fundamentals(self.symbol, wide_from, wide_to)
            fund_periods = fund_resp.get_periods()
            fund_list = []
            for period in fund_periods:
                scores = fund_resp.get_scores(period)
                dt = datetime.datetime.strptime(period, "%Y-%m-%d").date()
                fund_list.append({
                    "date": dt,
                    "fund_cashHealth": scores.get("cashHealthScore"),
                    "fund_leverage": scores.get("leverageScore"),
                    "fund_liquidity": scores.get("liquidityScore"),
                    "fund_profitability": scores.get("profitabilityScore")
                })
        finally:
            await fund_client.disconnect()
        df_fund = pd.DataFrame(fund_list)
        
        sec_headers = {"x-api-key": self.api_key} if self.api_key else {}
        sec_fetcher = SectorAnalysisFetcher(url=WS_URL, headers=sec_headers)
        sec_resp = await sec_fetcher.fetch_async(self.company_name)
        sec_list = []
        for item in sec_resp.data:
            dt = datetime.datetime.fromtimestamp(item.timestamp/1000, datetime.timezone.utc).date()
            ratio = item.ratio
            sec_list.append({
                "date": dt,
                "sec_pe": ratio.company_metrics.pe,
                "sec_roe": ratio.company_metrics.roe,
                "sec_npm": ratio.company_metrics.npm,
                "sec_sector_pe": ratio.sector_info.pe_ratio,
                "sec_sector_roe": ratio.sector_info.roe
            })
        df_sec = pd.DataFrame(sec_list)
        
        sent_headers = {"x-api-key": self.api_key} if self.api_key else {}
        sent_fetcher = SentimentAnalysisFetcher(url=WS_URL, headers=sent_headers)
        sent_resp = await sent_fetcher.fetch_async(self.symbol, str(wide_from), str(wide_to))
        sent_list = []
        for item in sent_resp.data:
            dt = datetime.datetime.fromtimestamp(item.timestamp/1000, datetime.timezone.utc).date()
            analysis = item.analysis
            sent_list.append({
                "date": dt,
                "sent_score": analysis.overall_sentiment.score,
                "sent_confidence": analysis.overall_sentiment.ai_confidence
            })
        df_sent = pd.DataFrame(sent_list)

        # --- Overall Analysis ---
        overall_client = OverallAnalysisClient(url=WS_URL, api_key=self.api_key)
        overall_list = []
        try:
            await overall_client.connect()
            overall_resp = await overall_client.fetch_overall_analysis(
                self.symbol,
                from_date=str(wide_from),
                to_date=str(wide_to)
            )
            for entry in overall_resp.get_entries():
                scores = overall_resp.get_scores(entry)
                ts = scores.get("timestamp")
                if ts is None:
                    continue
                dt = datetime.datetime.fromtimestamp(ts / 1000, datetime.timezone.utc).date()
                def _safe_float(v):
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        return None
                overall_list.append({
                    "date":                   dt,
                    "overall_overallScore":   _safe_float(scores.get("overallScore")),
                    "overall_technicalScore": _safe_float(scores.get("technicalScore")),
                    "overall_fundamentalScore": _safe_float(scores.get("fundamentalScore")),
                    "overall_sentimentScore": _safe_float(scores.get("sentimentScore")),
                    "overall_industryScore":  _safe_float(scores.get("industryScore")),
                    "overall_riskScore":      _safe_float(scores.get("riskScore")),
                })
        except Exception as e:
            logger.warning(f"Overall analysis fetch failed for {self.symbol}: {e}")
        finally:
            await overall_client.disconnect()
        df_overall = pd.DataFrame(overall_list)
        logger.info(f"Loaded {len(overall_list)} overall analysis entries for {self.symbol}")

        timeline = sorted(list(daily_closes_curr.keys()))
        df_align = pd.DataFrame({"date": timeline})
        
        df_align = df_align.merge(df_tech, on="date", how="left")
        
        if not df_fund.empty:
            df_align = df_align.merge(df_fund, on="date", how="left")
        else:
            for col in ["fund_cashHealth", "fund_leverage", "fund_liquidity", "fund_profitability"]:
                df_align[col] = None
                
        if not df_sec.empty:
            df_align = df_align.merge(df_sec, on="date", how="left")
        else:
            for col in ["sec_pe", "sec_roe", "sec_npm", "sec_sector_pe", "sec_sector_roe"]:
                df_align[col] = None
                
        if not df_sent.empty:
            df_align = df_align.merge(df_sent, on="date", how="left")
        else:
            for col in ["sent_score", "sent_confidence"]:
                df_align[col] = None

        overall_cols = [
            "overall_overallScore", "overall_technicalScore", "overall_fundamentalScore",
            "overall_sentimentScore", "overall_industryScore", "overall_riskScore"
        ]
        if not df_overall.empty:
            df_align = df_align.merge(df_overall, on="date", how="left")
        else:
            for col in overall_cols:
                df_align[col] = None
                
        df_align = df_align.ffill().bfill()
        
        growths = []
        aligned_close_curr = []
        aligned_close_5y_fwd = []
        
        dates_5y_fwd = sorted(list(daily_closes_5y_fwd.keys()))
        
        for idx, row in df_align.iterrows():
            d = row["date"]
            close_curr = daily_closes_curr[d]
            
            d_5y_fwd_target = d + datetime.timedelta(days=5*365)
            if not dates_5y_fwd:
                growths.append(None)
                aligned_close_curr.append(close_curr)
                aligned_close_5y_fwd.append(None)
                continue
                
            closest_d = min(dates_5y_fwd, key=lambda x: abs((x - d_5y_fwd_target).days))
            close_5y_fwd = daily_closes_5y_fwd[closest_d]
            
            growth = (close_5y_fwd - close_curr) / close_curr
            growths.append(growth)
            aligned_close_curr.append(close_curr)
            aligned_close_5y_fwd.append(close_5y_fwd)
            
        df_align["close_curr"] = aligned_close_curr
        df_align["close_5y_fwd"] = aligned_close_5y_fwd
        df_align["growth"] = growths
        
        df_align = df_align.dropna(subset=["growth"])
        return df_align
