from dataclasses import dataclass

@dataclass
class SentimentAnalysisRequest:
    symbol: str
    from_date: str
    to_date: str
    action: str = "GET_SENTIMENT_ANALYSIS"
    type: str = "sentimentAnalysis"

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "action": self.action,
            "filter": {
                "symbol": self.symbol,
                "dateRange": {
                    "fromDate": self.from_date,
                    "toDate": self.to_date
                }
            }
        }

@dataclass
class SentimentTrendRequest:
    category: str
    from_date: str
    to_date: str
    action: str = "GET_SENTIMENT_TREND"
    type: str = "sentimentTrend"

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "action": self.action,
            "filter": {
                "category": self.category,
                "dateRange": {
                    "fromDate": self.from_date,
                    "toDate": self.to_date
                }
            }
        }
