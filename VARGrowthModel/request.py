from dataclasses import dataclass

@dataclass
class OHLCVRequest:
    symbol: str
    from_date: str
    to_date: str
    real_time: bool = False
    action: str = "GET_OHLCV_DATA"
    type: str = "ohlcvData"

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "action": self.action,
            "filter": {
                "symbol": self.symbol,
                "realTime": self.real_time,
                "dateRange": {
                    "fromDate": self.from_date,
                    "toDate": self.to_date
                }
            }
        }
