from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DateRange:
    from_date: str
    to_date: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "fromDate": self.from_date,
            "toDate": self.to_date
        }

@dataclass
class FundamentalFilter:
    symbol: str
    date_range: DateRange

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "dateRange": self.date_range.to_dict()
        }

class FundamentalRequest:
    def __init__(self, symbol: str, from_date: str, to_date: str):
        self.type = "fundamentals"
        self.action = "GET_FUNDAMENTALS"
        self.filter = FundamentalFilter(symbol, DateRange(from_date, to_date))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "action": self.action,
            "filter": self.filter.to_dict()
        }
