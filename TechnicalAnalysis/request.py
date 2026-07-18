from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DateRange:
    from_date: int
    to_date: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "fromDate": self.from_date,
            "toDate": self.to_date
        }

@dataclass
class TechnicalFilter:
    symbol: str
    date_range: DateRange

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "dateRange": self.date_range.to_dict()
        }

class TechnicalRequest:
    def __init__(self, symbol: str, from_date: int, to_date: int):
        self.type = "technicalAnalysisData"
        self.action = "GET_TECHNICAL_ANALYSIS"
        self.filter = TechnicalFilter(symbol, DateRange(from_date, to_date))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "action": self.action,
            "filter": self.filter.to_dict()
        }
