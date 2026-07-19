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
class OverallAnalysisFilter:
    symbol: str
    date_range: DateRange

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "dateRange": self.date_range.to_dict()
        }


class OverallAnalysisRequest:



    def __init__(self, symbol: str, from_date: str, to_date: str):
        self.type = "overallAnalysis"
        self.action = "GET_OVERALL_ANALYSIS"
        self.filter = OverallAnalysisFilter(
            symbol=symbol,
            date_range=DateRange(from_date=str(from_date), to_date=str(to_date))
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "action": self.action,
            "filter": self.filter.to_dict()
        }
