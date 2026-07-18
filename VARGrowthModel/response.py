from dataclasses import dataclass
from typing import List

@dataclass
class OHLCVItem:
    symbol: str
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_dict(cls, data: dict) -> 'OHLCVItem':
        return cls(
            symbol=data.get("symbol", ""),
            time=data.get("time", 0),
            open=float(data.get("open", 0.0)),
            high=float(data.get("high", 0.0)),
            low=float(data.get("low", 0.0)),
            close=float(data.get("close", 0.0)),
            volume=float(data.get("volume", 0.0))
        )

@dataclass
class OHLCVResponse:
    status: str
    type: str
    data: List[OHLCVItem]

    @classmethod
    def from_dict(cls, data: dict) -> 'OHLCVResponse':
        items = [OHLCVItem.from_dict(item) for item in data.get("data", [])]
        return cls(
            status=data.get("status", ""),
            type=data.get("type", ""),
            data=items
        )
