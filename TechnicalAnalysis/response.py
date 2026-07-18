import json
from typing import Dict, Any, List
from .exceptions import ResponseError
from .config import (
    RSIConfig,
    ADXConfig,
    BollingerConfig,
    MACDConfig,
    StochasticConfig,
    ScorerConfig
)

class TechnicalAnalysisResponse:
    def __init__(self, raw_data: str):
        self.raw_data = raw_data
        self.data = self._parse(raw_data)

    def _parse(self, raw_data: str) -> Dict[str, Any]:
        try:
            return json.loads(raw_data)
        except json.JSONDecodeError as e:
            raise ResponseError(f"Failed to parse response JSON: {e}") from e

    @property
    def payload(self) -> Dict[str, Any]:
        return self.data

    def get_entries(self) -> List[Dict[str, Any]]:
        return self.data.get("data", [])

    def _get_enum_name(self, config_class: Any, value: float) -> str:
        if value is None:
            return "UNKNOWN"
        try:
            val_int = int(value)
        except (ValueError, TypeError):
            return str(value)
        for name, val in config_class.__dict__.items():
            if val == val_int and not name.startswith('_'):
                return name
        return f"UNKNOWN({value})"

    def get_mapped_entries(self) -> List[Dict[str, Any]]:
        mapped = []
        for entry in self.get_entries():
            item = {
                "symbol": entry.get("symbol"),
                "time": entry.get("time"),
                "trend": self._get_enum_name(ScorerConfig.Trend, entry.get("trend")),
                "momentum": self._get_enum_name(ScorerConfig.Momentum, entry.get("momentum")),
                "volatility": self._get_enum_name(ScorerConfig.Volatility, entry.get("volatility")),
                "trendStatus": self._get_enum_name(MACDConfig.Momentum, entry.get("trendStatus")),
                "crossOver": entry.get("crossOver"),
                "distanceToSma": entry.get("distanceToSma"),
                "distanceToEma": entry.get("distanceToEma"),
                "histogramValue": entry.get("histogramValue"),
                "divergence": self._get_enum_name(MACDConfig.Divergence, entry.get("divergence")),
                "adxScore": entry.get("adxScore"),
                "trendState": self._get_enum_name(ADXConfig.States, entry.get("trendState")),
                "alert": self._get_enum_name(ADXConfig.Alerts, entry.get("alert")),
                "rsiScore": entry.get("rsiScore"),
                "zoneStatus": self._get_enum_name(RSIConfig.Zones, entry.get("zoneStatus")),
                "reversalSignal": self._get_enum_name(RSIConfig.Signals, entry.get("reversalSignal")),
                "bollingerWidth": entry.get("bollingerWidth"),
                "breakoutStatus": self._get_enum_name(BollingerConfig.Breakouts, entry.get("breakoutStatus")),
                "atrVolatility": entry.get("atrVolatility"),
                "stopLoss": entry.get("stopLoss"),
                
            }
            if "risk" in entry:
                item["risk"] = self._get_enum_name(ScorerConfig.Risk, entry.get("risk"))
            mapped.append(item)
        return mapped
