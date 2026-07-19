import json
from typing import Dict, Any, List, Optional
from .exceptions import ResponseError


class OverallAnalysisResponse:



    def __init__(self, raw_data: str):
        self.raw_data = raw_data
        self.data = self._parse(raw_data)

    def _parse(self, raw_data: str) -> Dict[str, Any]:
        try:
            return json.loads(raw_data)
        except json.JSONDecodeError as e:
            raise ResponseError(f"Failed to parse overall analysis response JSON: {e}") from e

    @property
    def payload(self) -> Dict[str, Any]:
        return self.data

    def get_entries(self) -> List[Dict[str, Any]]:

        return self.data.get("data", [])

    def get_scores(self, entry: Dict[str, Any]) -> Dict[str, Any]:


        ratio = entry.get("ratio") or {}
        return {
            "timestamp":        entry.get("timestamp") or entry.get("timestampLong"),
            "symbol":           entry.get("symbol"),
            "overallScore":     ratio.get("overallScore"),
            "technicalScore":   ratio.get("technicalScore"),
            "fundamentalScore": ratio.get("fundamentalScore"),
            "sentimentScore":   ratio.get("sentimentScore"),
            "industryScore":    ratio.get("industryScore"),
            "riskScore":        ratio.get("riskScore"),
            "action":           ratio.get("action"),
        }

    def get_all_scores(self) -> List[Dict[str, Any]]:

        scores = [self.get_scores(entry) for entry in self.get_entries()]
        return sorted(scores, key=lambda s: s.get("timestamp") or 0)

    def get_summary(self) -> Optional[Dict[str, Any]]:


        entries = self.get_entries()
        if not entries:
            return None
        latest = max(
            entries,
            key=lambda e: e.get("timestamp") or e.get("timestampLong") or 0
        )
        return self.get_scores(latest)
