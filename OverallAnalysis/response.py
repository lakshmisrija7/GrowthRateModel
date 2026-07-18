import json
from typing import Dict, Any, List, Optional
from .exceptions import ResponseError


class OverallAnalysisResponse:
    """
    Parses and provides access to the GET_OVERALL_ANALYSIS WebSocket response.

    Actual server entry shape:
    {
        "id": "<str>",
        "symbol": "<str>",
        "timestamp": <ms>,
        "timestampLong": <ms>,
        "ratio": {
            "overallScore":      <float>,
            "technicalScore":    <float>,
            "fundamentalScore":  <float>,
            "sentimentScore":    <float>,
            "industryScore":     <float>,
            "riskScore":         <float>,
            "action":            <str>    -- e.g. "Hold", "Buy", "Sell"
        },
        "insights": <null | str>
    }
    """

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
        """Return the raw list of overall analysis data entries."""
        return self.data.get("data", [])

    def get_scores(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract flattened score fields from a single raw entry.
        Scores are nested under entry['ratio']; guards against null values.
        """
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
            "insights":         entry.get("insights"),
        }

    def get_all_scores(self) -> List[Dict[str, Any]]:
        """Return a list of flattened score dicts for all entries, sorted by timestamp."""
        scores = [self.get_scores(entry) for entry in self.get_entries()]
        return sorted(scores, key=lambda s: s.get("timestamp") or 0)

    def get_summary(self) -> Optional[Dict[str, Any]]:
        """
        Return the most recent entry's scores, or None if no data.
        Useful for a quick snapshot of the latest overall analysis.
        """
        entries = self.get_entries()
        if not entries:
            return None
        latest = max(
            entries,
            key=lambda e: e.get("timestamp") or e.get("timestampLong") or 0
        )
        return self.get_scores(latest)
