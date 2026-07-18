import json
from typing import Dict, Any
from .exceptions import ResponseError

class FundamentalResponse:
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

    def get_entries(self) -> list:
        return self.data.get("data", [])

    def get_periods(self) -> list:
        periods = []
        for entry in self.get_entries():
            date = entry.get("balanceSheet", {}).get("date")
            if date:
                periods.append(date)
        return periods

    def get_scores(self, period: str) -> Dict[str, Any]:
        scores = {}
        for entry in self.get_entries():
            bs = entry.get("balanceSheet", {})
            if bs.get("date") == period:
                analysis = entry.get("analysis") or {}
                ai_summary = analysis.get("aiSummary") or {}
                for key in ("cashHealthScore", "leverageScore", "liquidityScore", "profitabilityScore"):
                    if key in ai_summary:
                        scores[key] = ai_summary[key]
                detail_scores = analysis.get("scores") or {}
                for sheet_name, sheet_data in detail_scores.items():
                    if not isinstance(sheet_data, dict):
                        continue
                    overall = sheet_data.get("overallScore")
                    if overall is not None:
                        scores[f"{sheet_name}_overallScore"] = overall
                    for section_name, section_data in sheet_data.items():
                        if not isinstance(section_data, dict):
                            continue
                        for indicator_name, indicator_data in section_data.items():
                            if isinstance(indicator_data, dict):
                                ratio = indicator_data.get("ratio")
                                score = indicator_data.get("score")
                                if ratio is not None:
                                    scores[f"{sheet_name}_{section_name}_{indicator_name}_ratio"] = ratio
                                if score is not None:
                                    scores[f"{sheet_name}_{section_name}_{indicator_name}_score"] = score
                break
        return scores
