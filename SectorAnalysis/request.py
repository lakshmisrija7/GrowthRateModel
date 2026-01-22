from dataclasses import dataclass

@dataclass
class SectorAnalysisRequest:
    company_name: str
    action: str = "GET_SECTOR_ANALYSIS"
    type: str = "sectorAnalysis"

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "action": self.action,
            "filter": {
                "companyName": self.company_name
            }
        }
