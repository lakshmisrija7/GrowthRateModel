from dataclasses import dataclass
from typing import List, Any, Optional

@dataclass
class AnalystReport:
    competitive_edge: List[str]
    sector_landscape: List[str]
    company_vs_sector: List[str]

    @classmethod
    def from_dict(cls, data: dict) -> 'AnalystReport':
        return cls(
            competitive_edge=data.get("competitiveEdge", []),
            sector_landscape=data.get("sectorLandscape", []),
            company_vs_sector=data.get("companyVsSector", [])
        )

@dataclass
class SectorShareholding:
    fii: float
    promoters: float
    retail: float
    domestic_institutions: float

    @classmethod
    def from_dict(cls, data: dict) -> 'SectorShareholding':
        return cls(
            fii=data.get("fii", 0.0),
            promoters=data.get("promoters", 0.0),
            retail=data.get("retail", 0.0),
            domestic_institutions=data.get("domesticInstitutions", 0.0)
        )

@dataclass
class ComparativeAnalysis:
    date: str
    company_roe: float
    sector_roe: float

    @classmethod
    def from_dict(cls, data: dict) -> 'ComparativeAnalysis':
        return cls(
            date=data.get("date", ""),
            company_roe=data.get("companyROE", 0.0),
            sector_roe=data.get("sectorROE", 0.0)
        )

@dataclass
class CompanyMetrics:
    pb: Optional[float]
    pe: float
    price: float
    roe: float
    npm: float

    @classmethod
    def from_dict(cls, data: dict) -> 'CompanyMetrics':
        return cls(
            pb=data.get("pb"),
            pe=data.get("pe", 0.0),
            price=data.get("price", 0.0),
            roe=data.get("roe", 0.0),
            npm=data.get("npm", 0.0)
        )

@dataclass
class SectorSentimentScore:
    score: float
    sector: str

    @classmethod
    def from_dict(cls, data: dict) -> 'SectorSentimentScore':
        return cls(
            score=data.get("score", 0.0),
            sector=data.get("sector", "")
        )

@dataclass
class SectorComparison:
    sector_sentiment_scores: List[SectorSentimentScore]

    @classmethod
    def from_dict(cls, data: dict) -> 'SectorComparison':
        scores = [SectorSentimentScore.from_dict(item) for item in data.get("sectorSentimentScores", [])]
        return cls(sector_sentiment_scores=scores)

@dataclass
class SectorInfo:
    net_profit_margin: float
    pe_ratio: float
    dte_ratio: float
    roe: float
    sector_name: str

    @classmethod
    def from_dict(cls, data: dict) -> 'SectorInfo':
        return cls(
            net_profit_margin=data.get("netProfitMargin", 0.0),
            pe_ratio=data.get("peRatio", 0.0),
            dte_ratio=data.get("dteRatio", 0.0),
            roe=data.get("roe", 0.0),
            sector_name=data.get("sectorName", "")
        )

@dataclass
class PeerComparison:
    symbol: str
    market_cap: float
    pe_ratio: float
    roe: float

    @classmethod
    def from_dict(cls, data: dict) -> 'PeerComparison':
        return cls(
            symbol=data.get("symbol", ""),
            market_cap=data.get("marketCap", 0.0),
            pe_ratio=data.get("peRatio", 0.0),
            roe=data.get("roe", 0.0)
        )

@dataclass
class SectorRatio:
    period: str
    analyst_report: AnalystReport
    sector_shareholding: SectorShareholding
    comparative_analysis: List[ComparativeAnalysis]
    company_metrics: CompanyMetrics
    sector_comparison: SectorComparison
    sector_info: SectorInfo
    peer_comparison: List[PeerComparison]

    @classmethod
    def from_dict(cls, data: dict) -> 'SectorRatio':
        return cls(
            period=data.get("period", ""),
            analyst_report=AnalystReport.from_dict(data.get("analystReport", {})),
            sector_shareholding=SectorShareholding.from_dict(data.get("sectorShareholding", {})),
            comparative_analysis=[ComparativeAnalysis.from_dict(item) for item in data.get("comparativeAnalysis", [])],
            company_metrics=CompanyMetrics.from_dict(data.get("companyMetrics", {})),
            sector_comparison=SectorComparison.from_dict(data.get("sectorComparison", {})),
            sector_info=SectorInfo.from_dict(data.get("sectorInfo", {})),
            peer_comparison=[PeerComparison.from_dict(item) for item in data.get("peerComparison", [])]
        )

@dataclass
class SectorAnalysisItem:
    id: str
    company_name: str
    timestamp: int
    timestamp_long: int
    ratio: SectorRatio

    @classmethod
    def from_dict(cls, data: dict) -> 'SectorAnalysisItem':
        return cls(
            id=data.get("id", ""),
            company_name=data.get("companyName", ""),
            timestamp=data.get("timestamp", 0),
            timestamp_long=data.get("timestampLong", 0),
            ratio=SectorRatio.from_dict(data.get("ratio", {}))
        )

@dataclass
class SectorAnalysisResponse:
    status: str
    type: str
    data: List[SectorAnalysisItem]

    @classmethod
    def from_dict(cls, data: dict) -> 'SectorAnalysisResponse':
        items = [SectorAnalysisItem.from_dict(item) for item in data.get("data", [])]
        return cls(
            status=data.get("status", ""),
            type=data.get("type", ""),
            data=items
        )
