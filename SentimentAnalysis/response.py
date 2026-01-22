from dataclasses import dataclass
from typing import List

@dataclass
class SentimentOverall:
    score: int
    score_3m_comparison: float
    ai_confidence: float

    @classmethod
    def from_dict(cls, data: dict) -> 'SentimentOverall':
        return cls(
            score=data.get("score", 0),
            score_3m_comparison=data.get("score3mComparison", 0.0),
            ai_confidence=data.get("aiConfidence", 0.0)
        )

@dataclass
class SentimentCategoryDetail:
    score: int
    trend: float

    @classmethod
    def from_dict(cls, data: dict) -> 'SentimentCategoryDetail':
        return cls(
            score=data.get("score", 0),
            trend=data.get("trend", 0.0)
        )

@dataclass
class SentimentAnalysisCategories:
    overall_company_strategy: SentimentCategoryDetail
    general: SentimentCategoryDetail
    product_and_experience: SentimentCategoryDetail
    value: SentimentCategoryDetail
    financial_performance: SentimentCategoryDetail
    strategy_and_governance: SentimentCategoryDetail

    @classmethod
    def from_dict(cls, data: dict) -> 'SentimentAnalysisCategories':
        return cls(
            overall_company_strategy=SentimentCategoryDetail.from_dict(data.get("overallCompanyStrategy", {})),
            general=SentimentCategoryDetail.from_dict(data.get("general", {})),
            product_and_experience=SentimentCategoryDetail.from_dict(data.get("productAndExperience", {})),
            value=SentimentCategoryDetail.from_dict(data.get("value", {})),
            financial_performance=SentimentCategoryDetail.from_dict(data.get("financialPerformance", {})),
            strategy_and_governance=SentimentCategoryDetail.from_dict(data.get("strategyAndGovernance", {}))
        )

@dataclass
class SentimentAnalysisDetail:
    overall_sentiment: SentimentOverall
    categories: SentimentAnalysisCategories

    @classmethod
    def from_dict(cls, data: dict) -> 'SentimentAnalysisDetail':
        return cls(
            overall_sentiment=SentimentOverall.from_dict(data.get("overallSentiment", {})),
            categories=SentimentAnalysisCategories.from_dict(data.get("categories", {}))
        )

@dataclass
class SentimentAnalysisItem:
    symbol: str
    timestamp: int
    timestamp_long: int
    analysis: SentimentAnalysisDetail

    @classmethod
    def from_dict(cls, data: dict) -> 'SentimentAnalysisItem':
        return cls(
            symbol=data.get("symbol", ""),
            timestamp=data.get("timestamp", 0),
            timestamp_long=data.get("timestampLong", 0),
            analysis=SentimentAnalysisDetail.from_dict(data.get("analysis", {}))
        )

@dataclass
class SentimentAnalysisResponse:
    status: str
    type: str
    data: List[SentimentAnalysisItem]

    @classmethod
    def from_dict(cls, data: dict) -> 'SentimentAnalysisResponse':
        items = [SentimentAnalysisItem.from_dict(item) for item in data.get("data", [])]
        return cls(
            status=data.get("status", ""),
            type=data.get("type", ""),
            data=items
        )

@dataclass
class SentimentScoreDetail:
    negative: float
    neutral: float
    overall: float
    positive: float

    @classmethod
    def from_dict(cls, data: dict) -> 'SentimentScoreDetail':
        return cls(
            negative=data.get("negative", 0.0),
            neutral=data.get("neutral", 0.0),
            overall=data.get("overall", 0.0),
            positive=data.get("positive", 0.0)
        )

@dataclass
class SentimentTrendItem:
    id: str
    timestamp: int
    timestamp_long: int
    sentiment_score: SentimentScoreDetail
    category: str
    symbol: str
    article_id: str

    @classmethod
    def from_dict(cls, data: dict) -> 'SentimentTrendItem':
        return cls(
            id=data.get("id", ""),
            timestamp=data.get("timestamp", 0),
            timestamp_long=data.get("timestampLong", 0),
            sentiment_score=SentimentScoreDetail.from_dict(data.get("sentimentScore", {})),
            category=data.get("category", ""),
            symbol=data.get("symbol", ""),
            article_id=data.get("articleId", "")
        )

@dataclass
class SentimentTrendResponse:
    status: str
    type: str
    data: List[SentimentTrendItem]

    @classmethod
    def from_dict(cls, data: dict) -> 'SentimentTrendResponse':
        items = [SentimentTrendItem.from_dict(item) for item in data.get("data", [])]
        return cls(
            status=data.get("status", ""),
            type=data.get("type", ""),
            data=items
        )
