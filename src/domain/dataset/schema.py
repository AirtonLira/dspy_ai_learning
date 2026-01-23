from dataclasses import dataclass


@dataclass(frozen=True)
class B2WRawschema:
    """
    Schema ORIGINAL do CSV B2W Reviews.
    Nunca usar diretamente fora do loader.
    """
    REVIEW_TEXT: str = "review_text"
    OVERALL_RATING: str = "overall_rating"
    
@dataclass(frozen=True)
class DSPyReviewSchema:
    """
    Schema NORMALIZADO do domínio.
    Todo o projeto DSPy usa apenas isso.
    """
    TEXT: str = "text"
    SENTIMENT: str = "sentiment"