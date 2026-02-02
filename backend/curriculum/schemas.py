"""
Schemas for Curriculum App in Beifahrer
by Alexander João Peterson Santos
TUM Lehrstuhl für Ergonomie
"""

from pydantic import BaseModel


# Helpers
class DrivingAssistanceSystemRating(BaseModel):
    """
    Rating scores for a driving assistant feature.

    Attributes
    ----------
    mean : float
        Average rating across practical and theoretical
    practical : int
        Rating for practical understanding (0-4 scale)
    theoretical : int
        Rating for theoretical understanding (0-4 scale)
    """

    mean: float
    practical: int
    theoretical: int


class SurveyAnswers(BaseModel):
    """
    Survey responses for all driving assistant features.

    Attributes
    ----------
    Abstandsregeltempomat : AssistantRatings
        Adaptive cruise control ratings
    Ampelerkennung : AssistantRatings
        Traffic light recognition ratings
    Notbremsassistent : AssistantRatings
        Emergency brake assistant ratings
    Spurführungsassistent : AssistantRatings
        Lane keeping assistant ratings
    Verkehrszeichenassistent : AssistantRatings
        Traffic sign assistant ratings
    """

    Abstandsregeltempomat: DrivingAssistanceSystemRating
    Ampelerkennung: DrivingAssistanceSystemRating
    Notbremsassistent: DrivingAssistanceSystemRating
    Spurführungsassistent: DrivingAssistanceSystemRating
    Verkehrszeichenassistent: DrivingAssistanceSystemRating


# Requests
class SurveyAnswersRequest(BaseModel):
    """
    Request to save user survey answers.

    Attributes
    ----------
    user_id : str
        Either Beifahrer UUID or onboarding_user_id
    answers : SurveyAnswers
        Survey responses for all assistant features
    """

    user_id: str
    answers: SurveyAnswers


# Responses
class ErrorResponse(BaseModel):
    """
    Error response.

    Attributes
    ----------
    error : str
        Error message
    """

    error: str
