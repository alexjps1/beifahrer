"""
Schemas for Curriculum App in Beifahrer
by Alexander João Peterson Santos
TUM Lehrstuhl für Ergonomie
"""

from typing import Literal, Union

from pydantic import BaseModel, Field


# Helpers
class DrivingAssistanceSystemRating(BaseModel):
    """
    Rating scores for a driving assistant feature.

    Attributes
    ----------
    mean : float
        Average rating across practical and theoretical
    practical : int
        Rating for practical understanding (0-6 scale)
    theoretical : int
        Rating for theoretical understanding (0-6 scale)
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


# New Requests/Responses for Follow-up Questions and Recommendations
class UserIdRequest(BaseModel):
    """
    Request containing only a user ID.

    Attributes
    ----------
    user_id : str
        Either Beifahrer UUID or onboarding_user_id
    """

    user_id: str


class FollowUpQuestionsResponse(BaseModel):
    """
    Response containing follow-up questions (legacy format for open-ended).

    Attributes
    ----------
    question1 : str
        First follow-up question
    question2 : str
        Second follow-up question
    question3 : str
        Third follow-up question
    """

    question1: str
    question2: str
    question3: str


class OpenEndedQuestion(BaseModel):
    """
    Single open-ended follow-up question.

    Attributes
    ----------
    text : str
        The question text
    """

    text: str


class MultipleChoiceQuestion(BaseModel):
    """
    Single multiple choice follow-up question.

    Attributes
    ----------
    text : str
        The question text
    options : list[str]
        Answer options (2 for true/false, 3 for multiple choice)
    format : Literal["true_false", "multiple_choice"]
        Question format type
    favorable_answer : str
        The correct answer (must be one of the options)
    """

    text: str
    options: list[str] = Field(..., min_length=2, max_length=3)
    format: Literal["true_false", "multiple_choice"]
    favorable_answer: str


class FollowUpQuestionsResponseV2(BaseModel):
    """
    Response containing follow-up questions (v2 format supporting multiple types).

    Attributes
    ----------
    question_type : Literal["open_ended", "multiple_choice"]
        Type of questions being returned
    questions : list[Union[OpenEndedQuestion, MultipleChoiceQuestion]]
        List of questions
    """

    question_type: Literal["open_ended", "multiple_choice"]
    questions: list[Union[OpenEndedQuestion, MultipleChoiceQuestion]]


class QuestionAnswerPair(BaseModel):
    """
    Single question-answer pair.

    Attributes
    ----------
    q : str
        Question text
    a : str
        Answer text
    """

    q: str
    a: str


class FollowUpAnswersRequest(BaseModel):
    """
    Request to submit follow-up question answers.

    Attributes
    ----------
    user_id : str
        Either Beifahrer UUID or onboarding_user_id
    answers : list[QuestionAnswerPair]
        List of question-answer pairs
    """

    user_id: str
    answers: list[QuestionAnswerPair]


class RecommendedChaptersResponse(BaseModel):
    """
    Response containing recommended curriculum chapters.

    Attributes
    ----------
    Abstandsregeltempomat : bool
        User must read adaptive cruise control chapter
    Ampelerkennung : bool
        User must read traffic light recognition chapter
    Notbremsassistent : bool
        User must read emergency brake assistant chapter
    Spurführungsassistent : bool
        User must read lane keeping assistant chapter
    Verkehrszeichenassistent : bool
        User must read traffic sign assistant chapter
    """

    Abstandsregeltempomat: bool
    Ampelerkennung: bool
    Notbremsassistent: bool
    Spurführungsassistent: bool
    Verkehrszeichenassistent: bool
