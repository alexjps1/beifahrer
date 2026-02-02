"""
API Views for Beifahrer Curriculum Generation
by Alexander João Peterson Santos
TUM Lehrstuhl für Ergonomie
2026-01-24
"""

from drf_spectacular.utils import extend_schema
from pydantic import ValidationError
from rest_framework.decorators import api_view
from rest_framework.request import Request
from rest_framework.response import Response
from users.models import User

from .schemas import ErrorResponse, SurveyAnswersRequest


@extend_schema(
    request=SurveyAnswersRequest,
    responses={200: None, 400: ErrorResponse, 404: ErrorResponse},
)
@api_view(["POST"])
def post_survey_answers(request: Request) -> Response:
    """
    Save user survey answers for driving assistant features.

    Accepts survey responses with ratings for practical and theoretical understanding
    of various driving assistant features. The user can be identified by either their
    Beifahrer UUID or onboarding_user_id.

    Parameters
    ----------
    request : Request
        HTTP request containing SurveyAnswersRequest data with fields:
        - user_id : str
            Either Beifahrer UUID or onboarding_user_id
        - answers : SurveyAnswers
            Dictionary with ratings for each assistant feature

    Returns
    -------
    Response
        HTTP status code:
        - 200: Survey answers saved successfully
        - 400: Invalid request data
        - 404: User not found

    """
    # validate input
    try:
        data = SurveyAnswersRequest(**request.data)
    except ValidationError as e:
        return Response(e.errors(), status=400)

    # Try to find user by beifahrer user_id first
    user = User.objects.filter(user_id=data.user_id).first()

    # If not found, try by onboarding_user_id
    if not user:
        user = User.objects.filter(onboarding_user_id=data.user_id).first()

    if not user:
        return Response({"error": "User not found"}, status=404)

    # Update and save survey answers
    user.survey_answers = data.answers.model_dump()
    user.save()

    return Response(status=200)
