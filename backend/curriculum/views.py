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

from .schemas import (
    ErrorResponse,
    FollowUpAnswersRequest,
    FollowUpQuestionsResponse,
    FollowUpQuestionsResponseV2,
    RecommendedChaptersResponse,
    SurveyAnswersRequest,
    UserIdRequest,
)
from .services.agent import (
    generate_followup_questions,
    generate_recommended_chapters,
)


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


@extend_schema(
    request=UserIdRequest,
    responses={200: FollowUpQuestionsResponseV2, 400: ErrorResponse, 404: ErrorResponse},
)
@api_view(["POST"])
def post_generate_followup_questions(request: Request) -> Response:
    """
    Generate follow-up questions based on user's survey answers.

    Uses GPT to analyze the user's survey responses and generate three targeted
    follow-up questions. The questions are saved to the database and returned.

    Parameters
    ----------
    request : Request
        HTTP request containing UserIdRequest data with fields:
        - user_id : str
            Either Beifahrer UUID or onboarding_user_id

    Returns
    -------
    Response
        HTTP status code:
        - 200: Follow-up questions generated successfully (returns FollowUpQuestionsResponse)
        - 400: Invalid request data or missing survey answers
        - 404: User not found

    """
    # validate input
    try:
        data = UserIdRequest(**request.data)
    except ValidationError as e:
        return Response(e.errors(), status=400)

    # Try to find user by beifahrer user_id first
    user = User.objects.filter(user_id=data.user_id).first()

    # If not found, try by onboarding_user_id
    if not user:
        user = User.objects.filter(onboarding_user_id=data.user_id).first()

    if not user:
        return Response({"error": "User not found"}, status=404)

    # Check if survey answers exist
    if not user.survey_answers:
        return Response(
            {"error": "Survey answers not found. Please submit survey first."},
            status=400,
        )

    # Generate follow-up questions using the agent (returns v2 format)
    try:
        questions_response = generate_followup_questions(user.survey_answers)
    except Exception as e:
        print(f"Error generating follow-up questions: {e}")
        return Response({"error": "Fragen konnten nicht generiert werden."}, status=500)

    # Save questions to database (save the full response structure)
    user.followup_questions = questions_response
    user.save()

    return Response(questions_response, status=200)


@extend_schema(
    request=FollowUpAnswersRequest,
    responses={200: None, 400: ErrorResponse, 404: ErrorResponse},
)
@api_view(["POST"])
def post_followup_answers(request: Request) -> Response:
    """
    Submit answers to follow-up questions.

    Validates that the submitted answers match the questions stored in the database
    and saves the answers.

    Parameters
    ----------
    request : Request
        HTTP request containing FollowUpAnswersRequest data with fields:
        - user_id : str
            Either Beifahrer UUID or onboarding_user_id
        - answers : list[QuestionAnswerPair]
            List of question-answer pairs with format [{"q": "...", "a": "..."}, ...]

    Returns
    -------
    Response
        HTTP status code:
        - 200: Answers saved successfully
        - 400: Invalid request data, missing questions, or questions don't match
        - 404: User not found

    """
    # validate input
    try:
        data = FollowUpAnswersRequest(**request.data)
    except ValidationError as e:
        return Response(e.errors(), status=400)

    # Try to find user by beifahrer user_id first
    user = User.objects.filter(user_id=data.user_id).first()

    # If not found, try by onboarding_user_id
    if not user:
        user = User.objects.filter(onboarding_user_id=data.user_id).first()

    if not user:
        return Response({"error": "User not found"}, status=404)

    # Check if follow-up questions exist
    if not user.followup_questions:
        return Response(
            {
                "error": "Follow-up questions not found. Please generate questions first."
            },
            status=400,
        )

    # Extract the stored questions (handle both v1 and v2 formats)
    stored_questions = user.followup_questions

    # Check if it's the new v2 format with question_type
    if isinstance(stored_questions, dict) and "question_type" in stored_questions:
        # V2 format
        expected_questions = [q.get("text") for q in stored_questions.get("questions", [])]
        expected_count = len(expected_questions)
    else:
        # Legacy v1 format
        expected_questions = [
            stored_questions.get("question1"),
            stored_questions.get("question2"),
            stored_questions.get("question3"),
        ]
        expected_count = 3

    # Extract submitted questions
    submitted_questions = [qa.q for qa in data.answers]

    # Validate that submitted questions match stored questions
    if len(submitted_questions) != expected_count:
        return Response(
            {"error": f"Expected exactly {expected_count} question-answer pairs"}, status=400
        )

    for submitted_q in submitted_questions:
        if submitted_q not in expected_questions:
            return Response(
                {
                    "error": f"Question mismatch. Submitted question does not match stored questions: {submitted_q}"
                },
                status=400,
            )

    # Save answers to database
    user.followup_answers = [qa.model_dump() for qa in data.answers]
    user.save()

    return Response(status=200)


@extend_schema(
    request=UserIdRequest,
    responses={
        200: RecommendedChaptersResponse,
        400: ErrorResponse,
        404: ErrorResponse,
    },
)
@api_view(["POST"])
def get_recommended_chapters(request: Request) -> Response:
    """
    Get recommended curriculum chapters based on survey and follow-up answers.

    Uses GPT to analyze both the survey responses and follow-up Q&A to determine
    which manual chapters the user should read before driving.

    Parameters
    ----------
    request : Request
        HTTP request containing UserIdRequest data with fields:
        - user_id : str
            Either Beifahrer UUID or onboarding_user_id

    Returns
    -------
    Response
        HTTP status code:
        - 200: Recommendations generated successfully (returns RecommendedChaptersResponse)
        - 400: Invalid request data, missing survey answers, or missing follow-up data
        - 404: User not found

    """
    # validate input
    try:
        data = UserIdRequest(**request.data)
    except ValidationError as e:
        return Response(e.errors(), status=400)

    # Try to find user by beifahrer user_id first
    user = User.objects.filter(user_id=data.user_id).first()

    # If not found, try by onboarding_user_id
    if not user:
        user = User.objects.filter(onboarding_user_id=data.user_id).first()

    if not user:
        return Response({"error": "User not found"}, status=404)

    # Check if survey answers exist
    if not user.survey_answers:
        return Response(
            {"error": "Survey answers not found. Please submit survey first."},
            status=400,
        )

    # Check if follow-up questions exist
    if not user.followup_questions:
        return Response(
            {
                "error": "Follow-up questions not found. Please generate questions first."
            },
            status=400,
        )

    # Check if follow-up answers exist
    if not user.followup_answers:
        return Response(
            {"error": "Follow-up answers not found. Please submit answers first."},
            status=400,
        )

    # Build a lookup from question text to favorable_answer (MC questions only)
    favorable_answers: dict = {}
    stored_questions = user.followup_questions
    if (
        isinstance(stored_questions, dict)
        and stored_questions.get("question_type") == "multiple_choice"
    ):
        for q in stored_questions.get("questions", []):
            if "favorable_answer" in q:
                favorable_answers[q["text"]] = q["favorable_answer"]

    # Enrich each submitted answer with the favorable_answer where available
    enriched_answers = []
    for qa in user.followup_answers:
        enriched = dict(qa)
        if qa.get("q") in favorable_answers:
            enriched["favorable_answer"] = favorable_answers[qa["q"]]
        enriched_answers.append(enriched)

    # Generate recommendations using the agent
    recommendations = generate_recommended_chapters(
        user.survey_answers, enriched_answers
    )

    # Save recommendations to database
    user.recommended_chapters = recommendations
    user.save()

    return Response(recommendations, status=200)
