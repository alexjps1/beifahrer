"""
API Views for Beifahrer Users
by Alexander João Peterson Santos
TUM Lehrstuhl für Ergonomie
2026-01-24
"""

import uuid

from django.shortcuts import render

# for swagger to understand schemas
from drf_spectacular.utils import extend_schema
from pydantic import ValidationError
from rest_framework.decorators import api_view
from rest_framework.request import Request
from rest_framework.response import Response

# my models and schemas
from .models import User
from .schemas import (
    CreateUserRequest,
    CreateUserResponse,
    ErrorResponse,
)


@extend_schema(
    request=CreateUserRequest,
    responses={200: CreateUserResponse, 201: CreateUserResponse, 400: ErrorResponse},
)
@api_view(["POST"])
def post_upsert_user(request: Request) -> Response:
    """
    Create or update a User (upsert).

    Creates a new user or returns the ID of an existing user. If a user with the
    given onboarding_user_id already exists, returns that user's ID and updates
    the username if it has changed. Otherwise, creates a new user with a generated UUID.

    Parameters
    ----------
    request : Request
        HTTP request containing CreateUserRequest data with fields:
        - onboarding_user_id : str, optional
            Identifier from the onboarding app
        - user_name : str
            Name of the user

    Returns
    -------
    Response
        JSON response with user_id and HTTP status:
        - 200: User already exists (upserted with name update if needed)
        - 201: New user created
        - 400: Invalid request data

    """
    # validate input
    try:
        data = CreateUserRequest(**request.data)
    except ValidationError as e:
        # 400 Bad Request
        return Response(e.errors(), status=400)

    # Try to find existing user by onboarding_user_id if provided
    if data.onboarding_user_id:
        existing_user = User.objects.filter(
            onboarding_user_id=data.onboarding_user_id
        ).first()
        if existing_user:
            # User already exists, update name if changed and return existing user_id
            if existing_user.user_name != data.user_name:
                existing_user.user_name = data.user_name
                existing_user.save()
            return Response({"user_id": existing_user.user_id}, status=200)

    # Generate a UUID for the new user
    user_id = str(uuid.uuid4())

    User.objects.create(
        user_id=user_id,
        onboarding_user_id=data.onboarding_user_id or "",
        user_name=data.user_name,
        survey_answers={},
        followup_questions={},
        followup_answers={},
        recommended_chapters={},
        chat_ids=[],
        agent_scratchpad={},
    )
    return Response({"user_id": user_id}, status=201)


@extend_schema(responses={204: None, 404: ErrorResponse})
@api_view(["DELETE"])
def delete_user(_request: Request, user_id: str) -> Response:
    """
    Delete a User by either their Beifahrer user_id or onboarding_user_id.

    Deletes a user and their associated data. Accepts either the Beifahrer UUID
    or the onboarding app user ID for flexibility across different client systems.

    Parameters
    ----------
    _request : Request
        HTTP request (unused)
    any_user_id : str
        Either the Beifahrer user_id (UUID format) or onboarding_user_id

    Returns
    -------
    Response
        HTTP status code:
        - 204: User successfully deleted
        - 404: User not found

    Notes
    -----
    Associated chats are not currently deleted (TODO).

    """
    # Try to find user by beifahrer user_id first (UUID format)
    user_to_delete = User.objects.filter(user_id=user_id)

    # If not found, try to find by onboarding_user_id
    if not user_to_delete.exists():
        user_to_delete = User.objects.filter(onboarding_user_id=user_id)

    # TODO delete all chats (once chat functionality is implemented)

    if user_to_delete.exists():
        user_to_delete.delete()
        return Response(status=204)
    else:
        return Response({"error": "User not found"}, status=404)
