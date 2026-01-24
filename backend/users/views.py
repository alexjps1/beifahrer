"""
API Views for Beifahrer Users
by Alexander João Peterson Santos
TUM Lehrstuhl für Ergonomie
2026-01-24
"""

from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.request import Request
from rest_framework.response import Response
from pydantic import ValidationError

# for swagger to understand schemas
from drf_spectacular.utils import extend_schema

# my models and schemas
from .models import User
from .schemas import CreateUserRequest, DeleteUserRequest, ErrorResponse


@extend_schema(request=CreateUserRequest, responses={201: None, 400: ErrorResponse})
@api_view(["POST"])
def post_create_user(request: Request) -> Response:
    """
    Create a new User, to which survey answers, etc. can be saved.
    """
    # validate input
    try:
        data = CreateUserRequest(**request.data)
    except ValidationError as e:
        # 400 Bad Request
        return Response(e.errors(), status=400)

    if User.objects.filter(user_id=data.user_id).exists():
        return Response({"error": "User with this id already exists"}, status=409)

    User.objects.create(
        user_id=data.user_id,
        user_name=data.user_name,
        survey_answers={},
        followup_answers=[],
        recommended_chapters=[],
        chat_ids=[],
        agent_scratchpad={},
    )
    return Response(status=201)


@extend_schema(responses={204: None, 404: ErrorResponse})
@api_view(["DELETE"])
def delete_user(_request: Request, user_id: str) -> Response:
    """
    Delete a User.
    This function later needs to be updated to delete associated chats.
    """
    user_to_delete = User.objects.filter(user_id=user_id)

    # TODO delete all chats (once chat functionality is implemented)

    if user_to_delete.exists():
        user_to_delete.delete()
        return Response(status=204)
    else:
        return Response({"error": "User not found"}, status=404)
