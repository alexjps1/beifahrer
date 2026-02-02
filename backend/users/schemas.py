"""
Schemas for User App in Beifahrer
by Alexander João Peterson Santos
TUM Lehrstuhl für Ergonomie
"""

from pydantic import BaseModel

# requests


class CreateUserRequest(BaseModel):
    onboarding_user_id: str | None = None  # optional, for users from onboarding app
    user_name: str


# responses


class CreateUserResponse(BaseModel):
    user_id: str


class ErrorResponse(BaseModel):
    error: str
