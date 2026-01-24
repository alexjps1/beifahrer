"""
Schemas for User App in Beifahrer
by Alexander João Peterson Santos
TUM Lehrstuhl für Ergonomie
"""

from pydantic import BaseModel

# requests

class CreateUserRequest(BaseModel):
	user_id: str  # not optional, expected that user already has identifier in onboarding app DB
	user_name: str

class DeleteUserRequest(BaseModel):
	user_id: str

# responses

class ErrorResponse(BaseModel):
	error: str