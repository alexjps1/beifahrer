from typing import Literal

from pydantic import BaseModel


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    user_message_content: str


class NewChatResponse(BaseModel):
    chat_id: str


class ChatResponse(BaseModel):
    message: Message


class HistoryResponse(BaseModel):
    history: list[Message]
