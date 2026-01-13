import random

from drf_spectacular.utils import extend_schema
from rest_framework.decorators import api_view
from rest_framework.request import Request
from rest_framework.response import Response

from .models import Chat
from .schemas import (
    ChatRequest,
    ChatResponse,
    HistoryResponse,
    Message,
    NewChatResponse,
)
from .services.agent import generate_response


@extend_schema(request=ChatRequest, responses=NewChatResponse)
@api_view(["POST"])
def post_new_chat(request: Request) -> Response:
    """
    Create a new chat with the first message and return chat_id + assistant response.
    """
    user_content: str = request.data.get("user_message_content", "")
    if not user_content:
        return Response({"error": "Message content is required"}, status=400)

    # generate a unique chat_id with 6 digits
    generated_chat_id = str(random.randint(100000, 999999))
    unique: bool = False
    for i in range(5):
        if not Chat.objects.filter(id=generated_chat_id).exists():
            unique = True
            break
        generated_chat_id = str(random.randint(100000, 999999))
    if not unique:
        return Response({"error": "Could not generate a unique chat_id"}, status=500)

    # Generate AI response using RAG
    assistant_content: str = generate_response(user_content, chat_history=[])

    Chat.objects.create(
        id=generated_chat_id,
        chat_name=f"Chat {generated_chat_id}",
        history=[
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    )

    message: Message = Message(role="assistant", content=assistant_content)
    response: NewChatResponse = NewChatResponse(
        chat_id=generated_chat_id, message=message
    )
    return Response(response.model_dump(), status=200)


def get_chat_history(_request: Request, chat_id: str) -> Response:
    """
    Return the history of the chat associated with the chat_id.
    Assumes chat_id is valid and associated with a chat.
    """
    chat: Chat = Chat.objects.get(id=chat_id)
    response: HistoryResponse = HistoryResponse(history=chat.history)
    return Response(response.model_dump(), status=200)


def post_message(request: Request, chat_id: str) -> Response:
    """
    Send a message to an existing chat and get a response.
    Assumes chat_id is valid and associated with a chat.
    """
    user_content: str = request.data.get("user_message_content", "")

    chat, _created = Chat.objects.get_or_create(
        id=chat_id,
        defaults={"chat_name": f"Chat {chat_id}", "history": []},
    )

    # Generate AI response using RAG with conversation history
    assistant_content: str = generate_response(user_content, chat_history=chat.history)

    chat.history.append({"role": "user", "content": user_content})
    chat.history.append({"role": "assistant", "content": assistant_content})
    chat.save()

    message: Message = Message(role="assistant", content=assistant_content)
    response: ChatResponse = ChatResponse(message=message)
    return Response(response.model_dump(), 200)


@extend_schema(methods=["GET"], request=None, responses={200: HistoryResponse})
@extend_schema(methods=["POST"], request=ChatRequest, responses={200: ChatResponse})
@api_view(["GET", "POST"])
def chat_view(request: Request, chat_id: str) -> Response:
    """
    GET: Return the history of the chat associated with the chat_id.
    POST: Send a message to an existing chat and get a response.
    """
    if not Chat.objects.filter(id=chat_id).exists():
        return Response({"error": f"No chat with chat_id {chat_id}"}, status=404)

    if request.method == "GET":
        return get_chat_history(request, chat_id)
    elif request.method == "POST":
        return post_message(request, chat_id)
    else:
        return Response({"error": "Method not allowed"}, status=405)
