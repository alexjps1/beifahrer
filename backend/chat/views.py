from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Chat
from .schemas import ChatResponse, HistoryResponse, Message


class ChatView(APIView):
    def get(self, request: Request, chat_id: str) -> Response:
        """
        Return the history of the chat associated with the chat_id
        """
        try:
            chat: Chat = Chat.objects.get(id=chat_id)
        except Chat.DoesNotExist:
            return Response({"error": f"No chat with chat_id {chat_id}"}, status=404)
        response: HistoryResponse = HistoryResponse(history=chat.history)
        return Response(response.model_dump(), status=200)

    def post(self, request: Request, chat_id: str) -> Response:
        """
        Send a message and get a response.
        """
        user_content: str = request.data.get("content", "")
        assistant_content: str = f"user sent: {user_content}. llm response here."

        chat, _created = Chat.objects.get_or_create(
            id=chat_id,
            defaults={"chat_name": f"Chat {chat_id}", "history": []},
        )

        chat.history.append({"role": "user", "content": user_content})
        chat.history.append({"role": "assistant", "content": assistant_content})
        chat.save()

        message: Message = Message(role="assistant", content=assistant_content)
        response: ChatResponse = ChatResponse(message=message)
        return Response(response.model_dump(), 200)
