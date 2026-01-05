import chat.views as views
from django.contrib import admin
from django.urls import path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/chat/<str:chat_id>", views.chat_view, name="chat"),
    path("api/chat/", views.post_new_chat, name="chat-new"),
]
