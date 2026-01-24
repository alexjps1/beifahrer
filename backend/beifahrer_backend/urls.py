import chat.views as chat_views
import users.views as users_views

from django.contrib import admin
from django.urls import path
from drf_spectacular.views import (
    SpectacularAPIView,
    SpectacularRedocView,
    SpectacularSwaggerView,
)

urlpatterns = [
    path("admin/", admin.site.urls),
    path(
        "api/schema/swagger/",
        SpectacularSwaggerView.as_view(url_name="schema"),
        name="swagger",
    ),
    path(
        "api/schema/redoc/",
        SpectacularRedocView.as_view(url_name="schema"),
        name="redoc",
    ),
    path("api/schema/", SpectacularAPIView.as_view(), name="schema"),
    path("api/chat/<str:chat_id>/", chat_views.chat_view, name="chat"),
    path("api/chat/", chat_views.post_new_chat, name="chat-new"),
    path("api/users/", users_views.post_create_user, name="create-user"),
    path("api/users/<str:user_id>", users_views.delete_user, name="delete-user"),
]
