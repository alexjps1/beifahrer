from django.db import models


class Chat(models.Model):
    id = models.CharField(primary_key=True, max_length=6, editable=False)
    chat_name = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    history = models.JSONField()

    class Meta:
        db_table = "chats"

    def __str__(self):
        return self.chat_name
