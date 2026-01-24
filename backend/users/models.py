from django.db import models

# Create your models here.

class User(models.Model):
	# match "identification_code" and "name" fields in profiles table of onboarding app DB
	user_id = models.CharField(primary_key=True, editable=False)
	user_name = models.CharField(editable=False)

	# Curriculum attributes
	survey_answers = models.JSONField(default=dict)
	followup_answers= models.JSONField(default=dict)
	recommended_chapters = models.JSONField(default=dict)

	# Chat attributes
	chat_ids = models.JSONField(default=list)
	agent_scratchpad = models.JSONField(default=dict)

	class Meta:
		db_table = "users"

