from django.db import models


class User(models.Model):
    """
    User model for the Beifahrer application.

    Stores user profile information, curriculum progress, and chat history.
    Users can be identified by either their Beifahrer UUID (user_id) or their
    onboarding app identifier (onboarding_user_id).

    Attributes
    ----------
    user_id : CharField
        Unique UUID identifier for the user in the Beifahrer system.
        Primary key, auto-generated, non-editable.
    onboarding_user_id : CharField
        Identifier from the onboarding application (optional).
        Matches the "identification_code" field in onboarding app profiles.
        Non-editable, can be blank.
    user_name : CharField
        Name of the user. Non-editable after creation.
    survey_answers : JSONField
        Dictionary storing user responses to curriculum surveys.
        Default: empty dict
    followup_answers : JSONField
        Dictionary storing user responses to followup questions.
        Default: empty dict
    recommended_chapters : JSONField
        Dictionary of recommended curriculum chapters for the user.
        Default: empty dict
    chat_ids : JSONField
        List of chat session IDs associated with this user.
        Default: empty list
    agent_scratchpad : JSONField
        Dictionary used by the AI agent to store temporary working memory.
        Default: empty dict

    """

    user_id = models.CharField(primary_key=True, editable=False)
    onboarding_user_id = models.CharField(editable=False, blank=True)
    user_name = models.CharField(editable=False)

    survey_answers = models.JSONField(default=dict)
    followup_questions = models.JSONField(default=dict)
    followup_answers = models.JSONField(default=dict)
    recommended_chapters = models.JSONField(default=dict)

    chat_ids = models.JSONField(default=list)
    agent_scratchpad = models.JSONField(default=dict)

    class Meta:
        db_table = "users"
