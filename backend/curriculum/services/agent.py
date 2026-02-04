"""
Curriculum Generation Agent using OpenAI API
by Alexander João Peterson Santos
TUM Lehrstuhl für Ergonomie
2026-01-24
"""

import os
from typing import Any, cast

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

# Initialize OpenAI LLM
_llm = None


def get_llm() -> ChatOpenAI:
    """
    Lazy-load and return the OpenAI LLM instance.

    Returns
    -------
    ChatOpenAI
        Configured OpenAI LLM instance

    """
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            api_key=cast(SecretStr, os.getenv("OPENAI_API_KEY")),
        )
    return _llm


class FollowUpQuestions(BaseModel):
    """
    Schema for three follow-up questions.

    Attributes
    ----------
    question1 : str
        First follow-up question
    question2 : str
        Second follow-up question
    question3 : str
        Third follow-up question
    """

    question1: str = Field(description="First follow-up question")
    question2: str = Field(description="Second follow-up question")
    question3: str = Field(description="Third follow-up question")


class RecommendedChapters(BaseModel):
    """
    Schema for recommended curriculum chapters.

    Attributes
    ----------
    Abstandsregeltempomat : bool
        User must read adaptive cruise control chapter
    Ampelerkennung : bool
        User must read traffic light recognition chapter
    Notbremsassistent : bool
        User must read emergency brake assistant chapter
    Spurführungsassistent : bool
        User must read lane keeping assistant chapter
    Verkehrszeichenassistent : bool
        User must read traffic sign assistant chapter
    """

    Abstandsregeltempomat: bool = Field(
        description="True if user needs to read adaptive cruise control chapter"
    )
    Ampelerkennung: bool = Field(
        description="True if user needs to read traffic light recognition chapter"
    )
    Notbremsassistent: bool = Field(
        description="True if user needs to read emergency brake assistant chapter"
    )
    Spurführungsassistent: bool = Field(
        description="True if user needs to read lane keeping assistant chapter"
    )
    Verkehrszeichenassistent: bool = Field(
        description="True if user needs to read traffic sign assistant chapter"
    )


def generate_followup_questions(survey_answers: dict[str, Any]) -> dict[str, str]:
    """
    Generate three follow-up questions based on user's survey answers.

    Uses GPT to analyze the user's self-reported familiarity ratings and generate
    targeted questions to better understand their actual knowledge level.

    Parameters
    ----------
    survey_answers : dict[str, Any]
        Survey responses with ratings for each assistant feature.
        Format: {"Abstandsregeltempomat": {"mean": 3, "practical": 3, "theoretical": 3}, ...}

    Returns
    -------
    dict[str, str]
        Three follow-up questions as a dictionary with keys "question1", "question2", "question3"

    Examples
    --------
    >>> answers = {
    ...     "Abstandsregeltempomat": {"mean": 3, "practical": 3, "theoretical": 3},
    ...     "Ampelerkennung": {"mean": 1.5, "practical": 3, "theoretical": 0}
    ... }
    >>> questions = generate_followup_questions(answers)
    >>> print(questions["question1"])

    """
    system_prompt = """Du bist ein Experte für Fahrerassistenzsysteme und hilfst dabei, die Vertrautheit von Fahrern mit diesen Systemen einzuschätzen.

Basierend auf den Selbsteinschätzungen eines Fahrers zu verschiedenen Assistenzsystemen sollst du drei gezielte Folgefragen entwickeln.

Die Bewertungsskala ist:
- 0: Keine Erfahrung
- 1: Sehr wenig Erfahrung
- 2: Etwas Erfahrung
- 3: Gute Erfahrung
- 4: Sehr gute Erfahrung

Für jedes System gibt es:
- practical: Praktische Erfahrung (Nutzung)
- theoretical: Theoretisches Wissen
- mean: Durchschnitt beider Werte

Deine Aufgabe:
1. Identifiziere Systeme, bei denen es Diskrepanzen zwischen praktischer und theoretischer Erfahrung gibt
2. Konzentriere dich auf Systeme mit mittlerer Erfahrung (mean zwischen 1.5 und 3, oder practical/theoretical zwischen 1 und 3)
3. Formuliere drei offene Fragen, die helfen, die tatsächliche Vertrautheit besser einzuschätzen

KRITISCH – Diversifizierung der Fragen:
- Jede der drei Fragen MUSS sich auf ein ANDERES Fahrerassistenzsystem beziehen
- Verteile die Fragen über verschiedene Systeme, um ein breites Verständnis zu bekommen
- Priorisiere Systeme mit "mittleren" Bewertungen (2-4), da hier die größte Unsicherheit besteht
- Wenn mehrere Systeme in Frage kommen, wähle drei unterschiedliche aus
- VERMEIDE es, alle drei Fragen auf dasselbe System zu fokussieren

Prioritäten bei der System-Auswahl:
1. Systeme mit mean-Werten zwischen 1.5 und 2.5 (höchste Priorität – "eher wenig" bis "etwas")
2. Systeme mit Diskrepanzen zwischen practical und theoretical (Differenz > 1)
3. Systeme mit einzelnen Werten von 1 oder 2 (auch wenn mean höher ist)
4. Falls möglich, mindestens ein System mit niedriger practical-Erfahrung und eines mit niedriger theoretical-Erfahrung

WICHTIG – Ziel der Fragen:
Die Fragen sollen die VERTRAUTHEIT des Fahrers mit dem System einschätzen, NICHT sein technisches Wissen abfragen. Es geht nicht darum zu testen, ob der Fahrer weiß, wie ein System funktioniert oder was er in einer kritischen Situation tun sollte. Stattdessen sollen die Fragen herausfinden, wie viel alltägliche Erfahrung und Umgang der Fahrer mit dem System hat.

Gute Themen für Fragen:
- Wie lange und wie regelmäßig der Fahrer das System genutzt hat
- Ob das System im eigenen Auto vorhanden war oder nur in einem Mietwagen o.ä.
- Wie oft das System im Alltag aktiv war oder genutzt wurde
- Ob der Fahrer weiß, wie man das System ein- und ausschaltet
- Welche Funktionen oder Einstellungen das System im eigenen Fahrzeug hatte
- Ob der Fahrer das System bewusst verwendet oder eher passiv erlebt hat

VERMEIDE folgende Fragetypen:
- Fragen nach konkreten kritischen Situationen oder Beinahe-Unfällen
- Prüfungsfragen, die technisches Wissen testen (z.B. "Was passiert, wenn...")
- Fragen, die wie eine Wissensabfrage wirken

Die Fragen sollten:
- Offen formuliert sein (nicht ja/nein)
- Den alltäglichen Umgang und die Nutzungserfahrung ansprechen
- Einen gesprächigen, nicht prüfenden Ton haben
- Auf Deutsch sein
- Sich jeweils auf unterschiedliche Systeme beziehen
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Hier sind die Selbsteinschätzungen des Fahrers:\n\n{survey_answers}\n\nGeneriere drei gezielte Folgefragen.",
            ),
        ]
    )

    parser = JsonOutputParser(pydantic_object=FollowUpQuestions)
    llm = get_llm()

    # Add format instructions to the LLM
    llm_with_structure = llm.with_structured_output(FollowUpQuestions)

    chain = prompt | llm_with_structure

    try:
        result = chain.invoke({"survey_answers": str(survey_answers)})
        return result.model_dump()
    except Exception as e:
        print(f"Error generating follow-up questions: {e}")
        # Return default questions on error
        return {
            "question1": "Wie lange nutzen Sie den Abstandsregeltempomat schon und wie regelmäßig setzen Sie ihn ein?",
            "question2": "Ist der Spurführungsassistent in Ihrem eigenen Auto verbaut, und wenn ja, haben Sie ihn bewusst aktiviert oder läuft er automatisch?",
            "question3": "Haben Sie den Notbremsassistenten schon einmal in Ihrem Fahrzeug bemerkt – zum Beispiel durch eine Warnung oder ein Eingreifen?",
        }


def generate_recommended_chapters(
    survey_answers: dict[str, Any], followup_qa: list[dict[str, str]]
) -> dict[str, bool]:
    """
    Generate curriculum chapter recommendations based on survey and follow-up answers.

    Uses GPT to analyze both the initial survey responses and follow-up Q&A to determine
    which manual chapters the user should read before driving.

    Parameters
    ----------
    survey_answers : dict[str, Any]
        Survey responses with ratings for each assistant feature.
        Format: {"Abstandsregeltempomat": {"mean": 3, "practical": 3, "theoretical": 3}, ...}
    followup_qa : list[dict[str, str]]
        List of follow-up questions and answers.
        Format: [{"q": "question text", "a": "answer text"}, ...]

    Returns
    -------
    dict[str, bool]
        Recommended chapters where True means user must read that chapter.
        Format: {"Abstandsregeltempomat": true, "Ampelerkennung": false, ...}

    Examples
    --------
    >>> answers = {"Abstandsregeltempomat": {"mean": 1, "practical": 0, "theoretical": 2}}
    >>> qa = [{"q": "How does ACC work?", "a": "I'm not sure"}]
    >>> recommendations = generate_recommended_chapters(answers, qa)
    >>> print(recommendations["Abstandsregeltempomat"])
    True

    """
    system_prompt = """Du bist ein Experte für Fahrerassistenzsysteme und Fahrsicherheit.

Deine Aufgabe ist es, basierend auf den Selbsteinschätzungen eines Fahrers und seinen Antworten auf Folgefragen zu entscheiden, welche Kapitel des Fahrzeughandbuchs der Fahrer VOR der Fahrt lesen MUSS.

Die fünf Assistenzsysteme sind:
- Abstandsregeltempomat (Adaptive Cruise Control)
- Ampelerkennung (Traffic Light Recognition)
- Notbremsassistent (Emergency Brake Assistant)
- Spurführungsassistent (Lane Keeping Assistant)
- Verkehrszeichenassistent (Traffic Sign Assistant)

Bewertungskriterien:
- Ein Fahrer MUSS ein Kapitel lesen (true), wenn:
  * Die Selbsteinschätzung niedrig ist (mean < 2)
  * Große Diskrepanz zwischen praktischer und theoretischer Erfahrung besteht
  * Die Folgeantworten zeigen, dass wichtiges Wissen fehlt
  * Sicherheitsrelevante Missverständnisse erkennbar sind

- Ein Fahrer MUSS NICHT lesen (false), wenn:
  * Gute Selbsteinschätzung (mean >= 3) UND solide Folgeantworten
  * Klare Demonstration von Verständnis in den Antworten
  * Sowohl praktische als auch theoretische Kompetenz vorhanden

Sicherheit geht vor: Im Zweifelsfall sollte das Kapitel gelesen werden (true).
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                """Selbsteinschätzungen:
{survey_answers}

Folgefragen und Antworten:
{followup_qa}

Entscheide für jedes System, ob der Fahrer das entsprechende Kapitel lesen muss.""",
            ),
        ]
    )

    llm = get_llm()
    llm_with_structure = llm.with_structured_output(RecommendedChapters)

    chain = prompt | llm_with_structure

    try:
        result = chain.invoke(
            {"survey_answers": str(survey_answers), "followup_qa": str(followup_qa)}
        )
        return result.model_dump()
    except Exception as e:
        print(f"Error generating recommended chapters: {e}")
        # Conservative default: recommend all chapters on error
        return {
            "Abstandsregeltempomat": True,
            "Ampelerkennung": True,
            "Notbremsassistent": True,
            "Spurführungsassistent": True,
            "Verkehrszeichenassistent": True,
        }
