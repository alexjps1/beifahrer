"""
Curriculum Generation Agent using OpenAI API
by Alexander João Peterson Santos
TUM Lehrstuhl für Ergonomie
2026-01-24
"""

import os
from typing import Any, Literal, cast

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

# Configuration: Question generation mode
FOLLOWUP_QUESTION_MODE = "multiple_choice"  # Options: "open_ended", "multiple_choice"

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


def _get_relevant_systems(survey_answers: dict[str, Any]) -> dict[str, Any]:
    """
    Filter survey answers to only systems where the user has partial familiarity.

    A system is relevant if at least one of practical or theoretical is between
    2 and 5 (inclusive). Systems rated 0–1 on both dimensions are excluded
    (no real experience to probe), and systems rated 6 on both are excluded
    (already fully familiar).

    Parameters
    ----------
    survey_answers : dict[str, Any]
        Full survey responses.

    Returns
    -------
    dict[str, Any]
        Filtered subset of survey_answers. Falls back to the full dict if the
        filter would leave nothing.

    """
    relevant = {
        system: data
        for system, data in survey_answers.items()
        if 2 <= data.get("practical", 0) <= 5 or 2 <= data.get("theoretical", 0) <= 5
    }
    return relevant if relevant else survey_answers


def generate_followup_questions_open_ended(survey_answers: dict[str, Any]) -> dict[str, str]:
    """
    Generate three open-ended follow-up questions based on user's survey answers.

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
    >>> questions = generate_followup_questions_open_ended(answers)
    >>> print(questions["question1"])

    """
    system_prompt = """Du bist ein Experte für Fahrerassistenzsysteme und hilfst dabei, die Vertrautheit von Fahrern mit diesen Systemen einzuschätzen.

Basierend auf den Selbsteinschätzungen eines Fahrers zu verschiedenen Assistenzsystemen sollst du drei gezielte Folgefragen entwickeln.

Die Bewertungsskala ist (0-6):
- 0: keins (keine Erfahrung)
- 1: sehr wenig
- 2: wenig
- 3: eher wenig
- 4: eher viel
- 5: viel
- 6: sehr viel

WICHTIG: Werte 0-3 bedeuten geringe bis mäßige Vertrautheit. Erst ab 4 beginnt gute Vertrautheit.

Für jedes System gibt es:
- practical: Praktische Erfahrung (Nutzung)
- theoretical: Theoretisches Wissen
- mean: Durchschnitt beider Werte

Deine Aufgabe:
1. Identifiziere Systeme, bei denen es Diskrepanzen zwischen praktischer und theoretischer Erfahrung gibt
2. Konzentriere dich auf Systeme mit geringer bis mittlerer Erfahrung (mean zwischen 1 und 4, oder practical/theoretical zwischen 1 und 4)
3. Formuliere drei offene Fragen, die helfen, die tatsächliche Vertrautheit besser einzuschätzen

KRITISCH – Erlaubte Systeme:
- Stelle Fragen AUSSCHLIESSLICH zu den Systemen, die in der Nutzernachricht unter "Stelle Fragen NUR zu den folgenden Systemen" aufgelistet sind
- Systeme, die NICHT in dieser Liste stehen, sind verboten – auch wenn sie in den Selbsteinschätzungen auftauchen
- Es ist ausdrücklich erlaubt (und manchmal notwendig), mehrere Fragen zum selben System zu stellen, wenn die Liste nur wenige relevante Systeme enthält
- Versuche NICHT, die Fragen auf alle 5 Systeme zu verteilen – halte dich strikt an die erlaubte Liste

Prioritäten bei der System-Auswahl (nur innerhalb der erlaubten Systeme):
1. Systeme mit mean-Werten zwischen 2 und 4 (höchste Priorität – "wenig" bis "eher viel")
2. Systeme mit Diskrepanzen zwischen practical und theoretical (Differenz >= 2)
3. Falls möglich, mindestens ein System mit niedriger practical-Erfahrung und eines mit niedriger theoretical-Erfahrung

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
- Sich jeweils auf die unterschiedlichen Systeme beziehen
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Hier sind die Selbsteinschätzungen des Fahrers:\n\n{survey_answers}\n\n"
                "Stelle Fragen NUR zu den folgenden Systemen (alle anderen ignorieren):\n{relevant_system_names}\n\n"
                "Generiere drei gezielte Folgefragen.",
            ),
        ]
    )

    llm = get_llm()

    # Add format instructions to the LLM
    llm_with_structure = llm.with_structured_output(FollowUpQuestions)

    chain = prompt | llm_with_structure

    relevant_systems = _get_relevant_systems(survey_answers)
    relevant_system_names = "\n".join(
        f"- {name} (practical={data.get('practical', 0)}, theoretical={data.get('theoretical', 0)})"
        for name, data in relevant_systems.items()
    )

    result = cast(FollowUpQuestions, chain.invoke(
        {
            "survey_answers": str(relevant_systems),
            "relevant_system_names": relevant_system_names,
        }
    ))
    return result.model_dump()


class MCQuestionItem(BaseModel):
    """Single multiple choice follow-up question with a correct answer."""

    text: str
    options: list[str] = Field(min_length=2, max_length=3)
    format: Literal["true_false", "multiple_choice"]
    favorable_answer: str


class MultipleChoiceQuestions(BaseModel):
    """
    Schema for six multiple choice follow-up questions.

    Attributes
    ----------
    questions : list[MCQuestionItem]
        List of 6 questions, each with text, options, format, and favorable_answer
    """

    questions: list[MCQuestionItem] = Field(min_length=6, max_length=6)


def generate_followup_questions_mc(survey_answers: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Generate six multiple choice follow-up questions based on user's survey answers.

    Uses GPT to analyze the user's self-reported familiarity ratings and generate
    targeted multiple choice questions (true/false or 3-option MC) to assess knowledge.

    Parameters
    ----------
    survey_answers : dict[str, Any]
        Survey responses with ratings for each assistant feature.
        Format: {"Abstandsregeltempomat": {"mean": 3, "practical": 3, "theoretical": 3}, ...}

    Returns
    -------
    list[dict[str, Any]]
        Six multiple choice questions, each with format:
        {"text": "...", "options": ["...", "..."], "format": "true_false" or "multiple_choice"}

    Examples
    --------
    >>> answers = {"Abstandsregeltempomat": {"mean": 3, "practical": 3, "theoretical": 3}}
    >>> questions = generate_followup_questions_mc(answers)
    >>> print(questions[0]["text"])

    """
    system_prompt = """Du bist ein Experte für Fahrerassistenzsysteme und hilfst dabei, die Vertrautheit von Fahrern mit diesen Systemen einzuschätzen.

Basierend auf den Selbsteinschätzungen eines Fahrers zu verschiedenen Assistenzsystemen sollst du sechs Multiple-Choice-Fragen entwickeln.

Die Bewertungsskala ist (0-6):
- 0: keins (keine Erfahrung)
- 1: sehr wenig
- 2: wenig
- 3: eher wenig
- 4: eher viel
- 5: viel
- 6: sehr viel

WICHTIG: Werte 0-3 bedeuten geringe bis mäßige Vertrautheit. Erst ab 4 beginnt gute Vertrautheit.

Für jedes System gibt es:
- practical: Praktische Erfahrung (Nutzung)
- theoretical: Theoretisches Wissen
- mean: Durchschnitt beider Werte

Deine Aufgabe:
1. Identifiziere Systeme mit geringer bis mittlerer Erfahrung (mean zwischen 1 und 4, oder practical/theoretical zwischen 1 und 4)
2. Formuliere 6 Multiple-Choice-Fragen, die die tatsächliche Vertrautheit einschätzen

Fragetypen:
- True/False (Richtig/Falsch): 2 Antwortoptionen ["Stimmt", "Stimmt nicht"]
- Multiple Choice: Genau 3 Antwortoptionen

AUSSCHLUSS – Systeme mit hoher Vertrautheit:
- Stelle KEINE Fragen zu Systemen, bei denen practical >= 5 UND theoretical >= 5
- Diese Systeme zeigen ausreichende Vertrautheit und müssen nicht weiter überprüft werden
- Dies gilt auch dann, wenn dadurch weniger als 6 relevante Systeme übrig bleiben

Beispiel-Ausschluss:
  Notbremsassistent: practical=6, theoretical=6 → AUSGESCHLOSSEN, keine Frage stellen
  Abstandsregeltempomat: practical=6, theoretical=5 → AUSGESCHLOSSEN, keine Frage stellen
  Spurführungsassistent: practical=5, theoretical=4 → NICHT ausgeschlossen (theoretical < 5), Frage stellen
  Ampelerkennung: practical=2, theoretical=3 → NICHT ausgeschlossen, Frage stellen
  Verkehrszeichenassistent: practical=4, theoretical=6 → NICHT ausgeschlossen (practical < 5), Frage stellen

KRITISCH – Erlaubte Systeme:
- Stelle Fragen AUSSCHLIESSLICH zu den Systemen, die in der Nutzernachricht unter "Stelle Fragen NUR zu den folgenden Systemen" aufgelistet sind
- Systeme, die NICHT in dieser Liste stehen, sind verboten – auch wenn sie in den Selbsteinschätzungen auftauchen
- Es ist ausdrücklich erlaubt (und manchmal notwendig), mehrere Fragen zum selben System zu stellen, wenn die Liste nur wenige relevante Systeme enthält
- Versuche NICHT, die Fragen auf alle 5 Systeme zu verteilen – halte dich strikt an die erlaubte Liste

Diversifizierung (nur innerhalb der erlaubten Systeme):
- Priorisiere Systeme mit "mittleren" Bewertungen (2-4)
- Mische True/False und Multiple-Choice-Fragen

Fragetypen-Mix:
Stelle einen ausgewogenen Mix aus zwei Arten von Fragen:

1. Wissens-/Verständnisfragen (ca. 50%):
   - Ob der Fahrer weiß, wie das System aktiviert/deaktiviert wird
   - Welche Situationen das System erkennt oder wie es den Fahrer unterstützt
   - Typische Funktionen oder Einstellungen des Systems
   - favorable_answer = die sachlich korrekte Antwort

2. Erfahrungs-/Selbsteinschätzungsfragen (ca. 50%):
   - Wie lange oder wie regelmäßig der Fahrer das System genutzt hat
   - In welchem Fahrzeug das System genutzt wurde (z.B. eigenes Auto vs. Mietwagen)
   - Ob der Fahrer das System aktiv eingesetzt oder nur passiv erlebt hat
   - Ob der Fahrer sich sicher und komfortabel im Umgang mit dem System fühlt
   - favorable_answer = die Antwort, die auf die meiste Erfahrung und Vertrautheit hindeutet
     (z.B. "Ja, regelmäßig im eigenen Auto" ist günstiger als "Nur einmal in einem Mietwagen")

Die Fragen sollten:
- Klar und verständlich formuliert sein
- Nicht zu technisch sein
- Auf Deutsch sein
- Einen Mix aus True/False und Multiple-Choice darstellen

Format jeder Frage:
{{
    "text": "Fragetext hier",
    "options": ["Option 1", "Option 2"] oder ["Option 1", "Option 2", "Option 3"],
    "format": "true_false" oder "multiple_choice",
    "favorable_answer": "Die bevorzugte Antwort – bei Wissensfragen die sachlich korrekte, bei Erfahrungsfragen die Antwort mit der meisten Vertrautheit (muss exakt einer der Optionen entsprechen)"
}}
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Hier sind die Selbsteinschätzungen des Fahrers:\n\n{survey_answers}\n\n"
                "Stelle Fragen NUR zu den folgenden Systemen (alle anderen ignorieren):\n{relevant_system_names}\n\n"
                "Generiere sechs Multiple-Choice-Fragen.",
            ),
        ]
    )

    llm = get_llm()
    llm_with_structure = llm.with_structured_output(MultipleChoiceQuestions)

    chain = prompt | llm_with_structure

    relevant_systems = _get_relevant_systems(survey_answers)
    relevant_system_names = "\n".join(
        f"- {name} (practical={data.get('practical', 0)}, theoretical={data.get('theoretical', 0)})"
        for name, data in relevant_systems.items()
    )

    result = cast(MultipleChoiceQuestions, chain.invoke(
        {
            "survey_answers": str(relevant_systems),
            "relevant_system_names": relevant_system_names,
        }
    ))
    return [q.model_dump() for q in result.questions]


def generate_followup_questions(survey_answers: dict[str, Any]) -> dict[str, Any]:
    """
    Router function to generate follow-up questions based on configured mode.

    Calls either open-ended or multiple choice question generation based on
    FOLLOWUP_QUESTION_MODE constant.

    Parameters
    ----------
    survey_answers : dict[str, Any]
        Survey responses with ratings for each assistant feature.
        Format: {"Abstandsregeltempomat": {"mean": 3, "practical": 3, "theoretical": 3}, ...}

    Returns
    -------
    dict[str, Any]
        Response format depends on mode:
        - open_ended: {"question_type": "open_ended", "questions": [{"text": "..."}, ...]}
        - multiple_choice: {"question_type": "multiple_choice", "questions": [{...}, ...]}

    """
    if FOLLOWUP_QUESTION_MODE == "open_ended":
        questions_dict = generate_followup_questions_open_ended(survey_answers)
        return {
            "question_type": "open_ended",
            "questions": [
                {"text": questions_dict["question1"]},
                {"text": questions_dict["question2"]},
                {"text": questions_dict["question3"]},
            ],
        }
    elif FOLLOWUP_QUESTION_MODE == "multiple_choice":
        questions_list = generate_followup_questions_mc(survey_answers)
        return {"question_type": "multiple_choice", "questions": questions_list}
    else:
        raise ValueError(
            f"Invalid FOLLOWUP_QUESTION_MODE: {FOLLOWUP_QUESTION_MODE}. "
            'Must be "open_ended" or "multiple_choice".'
        )


def generate_recommended_chapters(
    survey_answers: dict[str, Any], followup_qa: list[dict[str, str]]
) -> dict[str, bool]:
    """
    Generate curriculum chapter recommendations based on survey and follow-up answers.

    Uses GPT to analyze both the initial survey responses and follow-up Q&A to determine
    which manual chapters the user should read before driving. Evaluates each system
    individually to avoid bias toward systems with follow-up questions.

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

    # Define a simple Pydantic model for single system recommendation
    class SingleSystemRecommendation(BaseModel):
        """Schema for a single system's chapter recommendation."""

        must_read: bool = Field(
            description="True if user must read this chapter before driving"
        )
        reasoning: str = Field(description="Brief explanation for the decision")

    system_prompt = """Du bist ein Experte für Fahrerassistenzsysteme und Fahrsicherheit.

Deine Aufgabe ist es, basierend auf den Selbsteinschätzungen eines Fahrers zu einem SPEZIFISCHEN System und allen verfügbaren Folgefragen zu entscheiden, ob der Fahrer das entsprechende Kapitel des Fahrzeughandbuchs VOR der Fahrt lesen MUSS.

Bewertungsskala (0-6):
- 0: keins (keine Erfahrung)
- 1: sehr wenig
- 2: wenig
- 3: eher wenig
- 4: eher viel
- 5: viel
- 6: sehr viel

WICHTIG: Werte 0-3 bedeuten geringe bis mäßige Vertrautheit und erfordern in der Regel das Lesen des Kapitels. Erst ab 5 kann von guter Vertrautheit ausgegangen werden.

WICHTIG – Unabhängige Bewertung:
Du bewertest NUR das aktuelle System. Die Bewertungen für andere Systeme sind irrelevant.

Bewertungskriterien:
Ein Fahrer MUSS ein Kapitel lesen (must_read: true), wenn eine oder mehrere der folgenden Bedingungen zutreffen:
  * Die Selbsteinschätzung ist niedrig bis mittel (mean <= 4)
  * practical-Wert ist 0, 1, 2, 3 oder 4 (geringe bis mittlere praktische Erfahrung)
  * theoretical-Wert ist 0, 1, 2, 3 oder 4 (geringes bis mittleres theoretisches Wissen)
  * Große Diskrepanz zwischen practical und theoretical besteht (Differenz >= 2)
  * Falls Folgefragen zu diesem System gestellt wurden: Die Antworten zeigen Unsicherheit oder fehlendes Verständnis
  * Bei Werten von 4 ("eher viel"): Nur überspringen wenn zusätzlich Folgeantworten klare Vertrautheit zeigen

Ein Fahrer MUSS NICHT lesen (must_read: false), wenn ALLE folgenden Bedingungen erfüllt sind:
  * Gute Selbsteinschätzung (mean >= 5)
  * Sowohl practical als auch theoretical >= 5
  * Falls Folgefragen gestellt wurden: Antworten zeigen klare, langjährige Vertrautheit und souveränen Umgang

KRITISCH – Umgang mit Folgefragen:
- Du erhältst NUR die Folgefragen, die sich SPEZIFISCH auf das aktuelle System beziehen
- Wenn "Keine Folgefragen zu [System] wurden gestellt" angezeigt wird, entscheide AUSSCHLIESSLICH anhand der Survey-Werte
- Das Fehlen von Folgefragen bedeutet NICHT, dass das System unwichtig ist
- Niedrige bis mittlere Survey-Werte (practical oder theoretical <= 4) sind allein ausreichend für must_read: true
- Verwende NIEMALS Informationen aus Folgefragen zu anderen Systemen
- Wenn Folgefragen vorhanden sind, bewerte sie ehrlich: Zeigen sie echte, langjährige Vertrautheit oder Unsicherheit?
- Falls "Bevorzugte Antwort" angegeben ist: Vergleiche sie mit der Antwort des Fahrers.
  * Bei Wissensfragen: Eine falsche Antwort ist ein klares Zeichen für Lernbedarf (must_read: true).
  * Bei Erfahrungsfragen: Eine ungünstige Antwort (z.B. "Nur in Mietwagen" statt "Im eigenen Auto") deutet auf geringe Vertrautheit hin und spricht für must_read: true.

Sicherheit geht vor: Im Zweifelsfall sollte das Kapitel gelesen werden (must_read: true).
Bei der Skala 0-6 sind nur Werte von 5-6 wirklich gut. Werte bis 4 bedeuten noch Lernbedarf.
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                """System: {system_name}

Selbsteinschätzung für dieses System:
- Praktische Erfahrung: {practical}
- Theoretisches Wissen: {theoretical}
- Durchschnitt: {mean}

Alle Folgefragen und Antworten (falls welche zu diesem System existieren):
{followup_qa}

Entscheide, ob der Fahrer das Kapitel zu diesem System lesen muss.""",
            ),
        ]
    )

    llm = get_llm()
    llm_with_structure = llm.with_structured_output(SingleSystemRecommendation)
    chain = prompt | llm_with_structure

    # List of all systems to evaluate
    systems = [
        "Abstandsregeltempomat",
        "Ampelerkennung",
        "Notbremsassistent",
        "Spurführungsassistent",
        "Verkehrszeichenassistent",
    ]

    recommendations = {}

    # Evaluate each system individually
    for system in systems:
        try:
            # Get ratings for this specific system
            system_data = survey_answers.get(
                system, {"mean": 0, "practical": 0, "theoretical": 0}
            )

            # Filter follow-up questions that are relevant to this system
            # Check if the system name appears in the question text
            relevant_qa = [
                qa for qa in followup_qa if system.lower() in qa.get("q", "").lower()
            ]

            # Format follow-up info
            if relevant_qa:
                followup_info = f"Folgefragen zu {system}:\n"
                for qa in relevant_qa:
                    followup_info += f"Frage: {qa.get('q', 'N/A')}\n"
                    if "favorable_answer" in qa:
                        followup_info += f"Bevorzugte Antwort: {qa.get('favorable_answer')}\n"
                    followup_info += f"Antwort des Fahrers: {qa.get('a', 'N/A')}\n\n"
            else:
                followup_info = f"Keine Folgefragen zu {system} wurden gestellt."

            # Invoke the chain for this system
            result = cast(SingleSystemRecommendation, chain.invoke(
                {
                    "system_name": system,
                    "practical": system_data.get("practical", 0),
                    "theoretical": system_data.get("theoretical", 0),
                    "mean": system_data.get("mean", 0),
                    "followup_qa": followup_info,
                }
            ))

            recommendations[system] = result.must_read

            # Print the decision and reasoning
            decision = "MUST READ" if result.must_read else "SKIP"
            print(f"\n[{system}] {decision}")
            print(
                f"Survey: practical={system_data.get('practical', 0)}, theoretical={system_data.get('theoretical', 0)}, mean={system_data.get('mean', 0)}"
            )

            # Print relevant Q&A if any
            if relevant_qa:
                for idx, qa in enumerate(relevant_qa, 1):
                    print(f"  Q{idx}: {qa.get('q', 'N/A')}")
                    print(f"  A{idx}: {qa.get('a', 'N/A')}")
            else:
                print(f"  (Keine Folgefragen zu diesem System)")

            print(f"Reasoning: {result.reasoning}")

        except Exception as e:
            print(f"Error evaluating {system}: {e}")
            # Conservative default: recommend chapter on error
            recommendations[system] = True

    return recommendations
