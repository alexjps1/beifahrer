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

KRITISCH – Diversifizierung der Fragen:
- Jede der drei Fragen MUSS sich auf ein ANDERES Fahrerassistenzsystem beziehen
- Verteile die Fragen über verschiedene Systeme, um ein breites Verständnis zu bekommen
- Priorisiere Systeme mit "mittleren" Bewertungen (2-4), da hier die größte Unsicherheit besteht
- Wenn mehrere Systeme in Frage kommen, wähle drei unterschiedliche aus
- VERMEIDE es, alle drei Fragen auf dasselbe System zu fokussieren

Prioritäten bei der System-Auswahl:
1. Systeme mit mean-Werten zwischen 2 und 4 (höchste Priorität – "wenig" bis "eher viel")
2. Systeme mit Diskrepanzen zwischen practical und theoretical (Differenz >= 2)
3. Systeme mit einzelnen Werten von 1, 2, 3 oder 4 (auch wenn mean anders ist)
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
- Sich jeweils auf die unterschiedlichen Systeme beziehen
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
                    followup_info += f"Frage: {qa.get('q', 'N/A')}\nAntwort: {qa.get('a', 'N/A')}\n\n"
            else:
                followup_info = f"Keine Folgefragen zu {system} wurden gestellt."

            # Invoke the chain for this system
            result = chain.invoke(
                {
                    "system_name": system,
                    "practical": system_data.get("practical", 0),
                    "theoretical": system_data.get("theoretical", 0),
                    "mean": system_data.get("mean", 0),
                    "followup_qa": followup_info,
                }
            )

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
