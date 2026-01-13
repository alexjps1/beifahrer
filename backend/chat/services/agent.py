import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .retriever import retrieve_context

# Initialize OpenAI LLM
# Uses OPENAI_API_KEY environment variable
_llm = None


def get_llm() -> ChatOpenAI:
    """
    Lazy-load and return the OpenAI LLM instance.
    Reuses the same instance across multiple calls.
    """
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    return _llm


def generate_response(user_message: str, chat_history: list[dict] = None) -> str:
    """
    Generate an assistant response using RAG (Retrieval-Augmented Generation).

    Args:
        user_message: The user's current message
        chat_history: Previous conversation history (list of dicts with 'role' and 'content')

    Returns:
        The assistant's response as a string
    """
    try:
        # Step 1: Retrieve relevant context from the knowledge base
        context = retrieve_context(user_message, k=3)

        # Step 2: Build the prompt
        system_prompt = """Du bist ein hilfreicher Fahrassistent für ein teilautomatisiertes Fahrzeug.
Deine Aufgabe ist es, Fragen zu den Fahrerassistenzsystemen zu beantworten.

Verwende die folgenden Informationen aus dem Fahrzeughandbuch, um die Frage des Benutzers zu beantworten:

{context}

Richtlinien:
- Antworte präzise und verständlich auf Deutsch
- Beziehe dich auf die bereitgestellten Informationen aus dem Handbuch
- Wenn die Informationen nicht ausreichen, sage ehrlich, dass du keine vollständige Antwort geben kannst
- Sei hilfsbereit und sicherheitsbewusst
- Halte deine Antworten kurz und auf den Punkt gebracht
"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{user_message}"),
            ]
        )

        # Step 3: Create the chain using LCEL (LangChain Expression Language)
        llm = get_llm()
        chain = prompt | llm | StrOutputParser()

        # Step 4: Generate response
        response = chain.invoke({"context": context, "user_message": user_message})

        return response.strip()

    except Exception as e:
        # Graceful error handling
        error_message = f"Entschuldigung, es ist ein Fehler aufgetreten: {str(e)}"
        print(f"Error in generate_response: {e}")
        return error_message
