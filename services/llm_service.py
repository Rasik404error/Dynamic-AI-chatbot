import os
import json
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv


# =====================================================
# CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

API_KEY = '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'

if not API_KEY:
    raise ValueError("OpenRouter API key not found.")


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
    timeout=60.0,
    max_retries=2
)

MODEL = "openrouter/free"


# =====================================================
# SENTIMENT + EMOTION + RESPONSE
# =====================================================

SYSTEM_PROMPT = """
You are a helpful conversational AI assistant.

Analyze the user's latest message and generate an appropriate response.

Return ONLY a valid JSON object with these keys:

{
    "sentiment": "positive, negative, or neutral",
    "emotion": "happy, sad, angry, frustrated, anxious,
                excited, confused, or neutral",
    "response": "your natural conversational response"
}

Rules:

1. Analyze the user's latest message, not the assistant's messages.
2. Use neutral when there is insufficient evidence of emotion.
3. Do not assume someone is sad or angry without evidence.
4. Adapt your response to the user's emotional context.
5. Be empathetic when the user expresses distress.
6. Remain professional and respectful.
7. Never claim that sentiment detection is perfectly accurate.
8. The response must answer the user's actual question.
9. Return valid JSON only, without Markdown fences.
"""


def generate_response(messages, preferences=None):

    """
    Generate a response and analyze the user's
    sentiment and emotion in one LLM API call.
    """

    preferences = preferences or []

    preference_text = ""

    if preferences:
        preference_text = (
            "\nUser preferences learned from feedback:\n"
            + "\n".join(f"- {p}" for p in preferences)
        )

    system_prompt = SYSTEM_PROMPT + preference_text

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            *messages
        ],
        temperature=0.5,
        max_tokens=1000
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("The LLM returned an empty response.")

    # Remove optional Markdown code fences
    content = content.strip()

    if content.startswith("```"):
        lines = content.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        content = "\n".join(lines)

    result = json.loads(content)

    sentiment = str(
        result.get("sentiment", "neutral")
    ).lower().strip()

    emotion = str(
        result.get("emotion", "neutral")
    ).lower().strip()

    answer = str(
        result.get("response", "")
    ).strip()

    valid_sentiments = {
        "positive",
        "negative",
        "neutral"
    }

    valid_emotions = {
        "happy",
        "sad",
        "angry",
        "frustrated",
        "anxious",
        "excited",
        "confused",
        "neutral"
    }

    if sentiment not in valid_sentiments:
        sentiment = "neutral"

    if emotion not in valid_emotions:
        emotion = "neutral"

    if not answer:
        raise ValueError("The LLM returned no answer.")

    return {
        "sentiment": sentiment,
        "emotion": emotion,
        "response": answer
    }
