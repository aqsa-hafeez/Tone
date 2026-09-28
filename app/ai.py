import os
from groq import Groq

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Groq client reads GROQ_API_KEY from env automatically,
# but we pass it explicitly to fail fast with a clear error if missing.
_api_key = os.getenv("GROQ_API_KEY")
if not _api_key:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
        "from https://console.groq.com/keys"
    )

client = Groq(api_key=_api_key)

# Tone -> instruction sent to the model.
# Add new tones here and they immediately become usable through /tones and /rewrite.
TONE_PROMPTS = {
    "genz": "Rewrite this in casual Gen Z slang, keep it fun, relatable, and use light emoji if natural.",
    "formal": "Rewrite this in a formal, grammatically precise, polite tone suitable for a formal letter.",
    "corporate": "Rewrite this in polished, professional corporate business language, as if for a workplace email.",
    "sarcastic": "Rewrite this with a witty, sarcastic tone using irony, while keeping it non-offensive.",
    "poetic": "Rewrite this in a poetic, metaphorical, rhythmic style, like a short piece of verse.",
    "casual": "Rewrite this in a relaxed, friendly, conversational tone, as if texting a friend.",
}


def get_available_tones() -> list[str]:
    return list(TONE_PROMPTS.keys())


def rewrite_text(text: str, tone: str) -> str:
    """
    Calls Groq's chat completion endpoint (OpenAI-compatible) to rewrite
    `text` into the style described by `tone`. Raises ValueError for an
    unknown tone so the API layer can return a clean 400 error.
    """
    instruction = TONE_PROMPTS.get(tone)
    if instruction is None:
        raise ValueError(f"Unknown tone '{tone}'. Available tones: {get_available_tones()}")

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You rewrite text into a requested tone. You must strictly preserve "
                    "the original meaning, the original language, AND the original writing "
                    "script/alphabet — if the input is written in Roman letters (e.g. Roman "
                    "Urdu/Hindi, like 'bhai mujhe kal tak'), your output must also be in Roman "
                    "letters, never transliterated into Devanagari, Urdu (Nastaliq), or any other "
                    "script. Do not translate into a different language either. Only output the "
                    "rewritten text itself — no explanations, no quotation marks, no preamble."
                ),
            },
            {
                "role": "user",
                "content": f"{instruction}\n\nText to rewrite:\n{text}",
            },
        ],
        temperature=0.7,
        max_tokens=400,
    )

    return completion.choices[0].message.content.strip()
