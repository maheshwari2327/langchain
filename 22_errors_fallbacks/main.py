
"""22 · Errors, Retries & Fallbacks

Run (with the venv active): python main.py
Uses two local Ollama models.
No OpenAI or Anthropic API keys required.
"""

import json

from langchain_ollama import ChatOllama


# 1. Primary local model
primary = ChatOllama(
    model="llama3.2",
    temperature=0
)

# 2. Backup local model
backup = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)

# 3. Create fallback chain
robust = primary.with_fallbacks([backup])


# 4. Function to generate valid JSON with retries
def get_json(prompt: str, attempts: int = 3) -> dict:

    current_prompt = prompt

    for attempt in range(1, attempts + 1):

        print(f"\nAttempt {attempt}/{attempts}")

        text = robust.invoke(current_prompt).content.strip()

        try:
            return json.loads(text)

        except json.JSONDecodeError as e:

            print("Invalid JSON. Retrying...")

            current_prompt = (
                f"{prompt}\n\n"
                f"Your previous response was invalid JSON: {e}\n"
                "Return ONLY a valid JSON object. "
                "Do not include markdown or explanations."
            )

    raise ValueError("Model never returned valid JSON")


# 5. Execute
try:
    result = get_json(
        'Return JSON with keys "city" and "country" for Paris.'
    )

    print("\nFinal JSON result:")
    print(result)

except Exception as e:
    print(
        "Sorry, the assistant is busy. Please try again.",
        type(e).__name__
    )
