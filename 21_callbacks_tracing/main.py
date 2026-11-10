
"""21 · Callbacks & Tracing

Run (with the venv active): python main.py
Uses local Ollama model: llama3.2
No OpenAI API key required.
"""

import time

from langchain_core.callbacks import BaseCallbackHandler
from langchain_ollama import ChatOllama


# 1. Create a custom callback handler
class Timer(BaseCallbackHandler):

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.start = time.perf_counter()
        print("[start] model called")

    def on_llm_end(self, response, **kwargs):
        elapsed = time.perf_counter() - self.start
        print(f"[end] took {elapsed:.2f}s")

    def on_llm_error(self, error, **kwargs):
        print("[error]", error)


# 2. Initialize local Ollama model
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


# 3. Invoke model with callback
response = llm.invoke(
    "Explain callbacks in one line.",
    config={"callbacks": [Timer()]}
)


# 4. Print the model response
print("Response:", response.content)


# Optional: LangSmith tracing requires a separate LangSmith API key.
# It is not needed to run this local exercise.
