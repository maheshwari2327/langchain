"""02 · Your First LLM Call

Run (with the venv active):  python main.py
Needs OPENAI_API_KEY in a .env file (see ../.env.example).
"""


from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

question = "Explain Python in one sentence."
response = llm.invoke(question)

print("Answer:", response.content)
print("Type:", type(response).__name__)
print("Tokens:", response.usage_metadata)
