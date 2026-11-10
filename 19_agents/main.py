
"""19 · Agents

Run (with the venv active): python main.py
Uses local Ollama model: llama3.2
No OpenAI API key required.
"""

import datetime

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


# 1. Define addition tool
@tool
def add(a: int, b: int) -> int:
    """Add two whole numbers."""
    return a + b


# 2. Define today's date tool
@tool
def today() -> str:
    """Return today's date."""
    return datetime.date.today().isoformat()


# 3. Initialize local Ollama model
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


# 4. Create agent with tools
agent = create_agent(
    model=llm,
    tools=[add, today],
    system_prompt="You are a helpful assistant. Use tools when they help."
)


# 5. Execute agent
result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What is today's date, and what is 15 + 27?"
        }
    ]
})


# 6. Display every agent step
for msg in result["messages"]:
    print(msg.type, "|", msg.content)
