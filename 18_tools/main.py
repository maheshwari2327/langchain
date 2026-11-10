
"""18 · Tools & Function Calling

Run (with the venv active): python main.py
Uses local Ollama model: llama3.2
No OpenAI API key required.
"""

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


# 1. Define a tool
@tool
def add(a: int, b: int) -> int:
    """Add two whole numbers and return the sum."""
    return a + b


# 2. Initialize local LLM and bind the tool
llm = ChatOllama(
    model="llama3.2",
    temperature=0
).bind_tools([add])


# 3. Send a question
messages = [
    HumanMessage(content="What is 12 plus 30?")
]

ai_msg = llm.invoke(messages)
messages.append(ai_msg)


# 4. Execute tool calls requested by the model
for call in ai_msg.tool_calls:
    result = add.invoke(call["args"])

    messages.append(
        ToolMessage(
            content=str(result),
            tool_call_id=call["id"]
        )
    )


# 5. Get the final answer
print(llm.invoke(messages).content)
