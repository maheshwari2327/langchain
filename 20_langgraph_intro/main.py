
"""20 · LangGraph Basics

Run (with the venv active): python main.py
Uses local LangGraph workflow. No API key required.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# 1. Define the state
class State(TypedDict):
    text: str
    count: int


# 2. Clean the text
def clean(state: State):
    return {"text": state["text"].strip()}


# 3. Count words
def count_words(state: State):
    return {"count": len(state["text"].split())}


# 4. Decide the route
def route(state: State) -> str:
    return "long" if state["count"] > 5 else "short"


# 5. Build the graph
builder = StateGraph(State)

builder.add_node("clean", clean)
builder.add_node("count", count_words)

builder.add_edge(START, "clean")
builder.add_edge("clean", "count")

builder.add_conditional_edges(
    "count",
    route,
    {
        "long": END,
        "short": END
    }
)


# 6. Compile and execute
graph = builder.compile()

result = graph.invoke({
    "text": "  hello big world  ",
    "count": 0
})

print(result)
