
"""23 · Capstone: Study Buddy Agent

Run (with the venv active): python main.py
Put your .txt study notes in a notes/ folder next to main.py.
Uses local Ollama and Hugging Face embeddings.
No API keys required.
"""

from pathlib import Path

from langchain.agents import create_agent
from langchain_community.document_loaders import TextLoader
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1. Ingest study notes and build vector store
def build_store(folder="notes"):

    docs = []

    for path in Path(folder).glob("*.txt"):
        docs += TextLoader(
            str(path),
            encoding="utf-8"
        ).load()

    if not docs:
        raise FileNotFoundError(
            "No .txt study notes found in the notes folder."
        )

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    ).split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    store = InMemoryVectorStore(embeddings)
    store.add_documents(chunks)

    return store


store = build_store()


# 2. Define tools

@tool
def search_notes(query: str) -> str:
    """Search the student's study notes. Use this for any course question."""

    hits = store.similarity_search(query, k=3)

    if not hits:
        return "No relevant information found in the study notes."

    return "\n\n".join(
        f"[{d.metadata.get('source')}] {d.page_content}"
        for d in hits
    )


@tool
def calculate(expression: str) -> str:
    """Evaluate simple arithmetic like '12 * (3 + 4)'."""

    allowed = set("0123456789+-*/(). ")

    if not set(expression) <= allowed:
        return "Only numbers and + - * / ( ) are allowed."

    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception:
        return "Invalid arithmetic expression."


# 3. Initialize local LLM
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


# 4. Create Study Buddy Agent
SYSTEM = (
    "You are a Study Buddy. Call search_notes before answering course questions. "
    "Cite the source file. If the notes do not contain the answer, say so. "
    "Use calculate for arithmetic."
)

agent = create_agent(
    model=llm,
    tools=[search_notes, calculate],
    system_prompt=SYSTEM
)


# 5. Conversation memory
history = []


def ask(question: str) -> str:

    history.append({
        "role": "user",
        "content": question
    })

    result = agent.invoke(
        {"messages": history[-10:]},
        config={"recursion_limit": 8}
    )

    answer = result["messages"][-1].content

    history.append({
        "role": "assistant",
        "content": answer
    })

    return answer


# 6. Ask questions
print("\nQuestion 1: What is photosynthesis?")
print("Study Buddy:", ask("What is photosynthesis?"))

print("\nQuestion 2: Explain that more simply.")
print("Study Buddy:", ask("Explain that more simply."))
