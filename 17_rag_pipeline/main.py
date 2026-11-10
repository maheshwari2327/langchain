
"""17 · RAG End to End

Run (with the venv active): python main.py
Uses local Ollama and Hugging Face embeddings.
Creates handbook.txt automatically.
"""

from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Index (once)
Path("handbook.txt").write_text(
    "Employees get 20 days of paid leave per year.\n"
    "Sick leave is 10 days and needs a doctor's note after 2 days.\n"
    "The office is open 9am to 6pm, Monday to Friday.\n",
    encoding="utf-8"
)

docs = TextLoader("handbook.txt", encoding="utf-8").load()

chunks = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
).split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

store = InMemoryVectorStore(embeddings)
store.add_documents(chunks)

retriever = store.as_retriever(search_kwargs={"k": 2})

# 2-4. Retrieve, augment, generate
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer ONLY from the context. If it is not there, say you do not know.\n\nContext:\n{context}"
    ),
    ("human", "{question}"),
])


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

rag = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("\nQuestion 1: How many leave days do I get?")
print("Answer:", rag.invoke("How many leave days do I get?"))

print("\nQuestion 2: What is the capital of France?")
print("Answer:", rag.invoke("What is the capital of France?"))
