
"""16 · Retrievers

Run (with the venv active): python main.py
Uses local Hugging Face embeddings.
"""

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

store = InMemoryVectorStore(embeddings)

store.add_documents([
    Document(page_content="Pay by card, UPI or cash on delivery."),
    Document(page_content="Online payments are processed securely."),
    Document(page_content="Refunds go back to the original payment method."),
    Document(page_content="Delivery takes 3 to 5 working days."),
])

similar = store.as_retriever(search_kwargs={"k": 3})

varied = store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3}
)

for name, r in [("similarity", similar), ("mmr", varied)]:
    print("--", name)

    for d in r.invoke("How can I pay?"):
        print("  ", d.page_content)
