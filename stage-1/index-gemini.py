import os
import chromadb
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.schema.document import Document

from embedding_client import EmbeddingClient

# 1. --- Set up your Gemini API key ---
os.environ["GOOGLE_API_KEY"] = "api"

# Use the client instead of loading the model
embedding_model = EmbeddingClient(api_url="http://127.0.0.1:2000")
persist_dir = "./chroma_store"

# Create a ChromaDB client
print("🔄 Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=persist_dir)
collection_name = "pulkit_info"

# Check if collection exists
try:
    # Try to get the collection
    print("🔍 Checking for existing collection...")
    chroma_client.get_collection(collection_name)
    print("✅ Found existing collection, loading vector DB...")
    
    # Collection exists, load it
    vectordb = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embedding_model
    )
    
except ValueError:
    # Collection doesn't exist, create it
    print("🆕 Collection not found, creating new vector DB...")
    raw_docs = [
        asdfasdfasda
    ]
    documents = [Document(page_content=d) for d in raw_docs]
    
    # Create the Chroma vectorstore with the new client
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        client=chroma_client,
        collection_name=collection_name
    )

# 6. --- Create retriever ---
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# 7. --- Use Gemini LLM via LangChain ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# 8. --- RAG Chain: Use retrieved context + LLM to answer questions ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 9. --- Ask a question ---
user_query = "what should i learn from pulkit and how do i dive into devops what do i do?"

response = qa_chain.invoke(user_query)

# 10. --- Print answer and sources ---
print("\n🧠 Answer:")
print(response['result'])

print("\n📚 Retrieved Sources:")
for doc in response["source_documents"]:
    print(" -", doc.page_content)
