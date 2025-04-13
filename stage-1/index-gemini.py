import os
import chromadb
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.schema.document import Document

from embedding_client import EmbeddingClient

# 1. --- Set up your Gemini API key ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyBrzFVvP6MDAEBKtmqrtkdmnIOCMIkjfo0"

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
        "Pulkit is 24 years old and works as a DevOps engineer at Freecharge.com.",
        "Freecharge is currently owned by Axis Bank and was previously owned by Snapdeal.",
        "Pulkit recently bought a graphic tablet that works well, though it lags slightly when used with online tools like Figma.",
        "Pulkit has a strict goal of retiring by the age of 40 or 45.",
        "He wants to have crores in his account and build a business of his own someday.",
        "A few days ago, Pulkit went to Udaipur with his friend Ankit.",
        "Just yesterday, Pulkit had a fight with his so-called girlfriend, who thinks his interest in AI and agentic tools is worthless.",
        "Pulkit believes in the potential of AI and is actively trying to learn more about it despite criticism."
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
user_query = "What will i need to learn to work with pulkit and in what way should i do that?"

response = qa_chain.invoke(user_query)

# 10. --- Print answer and sources ---
print("\n🧠 Answer:")
print(response['result'])

print("\n📚 Retrieved Sources:")
for doc in response["source_documents"]:
    print(" -", doc.page_content)
