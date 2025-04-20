# import os
# import chromadb
# from langchain_chroma import Chroma
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import RetrievalQA
# from langchain.schema.document import Document

# from embedding_client import EmbeddingClient

# # 1. --- Set up your Gemini API key ---
# os.environ["GOOGLE_API_KEY"] = "api"

# # Use the client instead of loading the model
# embedding_model = EmbeddingClient(api_url="http://127.0.0.1:2000")
# persist_dir = "./chroma_store"

# # Create a ChromaDB client
# print("🔄 Connecting to ChromaDB...")
# chroma_client = chromadb.PersistentClient(path=persist_dir)
# collection_name = "pulkit_info"

# # Check if collection exists
# try:
#     # Try to get the collection
#     print("🔍 Checking for existing collection...")
#     chroma_client.get_collection(collection_name)
#     print("✅ Found existing collection, loading vector DB...")
    
#     # Collection exists, load it
#     vectordb = Chroma(
#         client=chroma_client,
#         collection_name=collection_name,
#         embedding_function=embedding_model
#     )
    
# except ValueError:
#     # Collection doesn't exist, create it
#     print("🆕 Collection not found, creating new vector DB...")
#     raw_docs = [
#         "Pulkit is 24 years old and works as a DevOps engineer at Freecharge.com.",
#         "Freecharge is currently owned by Axis Bank and was previously owned by Snapdeal.",
#         "Pulkit recently bought a graphic tablet that works well, though it lags slightly when used with online tools like Figma.",
#         "Pulkit has a strict goal of retiring by the age of 40 or 45.",
#         "He wants to have crores in his account and build a business of his own someday.",
#         "A few days ago, Pulkit went to Udaipur with his friend Ankit.",
#         "Just yesterday, Pulkit had a fight with his so-called girlfriend, who thinks his interest in AI and agentic tools is worthless.",
#         "Pulkit believes in the potential of AI and is actively trying to learn more about it despite criticism."
#     ]
#     documents = [Document(page_content=d) for d in raw_docs]
    
#     # Create the Chroma vectorstore with the new client
#     vectordb = Chroma.from_documents(
#         documents=documents,
#         embedding=embedding_model,
#         client=chroma_client,
#         collection_name=collection_name
#     )


# # 6. --- Create retriever ---
# retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# # 7. --- Use Gemini LLM ---
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# from langchain_core.prompts import PromptTemplate

# # Helper: Score if retrieval is meaningful
# def is_domain_info_relevant(retrieved_docs, threshold=0.6):
#     """Use metadata/scores when available; fallback: content length or basic heuristics."""
#     # If no documents, definitely not relevant
#     if not retrieved_docs:
#         return False
    
#     # Heuristic: check for content length or score if exposed
#     combined_length = sum(len(doc.page_content.strip()) for doc in retrieved_docs)
#     return combined_length > 200  # or expose internal scores if Chroma supports it

# # 8. --- Smart RAG QA Function ---
# def smart_rag_qa(query: str):
#     # Step 1: Retrieve relevant chunks
#     retrieved_docs = retriever.get_relevant_documents(query)

#     # Step 2: Check if they are helpful
#     if is_domain_info_relevant(retrieved_docs):
#         print("📚 Using domain knowledge (RAG)...")
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             retriever=retriever,
#             return_source_documents=True
#         )
#         response = qa_chain.invoke(query)
#         return response['result'], response["source_documents"]
#     else:
#         print("🌐 Falling back to general LLM knowledge...")
#         response = llm.invoke(query)
#         return response.content, []

# # 9. --- Ask a question ---
# user_query = "What should I learn to get into DevOps?"

# answer, sources = smart_rag_qa(user_query)

# # 10. --- Print answer and sources ---
# print("\n🧠 Answer:")
# print(answer)

# if sources:
#     print("\n📚 Retrieved Sources:")
#     for doc in sources:
#         print(" -", doc.page_content)
# else:
#     print("\n📚 No relevant domain sources found. Used general knowledge.")














# OBEDIENT GEMINI WITH PDF DATA UPLOAD 
# WORKING SETUP 1




# import os
# import chromadb
# from langchain_chroma import Chroma
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import RetrievalQA
# from langchain.schema.document import Document

# from embedding_client import EmbeddingClient
# from langchain.document_loaders import PyPDFLoader, TextLoader
# from pathlib import Path

# # --- Set up Gemini API key ---
# os.environ["GOOGLE_API_KEY"] = "api"

# # --- Embedding and Chroma setup ---
# embedding_model = EmbeddingClient(api_url="http://127.0.0.1:2000")
# persist_dir = "./chroma_store"
# collection_name = "obviousadam_story"

# print("🔄 Connecting to ChromaDB...")
# chroma_client = chromadb.PersistentClient(path=persist_dir)

# # --- Load documents from PDF or TXT ---
# def load_documents_from_file(file_path: str):
#     ext = Path(file_path).suffix.lower()
#     if ext == ".pdf":
#         print(f"📄 Loading PDF: {file_path}")
#         loader = PyPDFLoader(file_path)
#     elif ext == ".txt":
#         print(f"📄 Loading TXT: {file_path}")
#         loader = TextLoader(file_path)
#     else:
#         raise ValueError("❌ Unsupported file type. Only .pdf and .txt are supported.")
    
#     raw_docs = loader.load()
    
#     # Filter out empty documents
#     cleaned_docs = [
#         doc for doc in raw_docs if doc.page_content.strip()
#     ]

#     if not cleaned_docs:
#         raise ValueError("❌ No valid (non-empty) pages found in the document.")

#     print(f"✅ Loaded {len(cleaned_docs)} valid pages.")
#     return cleaned_docs

# # Path to your input document
# input_file = "obviousadam.pdf"  # Change to your actual file name: "data.txt" or "notes.pdf"

# print("🔍 Checking for existing collection...")

# try:
#     # Try to get existing collection
#     collection = chroma_client.get_collection(collection_name)
#     print("✅ Found existing collection, checking if it's populated...")

#     # Check number of documents in the collection
#     if len(collection.get()["ids"]) > 0:
#         print("📦 Collection is populated. Loading vector DB...")
#         vectordb = Chroma(
#             client=chroma_client,
#             collection_name=collection_name,
#             embedding_function=embedding_model
#         )
#     else:
#         print("📭 Collection is empty. Loading from file and creating embeddings...")
#         documents = load_documents_from_file(input_file)
#         vectordb = Chroma.from_documents(
#             documents=documents,
#             embedding=embedding_model,
#             client=chroma_client,
#             collection_name=collection_name
#         )

# except chromadb.errors.InvalidCollectionException:
#     print("🆕 Collection not found. Creating new one from file...")
#     documents = load_documents_from_file(input_file)
#     vectordb = Chroma.from_documents(
#         documents=documents,
#         embedding=embedding_model,
#         client=chroma_client,
#         collection_name=collection_name
#     )


# # --- Retriever ---
# retriever = vectordb.as_retriever(search_kwargs={"k": 3})
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# # --- Relevance Checker ---
# def is_domain_info_relevant(retrieved_docs, threshold=0.6):
#     if not retrieved_docs:
#         return False
#     combined_length = sum(len(doc.page_content.strip()) for doc in retrieved_docs)
#     return combined_length > 200

# # --- Smart RAG QA ---
# def smart_rag_qa(query: str):
#     retrieved_docs = retriever.get_relevant_documents(query)
#     if is_domain_info_relevant(retrieved_docs):
#         print("📚 Using domain knowledge (RAG)...")
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             retriever=retriever,
#             return_source_documents=True
#         )
#         response = qa_chain.invoke(query)
#         return response['result'], response["source_documents"]
#     else:
#         print("🌐 Falling back to general LLM knowledge...")
#         response = llm.invoke(query)
#         return response.content, []

# # --- Ask a Question ---
# user_query = "Who is the president of india? and what do you know about devops?"

# answer, sources = smart_rag_qa(user_query)

# # --- Output ---
# print("\n🧠 Answer:")
# print(answer)

# if sources:
#     print("\n📚 Retrieved Sources:")
#     for doc in sources:
#         print(" -", doc.page_content)
# else:
#     print("\n📚 No relevant domain sources found. Used general knowledge.")

















# SETUP 2 







import os
import re
import chromadb
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.schema.document import Document
import textwrap
from embedding_client import EmbeddingClient
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from pathlib import Path

# --- Set up Gemini API key ---
os.environ["GOOGLE_API_KEY"] = "api"

# --- Embedding and Chroma setup ---
embedding_model = EmbeddingClient(api_url="http://127.0.0.1:2000")
persist_dir = "./chroma_store"
collection_name = "obviousadam_story"

print("🔄 Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=persist_dir)

# --- Load documents from PDF or TXT ---
def load_documents_from_file(file_path: str):
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        print(f"📄 Loading PDF: {file_path}")
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        print(f"📄 Loading TXT: {file_path}")
        loader = TextLoader(file_path)
    else:
        raise ValueError("❌ Unsupported file type. Only .pdf and .txt are supported.")
    
    raw_docs = loader.load()
    
    # Filter out empty documents
    cleaned_docs = [
        doc for doc in raw_docs if doc.page_content.strip()
    ]

    if not cleaned_docs:
        raise ValueError("❌ No valid (non-empty) pages found in the document.")

    print(f"✅ Loaded {len(cleaned_docs)} valid pages.")
    return cleaned_docs

def pretty_source(text: str) -> str:
    # Step 1: Replace line breaks and reduce excess whitespace
    cleaned = re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()

    # Step 2: Add space between camel case / smashed words (e.g., "BU SI NE SS" -> "BUSINESS")
    cleaned = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', cleaned)

    # Step 3: Fix broken up capital words (e.g., "BU SI NE SS" -> "BUSINESS")
    cleaned = re.sub(r'\b(?:[A-Z]{1,2}\s+){2,}[A-Z]{1,2}\b', 
                     lambda m: m.group(0).replace(' ', ''), cleaned)

    # Step 4: Word wrap to 100 chars and truncate preview
    wrapped = textwrap.fill(cleaned, width=100)
    preview = wrapped[:500] + "..." if len(wrapped) > 500 else wrapped
    return preview


# Path to your input document
input_file = "obviousadam.pdf"

print("🔍 Checking for existing collection...")

try:
    # Try to get existing collection
    collection = chroma_client.get_collection(collection_name)
    print("✅ Found existing collection, checking if it's populated...")

    # Check number of documents in the collection
    if len(collection.get()["ids"]) > 0:
        print("📦 Collection is populated. Loading vector DB...")
        vectordb = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=embedding_model
        )
    else:
        print("📭 Collection is empty. Loading from file and creating embeddings...")
        documents = load_documents_from_file(input_file)
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            client=chroma_client,
            collection_name=collection_name
        )

except chromadb.errors.InvalidCollectionException:
    print("🆕 Collection not found. Creating new one from file...")
    documents = load_documents_from_file(input_file)
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        client=chroma_client,
        collection_name=collection_name
    )

# --- Retriever ---
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# --- Relevance Checker ---
def is_domain_info_relevant(retrieved_docs, threshold=0.6):
    if not retrieved_docs:
        return False
    combined_length = sum(len(doc.page_content.strip()) for doc in retrieved_docs)
    return combined_length > 200

# --- Smart RAG QA ---
def smart_rag_qa(query: str):
    retrieved_docs = retriever.invoke(query)
    if is_domain_info_relevant(retrieved_docs):
        print("📚 Using domain knowledge (RAG)...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
        response = qa_chain.invoke(query)
        return response['result'], response["source_documents"]
    else:
        print("🌐 Falling back to general LLM knowledge...")
        response = llm.invoke(query)
        return response.content, []

# --- Ask a Question ---
user_query = "What did adam figure out why hats are not able to sell? explain"

answer, sources = smart_rag_qa(user_query)

# --- Output ---
print("\n🧠 Answer:")
print(answer)

if sources:
    print("\n📚 Retrieved Sources:")
    for doc in sources:
        source_text = doc.page_content.strip().replace('\n', ' ')
        wrapped = textwrap.fill(pretty_source(source_text), width=100)
        preview = wrapped[:500] + "..." if len(wrapped) > 500 else wrapped
        print(f" - {preview}\n")
else:
    print("\n📚 No relevant domain sources found. Used general knowledge.")
