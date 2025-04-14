# import os
# import chromadb
# from langchain_chroma import Chroma
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import RetrievalQA
# from langchain.schema.document import Document

# from embedding_client import EmbeddingClient

# # 1. --- Set up your Gemini API key ---
# os.environ["GOOGLE_API_KEY"] = "AIzaSyBrzFVvP6MDAEBKtmqrtkdmnIOCMIkjfo0"

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






# USE DATA FROM PDF OR TXT 



import os
import chromadb
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.schema.document import Document
from langchain.document_loaders import PyPDFLoader, TextLoader
from pathlib import Path


from embedding_client import EmbeddingClient

# 1. --- Set up your Gemini API key ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyBrzFVvP6MDAEBKtmqrtkdmnIOCMIkjfo0"

# Use the client instead of loading the model
embedding_model = EmbeddingClient(api_url="http://127.0.0.1:2000")
persist_dir = "./chroma_store"

# Create a ChromaDB client
print("🔄 Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=persist_dir)
collection_name = "pdf_story"

def get_vector_store():
    try:
        collection = chroma_client.get_collection(collection_name)
        if collection.count() == 0:
            raise chromadb.errors.InvalidCollectionException("Empty collection")
        return Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"}  # Better similarity metric
        )
    except:
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
            
            return loader.load()
        

        input_file = "story.pdf"  # or "notes.txt"
        documents = load_documents_from_file(input_file)


        return Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            client=chroma_client,
            collection_name=collection_name,
            collection_metadata={"hnsw:space": "cosine"}
        )

vectordb = get_vector_store()


# 6. --- Create retriever ---
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# 7. --- Use Gemini LLM ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

from langchain_core.prompts import PromptTemplate

# Helper: Score if retrieval is meaningful
def is_domain_info_relevant(retrieved_docs, threshold=0.6):
    """Use metadata/scores when available; fallback: content length or basic heuristics."""
    # If no documents, definitely not relevant
    if not retrieved_docs:
        return False
    
    # Heuristic: check for content length or score if exposed
    combined_length = sum(len(doc.page_content.strip()) for doc in retrieved_docs)
    return combined_length > 200  # or expose internal scores if Chroma supports it

# 8. --- Smart RAG QA Function ---
def smart_rag_qa(query: str):
    # Step 1: Retrieve relevant chunks
    retrieved_docs = retriever.get_relevant_documents(query)

    # Step 2: Check if they are helpful
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

# 9. --- Ask a question ---
user_query = "What is CI/CD?"

answer, sources = smart_rag_qa(user_query)

# 10. --- Print answer and sources ---
print("\n🧠 Answer:")
print(answer)

if sources:
    print("\n📚 Retrieved Sources:")
    for doc in sources:
        print(" -", doc.page_content)
else:
    print("\n📚 No relevant domain sources found. Used general knowledge.")