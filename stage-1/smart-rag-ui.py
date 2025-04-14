# VERSION -1 WORKING UMMM GOOD..

import streamlit as st
import os
import chromadb
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.document import Document
from embedding_client import EmbeddingClient

# Configuration
GOOGLE_API_KEY = "AIzaSyBrzFVvP6MDAEBKtmqrtkdmnIOCMIkjfo0"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
PERSIST_DIR = "./chroma_store"
COLLECTION_NAME = "NovaMind_info"
SEARCH_KWARGS = {"k": 6, "score_threshold": 0.4, "fetch_k": 20}

# Cache resources
@st.cache_resource
def load_rag_system():
    embedding_model = EmbeddingClient(api_url="http://127.0.0.1:2000")
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    
    # Vector store initialization
    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
        if collection.count() == 0:
            raise ValueError("Empty collection")
        vectordb = Chroma(
            client=chroma_client,
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"}
        )
    except:
        raw_docs = [Document(page_content=d) for d in [
            # Your document texts here
        ]]
        vectordb = Chroma.from_documents(
            documents=raw_docs,
            embedding=embedding_model,
            client=chroma_client,
            collection_name=COLLECTION_NAME,
            collection_metadata={"hnsw:space": "cosine"}
        )

    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs=SEARCH_KWARGS
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )

    return llm, retriever

# Initialize app
st.set_page_config(page_title="NovaMind AI Assistant", page_icon="🤖")
st.title("NovaMind Smart Conversational Agent")

# Session state management
if "history" not in st.session_state:
    st.session_state.history = []

# Display chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.markdown(f"📄 {src}")

# User input
if prompt := st.chat_input("Ask about NovaMind AI:"):
    # Add user message to history
    st.session_state.history.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Analyzing..."):
        llm, retriever = load_rag_system()
        
        try:
            # Smart retrieval with history context
            expanded_query = f"{prompt} - company information, business details, product features"
            retrieved_docs = retriever.get_relevant_documents(expanded_query)
            context = "\n".join({d.page_content for d in retrieved_docs})

            # Format prompt with history
            history_context = "\n".join(
                [f"{msg['role']}: {msg['content']}" 
                 for msg in st.session_state.history[-4:]]  # Keep last 4 messages
            )
            
            final_prompt = f"""**Conversation History**
{history_context}

**Current Context**
{context if context else "No relevant documents found"}

**Question**
{prompt}

**Instructions**
1. {"Use context first" if context else "Use general knowledge"}
2. {"Maintain conversation flow" if history_context else "Be concise"}
3. {"Cite sources" if context else "Mention this is general knowledge"}"""

            # Generate response
            response = llm.invoke(final_prompt)
            answer = response.content

            # Add to history
            st.session_state.history.append({
                "role": "assistant",
                "content": answer,
                "sources": [d.page_content[:150] + "..." for d in retrieved_docs[:3]] if context else []
            })

            # Display response
            with st.chat_message("assistant"):
                st.markdown(answer)
                if context:
                    with st.expander("Relevant Sources"):
                        for doc in retrieved_docs[:3]:
                            st.markdown(f"🔗 {doc.page_content[:120]}...")

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.history.append({
                "role": "assistant",
                "content": f"Error processing request: {str(e)}"
            })

# Sidebar controls
with st.sidebar:
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()
    st.markdown("---")
    st.markdown("**Smart Features:**")
    st.markdown("- Context-aware responses")
    st.markdown("- Conversation history tracking")
    st.markdown("- Automatic source verification")














#   VERSION - 2 TESTING 



# import streamlit as st
# import os
# import chromadb
# import numpy as np
# from langchain_chroma import Chroma
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.schema.document import Document
# from embedding_client import EmbeddingClient

# # Configuration
# GOOGLE_API_KEY = "AIzaSyBrzFVvP6MDAEBKtmqrtkdmnIOCMIkjfo0"
# os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
# PERSIST_DIR = "./chroma_store"
# COLLECTION_NAME = "NovaMind_info"
# SEARCH_KWARGS = {"k": 6}  # Removed score threshold

# # Cache resources with enhanced validation
# @st.cache_resource
# def load_rag_system():
#     embedding_model = EmbeddingClient(api_url="http://127.0.0.1:2000")
#     chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    
#     # Vector store initialization with validation
#     try:
#         collection = chroma_client.get_collection(COLLECTION_NAME)
#         if collection.count() == 0:
#             raise ValueError("Empty collection")
#         vectordb = Chroma(
#             client=chroma_client,
#             collection_name=COLLECTION_NAME,
#             embedding_function=embedding_model,
#             collection_metadata={"hnsw:space": "cosine"}
#         )
#     except:
#         raw_docs = [Document(page_content=d) for d in [
#             "NovaMind AI is a startup founded in 2024 with the mission of building intelligent AI agents for enterprise workflows.",
#             "The founding team consists of three members: Aisha Kapoor (CEO), Rahul Mehta (CTO), and Jenny Lin (COO), all with strong backgrounds in AI research and product development.",
#             # ... other documents ...
#         ]]
#         vectordb = Chroma.from_documents(
#             documents=raw_docs,
#             embedding=embedding_model,
#             client=chroma_client,
#             collection_name=COLLECTION_NAME,
#             collection_metadata={"hnsw:space": "cosine"}
#         )

#     retriever = vectordb.as_retriever(
#         search_type="similarity",  # Changed from MMR to basic similarity
#         search_kwargs=SEARCH_KWARGS
#     )

#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.0-flash",
#         temperature=0.3,
#         google_api_key=GOOGLE_API_KEY
#     )

#     return llm, retriever, embedding_model

# # Enhanced relevance checking
# def is_context_relevant(query: str, docs: list, embedder: EmbeddingClient) -> bool:
#     if not docs:
#         return False
    
#     try:
#         query_embedding = embedder.embed_query(query)
#         doc_embeddings = [embedder.embed_query(d.page_content) for d in docs]
#         similarities = [
#             np.dot(query_embedding, doc_embed) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embed))
#             for doc_embed in doc_embeddings
#         ]
#         return max(similarities) > 0.5  # Check max similarity instead of average
#     except:
#         return False

# # Response generation pipeline
# def generate_response(query: str, llm, retriever, embedder, history: list) -> tuple:
#     # Retrieve documents with expanded query
#     expanded_query = f"{query} organization team product details"  # Better expansion terms
#     retrieved_docs = retriever.get_relevant_documents(expanded_query)
    
#     # Enhanced relevance check
#     context_valid = is_context_relevant(query, retrieved_docs, embedder)
    
#     # Prepare context
#     context = "\n".join({d.page_content for d in retrieved_docs}) if context_valid else ""
    
#     # Generate answer with strict prompt
#     prompt_template = f"""**Response Protocol**
# 1. Available Context:
# {context if context else "No relevant company information available"}

# 2. User Question:
# {query}

# **Instruction Set**
# - Answer using context when available
# - Be precise with names and numbers
# - For partial info: "According to company documents: [known facts]. No data on [missing info]"
# - General knowledge: Answer normally without context references"""

#     response = llm.invoke(prompt_template)
#     return response.content, retrieved_docs if context_valid else []

# # UI Setup
# st.set_page_config(page_title="NovaMind AI Assistant", page_icon="🤖")
# st.title("NovaMind Enterprise Knowledge Assistant")

# # Session state management
# if "history" not in st.session_state:
#     st.session_state.history = []

# # Display chat history
# for msg in st.session_state.history:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])
#         if msg.get("sources"):
#             with st.expander("Verified Sources"):
#                 for src in msg["sources"]:
#                     st.markdown(f"📄 {src}")

# # User input handling
# if prompt := st.chat_input("Ask about NovaMind AI:"):
#     st.session_state.history.append({"role": "user", "content": prompt})
    
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.spinner("Verifying information..."):
#         try:
#             llm, retriever, embedder = load_rag_system()
#             answer, sources = generate_response(prompt, llm, retriever, embedder, st.session_state.history)
            
#             # Add to history
#             st.session_state.history.append({
#                 "role": "assistant",
#                 "content": answer,
#                 "sources": [d.page_content[:150] + "..." for d in sources[:3]] if sources else []
#             })

#             # Display response
#             with st.chat_message("assistant"):
#                 st.markdown(answer)
#                 if sources:
#                     with st.expander("Verified Sources"):
#                         for doc in sources[:3]:
#                             st.markdown(f"🔍 {doc.page_content[:120]}...")

#         except Exception as e:
#             st.error(f"System error: {str(e)}")
#             st.session_state.history.append({
#                 "role": "assistant",
#                 "content": f"Error processing request: {str(e)}"
#             })

# # Sidebar controls
# with st.sidebar:
#     if st.button("🧹 Clear Conversation History"):
#         st.session_state.history = []
#         st.rerun()
    
#     st.markdown("---")
#     st.markdown("**System Features**")
#     st.markdown("✅ Hallucination-protected responses  \n"
#                 "🔍 Context-aware verification  \n"
#                 "📑 Source-tracked answers  \n"
#                 "🤖 Smart conversation memory")