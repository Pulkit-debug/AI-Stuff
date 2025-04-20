import os
import time
import numpy as np
import chromadb
import streamlit as st
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
# from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from pathlib import Path
from datetime import datetime
import base64

# Import the embedding client
from embedding_client import EmbeddingClient

# Page configuration
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin-bottom: 1rem;
        display: flex;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .chat-message.user {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
    }
    .chat-message.assistant {
        background-color: #f1f8e9;
        border-left: 5px solid #8bc34a;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }
    .chat-message .message {
        flex-grow: 1;
    }
    .source-docs {
        padding: 0.8rem;
        border-radius: 0.5rem;
        background-color: #f5f5f5;
        margin-top: 0.5rem;
        font-size: 0.85rem;
        border-left: 3px solid #ff9800;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .file-uploader {
        border: 2px dashed #ccc;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        transition: all 0.3s;
    }
    .file-uploader:hover {
        border-color: #4CAF50;
    }
    .title-container {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1rem;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        font-size: 0.8rem;
        color: #666;
    }
    .stMarkdown a {
        color: #2196f3;
        text-decoration: none;
    }
    .stMarkdown a:hover {
        text-decoration: underline;
    }
    .settings-box {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# Sidebar for settings
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>⚙️ Settings</h2>", unsafe_allow_html=True)
    
    st.markdown("### 🔑 API Keys")
    api_key = st.text_input("Google API Key", 
                            value=os.environ.get("GOOGLE_API_KEY", ""),
                            type="password",
                            help="Enter your Google API key")
    
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    st.markdown("### 🤖 Model Settings")
    with st.expander("Advanced Settings", expanded=False):
        st.markdown("#### Model Selection")
        model_option = st.selectbox(
            "Gemini Model",
            ["gemini-2.0-flash", "gemini-2.0-pro", "gemini-1.5-flash", "gemini-1.5-pro"],
            index=0
        )
        
        st.markdown("#### RAG Settings")
        k_docs = st.slider("Number of documents to retrieve", 1, 10, 3)
        
        st.markdown("#### Memory Settings")
        memory_limit = st.slider("Memory limit (number of exchanges)", 1, 20, 5,
                                help="Maximum number of conversation exchanges to remember")
        
        if st.button("Clear Conversation History"):
            st.session_state.messages = []
            st.session_state.conversation_memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            st.info("Conversation history cleared!")

    st.markdown("### 📊 Document Stats")
    if st.session_state.document_processed:
        st.success("✅ Document loaded and processed")
    else:
        st.warning("⚠️ No document loaded yet")

    st.markdown("### 🎨 Appearance")
    dark_mode = st.toggle("Dark Mode", value=st.session_state.dark_mode)
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        st.experimental_rerun()

# Document uploader function
def document_uploader():
    st.markdown("<h3 style='text-align: center;'>📄 Upload Your Document</h3>", unsafe_allow_html=True)
    
    # Create two columns
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a PDF or TXT file",
            type=["pdf", "txt"],
            help="Upload a document to create a knowledge base"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Space for alignment
        if uploaded_file is not None:
            if st.button("Process Document", use_container_width=True):
                with st.spinner("Processing document..."):
                    # Save uploaded file temporarily
                    temp_file_path = f"temp_{uploaded_file.name}"
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process the document
                    try:
                        process_document(temp_file_path)
                        st.session_state.document_processed = True
                        st.success(f"Document '{uploaded_file.name}' processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
                    
                    # Clean up
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)

# Function to process documents
def process_document(file_path):
    # Initialize embedding model
    embedding_model = EmbeddingClient(api_url="http://127.0.0.1:2000")
    persist_dir = "./chroma_store"
    
    # Create a ChromaDB client
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    collection_name = "pdf_story"
    
    # Delete existing collection if it exists
    try:
        chroma_client.delete_collection(collection_name)
    except:
        pass
    
    # Load documents based on file type
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Only .pdf and .txt are supported.")
    
    documents = loader.load()
    
    # Create vector store
    Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        client=chroma_client,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"}
    )

# Function to get vector store
def get_vector_store():
    embedding_model = EmbeddingClient(api_url="http://127.0.0.1:2000")
    persist_dir = "./chroma_store"
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    collection_name = "pdf_story"
    
    try:
        collection = chroma_client.get_collection(collection_name)
        if collection.count() == 0:
            raise chromadb.errors.InvalidCollectionException("Empty collection")
        return Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"}
        )
    except:
        st.error("No document has been processed yet. Please upload and process a document first.")
        return None

# Helper: Score if retrieval is meaningful
def is_domain_info_relevant(retrieved_docs, threshold=0.6):
    if not retrieved_docs:
        return False
    
    combined_length = sum(len(doc.page_content.strip()) for doc in retrieved_docs)
    return combined_length > 200

# Function to answer questions with memory
def answer_question(query, k=3, model="gemini-2.0-flash"):
    vectordb = get_vector_store()
    if not vectordb:
        return "Please upload and process a document first.", []
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model=model)
    
    # Create retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    
    # Get relevant docs
    retrieved_docs = retriever.get_relevant_documents(query)
    
    # Check if they're helpful
    if is_domain_info_relevant(retrieved_docs):
        # Use conversational retrieval chain with memory
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=st.session_state.conversation_memory,
            return_source_documents=True
        )
        
        # Get answer
        result = qa_chain({"question": query})
        return result['answer'], result.get("source_documents", [])
    else:
        # Use general knowledge
        conversation_history = st.session_state.conversation_memory.chat_memory.messages
        context = "\n".join([f"{msg.type}: {msg.content}" for msg in conversation_history])
        
        prompt = f"""
        Conversation history:
        {context}
        
        User question: {query}
        
        Please provide a helpful response:
        """
        
        response = llm.invoke(prompt)
        
        # Update memory
        st.session_state.conversation_memory.chat_memory.add_user_message(query)
        st.session_state.conversation_memory.chat_memory.add_ai_message(response.content)
        
        return response.content, []

# Function to create a pulsing animation
def animated_loading():
    progress_text = "Thinking..."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    my_bar.empty()

# Main page content
def main():
    # Beautiful header
    st.markdown("""
    <div class="title-container">
        <h1>🧠 Smart RAG Assistant</h1>
    </div>
    <p style="margin-top: -15px; margin-bottom: 30px;">
        Upload documents and ask questions with memory-enhanced retrieval.
    </p>
    """, unsafe_allow_html=True)
    
    # Document uploader
    document_uploader()
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Chat interface
    st.markdown("<h3 style='text-align: center;'>💬 Chat with Your Documents</h3>", unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "🤖"
        with st.container():
            st.markdown(f"""
            <div class="chat-message {message['role']}">
                <div class="avatar">{avatar}</div>
                <div class="message">{message['content']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display sources if available
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander("View sources", expanded=False):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"""
                        <div class="source-docs">
                            <strong>Source {i}:</strong> {source}
                        </div>
                        """, unsafe_allow_html=True)
    
    # Chat input
    query = st.chat_input("Ask a question about your document...")
    
    if query:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Show the user message immediately
        avatar = "👤"
        st.markdown(f"""
        <div class="chat-message user">
            <div class="avatar">{avatar}</div>
            <div class="message">{query}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show a thinking animation
        animated_loading()
        
        # Get model settings
        model = model_option if "model_option" in locals() else "gemini-2.0-flash"
        k = k_docs if "k_docs" in locals() else 3
        
        # Get answer
        answer, sources = answer_question(query, k=k, model=model)
        
        # Format sources for display
        formatted_sources = [doc.page_content for doc in sources] if sources else []
        
        # Add assistant message to chat
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "sources": formatted_sources
        })
        
        # Show the assistant message
        avatar = "🤖"
        st.markdown(f"""
        <div class="chat-message assistant">
            <div class="avatar">{avatar}</div>
            <div class="message">{answer}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show sources if available
        if formatted_sources:
            with st.expander("View sources", expanded=False):
                for i, source in enumerate(formatted_sources, 1):
                    st.markdown(f"""
                    <div class="source-docs">
                        <strong>Source {i}:</strong> {source}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Limit memory if needed
        if "memory_limit" in locals() and memory_limit > 0:
            if len(st.session_state.conversation_memory.chat_memory.messages) > memory_limit * 2:
                # Keep only the latest exchanges (each exchange is a user message and an AI response)
                memory_messages = st.session_state.conversation_memory.chat_memory.messages
                st.session_state.conversation_memory.chat_memory.messages = memory_messages[-memory_limit*2:]
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Smart RAG Assistant with Memory • Powered by Gemini • Built with Streamlit</p>
        <p>© 2025 • All rights reserved</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()