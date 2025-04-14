import os
import chromadb
import streamlit as st
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.schema.document import Document
from langchain.prompts import PromptTemplate

from embedding_client import EmbeddingClient

# Page configuration
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# Add custom styling for a better chat UI
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stTextInput>div>div>input {
        background-color: white;
        border-radius: 20px;
    }
    .stChatMessage {
        background-color: white;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stChatMessage.user {
        background-color: #e6f7ff;
    }
    .stChatMessage.assistant {
        background-color: #f0f0f0;
    }
    /* Fix for text color in chat messages */
    .stChatMessage p, .stChatMessage div, .stMarkdown p {
        color: #333333 !important;
    }
    /* Ensure expander content is visible */
    .streamlit-expanderContent {
        background-color: #f8f9fa;
        color: #333333 !important;
    }
    .chat-container {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .sources-container {
        margin-top: 10px;
        padding: 10px;
        background-color: #f8f9fa;
        border-radius: 5px;
        font-size: 0.9em;
    }
    .chat-header {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    .chat-header h1 {
        margin: 0;
        margin-left: 10px;
    }
    /* Additional styling for better contrast */
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        margin-top: 10px;
    }
    .stExpander details summary {
        color: #1e88e5 !important;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Create a container for the chat interface
st.container()

# Header with improved styling
with st.container():
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown("<h1 style='font-size: 3rem; margin: 0;'>🤖</h1>", unsafe_allow_html=True)
    with col2:
        st.markdown("<h1>Smart AI Assistant</h1>", unsafe_allow_html=True)
        st.markdown("<p style='margin-top: -10px;'>I can answer questions from my knowledge base or use general knowledge!</p>", unsafe_allow_html=True)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add a welcome message
    st.session_state.messages.append({"role": "assistant", "content": "👋 Hi there! I'm your AI assistant. Ask me any question and I'll try to help you!"})

# Function to load the RAG system
@st.cache_resource
def load_rag_system():
    # Set up Gemini API key
    os.environ["GOOGLE_API_KEY"] = "AIzaSyBrzFVvP6MDAEBKtmqrtkdmnIOCMIkjfo0"
    
    # Use the client instead of loading the model
    embedding_model = EmbeddingClient(api_url="http://127.0.0.1:2000")
    persist_dir = "./chroma_store"
    
    # Create a ChromaDB client
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    collection_name = "pulkit_info"
    
    # Check if collection exists
    try:
        # Try to get the collection
        chroma_client.get_collection(collection_name)
        
        # Collection exists, load it
        vectordb = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=embedding_model
        )
        
    except ValueError:
        # Collection doesn't exist, create it
        raw_docs = [
            "Pulkit is a 24-year-old DevOps engineer working at Freecharge.com, which is owned by Axis Bank. He is deeply passionate about AI and wants to eventually build his own business.",
            "Pulkit dreams of retiring early, ideally by the age of 40 or 45. He hopes to accumulate significant wealth, have crores in his account, and live a life of freedom and purpose.",
            "Recently, Pulkit bought a graphic tablet. Although it performs well, it sometimes lags when used with web-based design tools like Figma.",
            "He recently visited Udaipur with his close friend Ankit. They explored local culture and food during the trip.",
            "Pulkit's relationship with his girlfriend has been turbulent lately. She believes his obsession with AI tools is a waste of time, but Pulkit disagrees and continues learning with enthusiasm.",
        ]
        documents = [Document(page_content=d) for d in raw_docs]
        
        # Create the Chroma vectorstore with the new client
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            client=chroma_client,
            collection_name=collection_name
        )
    
    # Use Gemini LLM via LangChain
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    
    # Two-stage prompt for query classification
    query_classifier_template = """
    You're a helpful AI assistant with access to a knowledge base. Your task is to analyze this query:
    
    QUERY: {query}
    
    Analyze this query and classify if it is best answered using:
    1. KNOWLEDGE_BASE - If the query appears to be asking for specific information that might be in a knowledge base
    2. GENERAL_KNOWLEDGE - If the query is asking for general information, facts, concepts, or questions that would be part of common knowledge
    
    Respond with ONLY one of these two classifications: KNOWLEDGE_BASE or GENERAL_KNOWLEDGE
    Don't explain your reasoning or add any additional text.
    """
    
    QUERY_CLASSIFIER_PROMPT = PromptTemplate(
        template=query_classifier_template,
        input_variables=["query"]
    )
    
    # Create a custom prompt template for RAG answers
    rag_template = """You are a helpful AI assistant with access to a knowledge base.

Answer the question based on the context provided. If the context doesn't contain the information needed to answer the question completely, just use what's available in the context and don't make up details.

If the question is completely unrelated to anything in the context, explicitly state that you don't have information about that in your knowledge base, and then try to provide a helpful general answer.

CONTEXT:
{context}

QUESTION:
{question}

YOUR ANSWER:"""

    RAG_PROMPT = PromptTemplate(
        template=rag_template,
        input_variables=["context", "question"]
    )
    
    # Create a retriever that doesn't filter by similarity threshold
    # We'll be using our LLM-based classifier instead
    retriever = vectordb.as_retriever(
        search_type="mmr",  # Use MMR to get diverse results
        search_kwargs={"k": 4, "fetch_k": 6}  # Retrieve more docs for better context
    )
    
    # RAG Chain with custom prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT}
    )
    
    return qa_chain, llm, QUERY_CLASSIFIER_PROMPT

# Load the RAG system with a nicer loading indicator
with st.spinner("🧠 Loading AI system..."):
    qa_chain, llm, QUERY_CLASSIFIER_PROMPT = load_rag_system()
    st.success("✅ AI system loaded successfully!")

# Create a chat container with better styling
chat_container = st.container()

# Display chat history with improved styling
with chat_container:
    for message in st.session_state.messages:
        avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            # Split content to separate answer from sources if it's an assistant message
            if message["role"] == "assistant" and "**Sources:**" in message["content"]:
                parts = message["content"].split("**Sources:**")
                st.markdown(parts[0])
                with st.expander("View Sources"):
                    st.markdown(f"**Sources:**{parts[1]}")
            else:
                st.markdown(message["content"])

# Chat input with improved styling
user_query = st.chat_input("Ask me anything...")

# When user submits a question
if user_query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Display user message
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_query)
    
    # Display assistant response with typing animation
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        
        with st.spinner("Thinking..."):
            # First, classify if this query is better answered with knowledge base or general knowledge
            classification_prompt = QUERY_CLASSIFIER_PROMPT.format(query=user_query)
            classification_response = llm.invoke(classification_prompt)
            classification = classification_response.content.strip()
            
            if "KNOWLEDGE_BASE" in classification:
                # This likely needs our knowledge base - use RAG
                response = qa_chain.invoke(user_query)
                answer = response['result']
                sources = [doc.page_content for doc in response["source_documents"]]
                
                # Verify that our answer actually uses the context
                # Sometimes even with RAG, if the context is irrelevant, the model might ignore it
                uses_context = any(
                    sentence in answer.lower() 
                    for doc in sources 
                    for sentence in doc.lower().split(".")
                    if len(sentence.strip()) > 15  # Only check substantial sentences
                )
                
                if not uses_context:
                    # Double-check with the model if retrieved context is relevant
                    relevance_check_prompt = f"""
                    QUERY: {user_query}
                    
                    RETRIEVED CONTEXT:
                    {' '.join(sources)}
                    
                    Is this context actually relevant for answering the query?
                    Answer with YES or NO only.
                    """
                    
                    relevance_check = llm.invoke(relevance_check_prompt)
                    is_relevant = "YES" in relevance_check.content.upper()
                    
                    if not is_relevant:
                        # If the model confirms context isn't relevant, use general knowledge
                        general_prompt = f"Answer this question using your general knowledge: {user_query}"
                        general_response = llm.invoke(general_prompt)
                        answer = general_response.content
                        sources = []
            else:
                # This is a general knowledge question - use direct LLM
                general_prompt = f"Answer this question using your general knowledge: {user_query}"
                general_response = llm.invoke(general_prompt)
                answer = general_response.content
                sources = []
            
            # Format the answer part
            answer_part = answer
            
            # Format the sources part if we have any
            if sources:
                sources_part = "**Sources:**\n"
                for i, source in enumerate(sources, 1):
                    sources_part += f"{i}. {source}\n"
                
                # Display the answer
                message_placeholder.markdown(answer_part)
                
                # Display sources in an expander
                with st.expander("View Sources"):
                    st.markdown(sources_part)
                
                # Add assistant response to chat history (combined for history)
                full_response = f"{answer_part}\n\n{sources_part}"
            else:
                # No sources, just display the answer
                message_placeholder.markdown(answer_part)
                full_response = answer_part
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# Add a footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Powered by Gemini + ChromaDB + LangChain</p>", unsafe_allow_html=True)