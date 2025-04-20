# SMART RAG TAKE #2 TRYING TO MAKE IT BETTER
# WORKING LIKE A CHARM
# has issue with telling which asnwer is what


# import streamlit as st
# import os
# import chromadb
# from langchain_chroma import Chroma
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.schema.document import Document
# from embedding_client import EmbeddingClient
# from typing import List, Optional
# from langchain_community.document_loaders import PyPDFLoader, TextLoader
# import tempfile
# import re


# # Configuration
# GOOGLE_API_KEY = "api"
# os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
# PERSIST_DIR = "./chroma_store"
# COLLECTION_NAME = "normal_story"
# SEARCH_KWARGS = {"k": 6, "score_threshold": 0.4, "fetch_k": 20}


# def clean_text(text: str) -> str:
#     # Basic spacing fix: add space between lowercase-uppercase (e.g., homeHe → home He)
#     text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
#     # Add space between a word and a capitalized word following it
#     text = re.sub(r'([a-zA-Z])([A-Z][a-z])', r'\1 \2', text)

#     # Replace multiple spaces with one
#     text = re.sub(r'\s{2,}', ' ', text)

#     return text.strip()

# def process_uploaded_files(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> List[Document]:
#     """Process uploaded files into LangChain Documents"""
#     docs = []
#     for file in uploaded_files:
#         with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#             temp_file.write(file.getvalue())
#             temp_path = temp_file.name
        
#         try:
#             if file.name.endswith(".pdf"):
#                 loader = PyPDFLoader(temp_path)
#                 # docs.extend(loader.load())
#                 docs.extend([
#                     Document(page_content=clean_text(doc.page_content), metadata=doc.metadata)
#                     for doc in loader.load()
#                 ])
#             elif file.name.endswith((".txt", ".md")):
#                 loader = TextLoader(temp_path)
#                 # docs.extend(loader.load())
#                 docs.extend([
#                     Document(page_content=clean_text(doc.page_content), metadata=doc.metadata)
#                     for doc in loader.load()
#                 ])
#             else:
#                 st.warning(f"Unsupported file type: {file.name}")
#         except Exception as e:
#             st.error(f"Error processing {file.name}: {str(e)}")
#         finally:
#             os.remove(temp_path)
#     return docs

# @st.cache_resource
# def load_rag_system(uploaded_files: Optional[List] = None):
#     embedding_model = EmbeddingClient(api_url="http://127.0.0.1:2000")
#     chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    
#     # Process uploaded files if any
#     new_docs = process_uploaded_files(uploaded_files) if uploaded_files else []

#     try:
#         collection = chroma_client.get_collection(COLLECTION_NAME)
#         existing_count = collection.count()
        
#         # Add new documents if any
#         if new_docs:
#             ids = [str(existing_count + i) for i in range(len(new_docs))]
#             collection.add(
#                 documents=[doc.page_content for doc in new_docs],
#                 ids=ids
#             )
            
#         vectordb = Chroma(
#             client=chroma_client,
#             collection_name=COLLECTION_NAME,
#             embedding_function=embedding_model,
#             collection_metadata={"hnsw:space": "cosine"}
#         )
#     except:
#         # Create new collection with initial documents
#         raw_docs = new_docs if new_docs else [Document(page_content="NovaMind AI default information")]
#         vectordb = Chroma.from_documents(
#             documents=raw_docs,
#             embedding=embedding_model,
#             client=chroma_client,
#             collection_name=COLLECTION_NAME,
#             collection_metadata={"hnsw:space": "cosine"}
#         )

#     retriever = vectordb.as_retriever(
#         search_type="mmr",
#         search_kwargs=SEARCH_KWARGS
#     )

#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.0-flash",
#         temperature=0.3,
#         google_api_key=GOOGLE_API_KEY
#     )

#     return llm, retriever, vectordb._collection.count()

# # Initialize app
# st.set_page_config(page_title="NovaMind AI Assistant", page_icon="🤖")
# st.title("NovaMind Smart Conversational Agent")

# # Session state management
# if "history" not in st.session_state:
#     st.session_state.history = []
# if "uploaded_files" not in st.session_state:
#     st.session_state.uploaded_files = []

# # Sidebar controls
# with st.sidebar:
#     st.header("Document Management")
#     uploaded_files = st.file_uploader(
#         "Upload knowledge documents",
#         type=["pdf", "txt", "md"],
#         accept_multiple_files=True
#     )
    
#     if uploaded_files and st.button("Process Documents"):
#         st.session_state.uploaded_files = uploaded_files
#         with st.spinner("Processing documents..."):
#             try:
#                 llm, retriever, doc_count = load_rag_system(uploaded_files)
#                 st.success(f"Processed {len(uploaded_files)} files. Total documents: {doc_count}")
#             except Exception as e:
#                 st.error(f"Error processing documents: {str(e)}")
    
#     if st.button("Clear History"):
#         st.session_state.history = []
#         st.rerun()
    
#     st.markdown("---")
#     st.markdown("**Smart Features:**")
#     st.markdown("- Automatic document ingestion")
#     st.markdown("- Context-aware query understanding")
#     st.markdown("- Dynamic knowledge integration")

# # Display chat history
# for msg in st.session_state.history:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])
#         if msg.get("sources"):
#             with st.expander("Sources"):
#                 for src in msg["sources"]:
#                     st.markdown(f"📄 {src}")

# # User input
# if prompt := st.chat_input("Ask about NovaMind AI:"):
#     st.session_state.history.append({"role": "user", "content": prompt})
    
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.spinner("Analyzing..."):
#         try:
#             llm, retriever, doc_count = load_rag_system(st.session_state.uploaded_files)
            
#             # Generate context-aware query
#             query_prompt = f"""Based on the conversation history and current question, generate an optimized search query.

# History:
# {" ".join([msg['content'] for msg in st.session_state.history[-4:]])}

# Question: {prompt}

# Generate a comprehensive search query that considers potential document context:"""
            
#             optimized_query = llm.invoke(query_prompt).content

#             # Retrieve documents
#             retrieved_docs = retriever.get_relevant_documents(optimized_query)
#             context = "\n".join({d.page_content for d in retrieved_docs}) if retrieved_docs else ""
#             if not context.strip():
#                 st.info("No relevant context found in documents. Providing a general response.")


#             print("this is context -> " + context)

#             # Prepare LLM prompt
#             llm_prompt = f"""You are a helpful AI assistant. Use the following context if relevant, otherwise use your general knowledge.

# Context: {context or 'No specific context available'}

# Conversation History:
# {" ".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.history[-4:]])}

# Question: {prompt}

# Provide a concise, accurate response:"""
            
#             response = llm.invoke(llm_prompt)
#             answer = response.content

#             # Add to history
#             st.session_state.history.append({
#                 "role": "assistant",
#                 "content": answer,
#                 "sources": list({d.metadata.get('source', 'Unknown')[:50] for d in retrieved_docs[:3]}) if context else []
#             })

#             # Display response
#             with st.chat_message("assistant"):
#                 st.markdown(answer)
#                 if context:
#                     with st.expander("Relevant Sources"):
#                         for doc in retrieved_docs[:3]:
#                             source = doc.metadata.get('source', 'Unknown')
#                             st.markdown(f"🔗 {source.split('/')[-1][:40]} - {doc.page_content[:100]}...")

#         except Exception as e:
#             st.error(f"Error: {str(e)}")
#             st.session_state.history.append({
#                 "role": "assistant",
#                 "content": f"Error processing request: {str(e)}"
#             })
















# SMART RAG TAKE #3 WITH MORE IMPROVEMENTS OF DOCUMENTS AND FALLBACK NOTIFICATION
# TELLS YOU THAT THIS ANSWER IS THAT GENERAL NKOWLEGE OR CONTEXT.


# import streamlit as st
# import os
# import chromadb
# from langchain_chroma import Chroma
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.schema.document import Document
# from embedding_client import EmbeddingClient
# from typing import List, Optional
# from langchain_community.document_loaders import PyPDFLoader, TextLoader
# import tempfile
# import re


# # Configuration
# GOOGLE_API_KEY = "api"
# os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
# PERSIST_DIR = "./chroma_store"
# COLLECTION_NAME = "buisness_story"
# SEARCH_KWARGS = {"k": 6, "score_threshold": 0.4, "fetch_k": 20}


# def clean_text(text: str) -> str:
#     # Basic spacing fix: add space between lowercase-uppercase (e.g., homeHe → home He)
#     text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
#     # Add space between a word and a capitalized word following it
#     text = re.sub(r'([a-zA-Z])([A-Z][a-z])', r'\1 \2', text)

#     # Replace multiple spaces with one
#     text = re.sub(r'\s{2,}', ' ', text)

#     return text.strip()

# def process_uploaded_files(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> List[Document]:
#     """Process uploaded files into LangChain Documents"""
#     docs = []
#     for file in uploaded_files:
#         with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#             temp_file.write(file.getvalue())
#             temp_path = temp_file.name
        
#         try:
#             if file.name.endswith(".pdf"):
#                 loader = PyPDFLoader(temp_path)
#                 # docs.extend(loader.load())
#                 docs.extend([
#                     Document(page_content=clean_text(doc.page_content), metadata=doc.metadata)
#                     for doc in loader.load()
#                 ])
#             elif file.name.endswith((".txt", ".md")):
#                 loader = TextLoader(temp_path)
#                 # docs.extend(loader.load())
#                 docs.extend([
#                     Document(page_content=clean_text(doc.page_content), metadata=doc.metadata)
#                     for doc in loader.load()
#                 ])
#             else:
#                 st.warning(f"Unsupported file type: {file.name}")
#         except Exception as e:
#             st.error(f"Error processing {file.name}: {str(e)}")
#         finally:
#             os.remove(temp_path)
#     return docs

# @st.cache_resource
# def load_rag_system(uploaded_files: Optional[List] = None):
#     embedding_model = EmbeddingClient(api_url="http://127.0.0.1:2000")
#     chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    
#     # Process uploaded files if any
#     new_docs = process_uploaded_files(uploaded_files) if uploaded_files else []

#     try:
#         collection = chroma_client.get_collection(COLLECTION_NAME)
#         existing_count = collection.count()
        
#         # Add new documents if any
#         if new_docs:
#             ids = [str(existing_count + i) for i in range(len(new_docs))]
#             collection.add(
#                 documents=[doc.page_content for doc in new_docs],
#                 ids=ids
#             )
            
#         vectordb = Chroma(
#             client=chroma_client,
#             collection_name=COLLECTION_NAME,
#             embedding_function=embedding_model,
#             collection_metadata={"hnsw:space": "cosine"}
#         )
#     except:
#         # Create new collection with initial documents
#         raw_docs = new_docs if new_docs else [Document(page_content="NovaMind AI default information")]
#         vectordb = Chroma.from_documents(
#             documents=raw_docs,
#             embedding=embedding_model,
#             client=chroma_client,
#             collection_name=COLLECTION_NAME,
#             collection_metadata={"hnsw:space": "cosine"}
#         )

#     retriever = vectordb.as_retriever(
#         search_type="mmr",
#         search_kwargs=SEARCH_KWARGS
#     )

#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.0-flash",
#         temperature=0.3,
#         google_api_key=GOOGLE_API_KEY
#     )

#     return llm, retriever, vectordb._collection.count()

# # Initialize app
# st.set_page_config(page_title="NovaMind AI Assistant", page_icon="🤖")
# st.title("NovaMind Smart Conversational Agent")

# # Session state management
# if "history" not in st.session_state:
#     st.session_state.history = []
# if "uploaded_files" not in st.session_state:
#     st.session_state.uploaded_files = []

# # Sidebar controls
# with st.sidebar:
#     st.header("Document Management")
#     uploaded_files = st.file_uploader(
#         "Upload knowledge documents",
#         type=["pdf", "txt", "md"],
#         accept_multiple_files=True
#     )
    
#     if uploaded_files and st.button("Process Documents"):
#         st.session_state.uploaded_files = uploaded_files
#         with st.spinner("Processing documents..."):
#             try:
#                 llm, retriever, doc_count = load_rag_system(uploaded_files)
#                 st.success(f"Processed {len(uploaded_files)} files. Total documents: {doc_count}")
#             except Exception as e:
#                 st.error(f"Error processing documents: {str(e)}")
    
#     if st.button("Clear History"):
#         st.session_state.history = []
#         st.rerun()
    
#     st.markdown("---")
#     st.markdown("**Smart Features:**")
#     st.markdown("- Automatic document ingestion")
#     st.markdown("- Context-aware query understanding")
#     st.markdown("- Dynamic knowledge integration")

# # Display chat history
# for msg in st.session_state.history:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])
#         if msg.get("sources"):
#             with st.expander("Sources"):
#                 for src in msg["sources"]:
#                     st.markdown(f"📄 {src}")
#         if msg.get("source_type"):
#             st.caption(f"*{msg['source_type']}*")

# # User input
# if prompt := st.chat_input("Ask about NovaMind AI:"):
#     st.session_state.history.append({"role": "user", "content": prompt})
    
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.spinner("Analyzing..."):
#         try:
#             llm, retriever, doc_count = load_rag_system(st.session_state.uploaded_files)
            
#             # Generate context-aware query
#             query_prompt = f"""Based on the conversation history and current question, generate an optimized search query.

# History:
# {" ".join([msg['content'] for msg in st.session_state.history[-4:]])}

# Question: {prompt}

# Generate a comprehensive search query that considers potential document context:"""
            
#             optimized_query = llm.invoke(query_prompt).content

#             # Retrieve documents
#             retrieved_docs = retriever.get_relevant_documents(optimized_query)
#             context = "\n".join({d.page_content for d in retrieved_docs}) if retrieved_docs else ""
            
#             # Determine if we have relevant context to use
#             has_relevant_context = bool(context.strip() and len(context) > 50)
            
#             # Print debug info
#             if has_relevant_context:
#                 print("Answering from uploaded documents")
#                 source_type = "Response based on uploaded documents"
#             else:
#                 print("Answering from general LLM knowledge")
#                 source_type = "Response based on general knowledge"
                
#             print("Context length:", len(context))

#             # Prepare LLM prompt
#             llm_prompt = f"""You are a helpful AI assistant. Use the following context if relevant, otherwise use your general knowledge.

# Context: {context or 'No specific context available'}

# Conversation History:
# {" ".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.history[-4:]])}

# Question: {prompt}

# Provide a concise, accurate response:"""
            
#             response = llm.invoke(llm_prompt)
#             answer = response.content

#             # Add to history with source type indication
#             st.session_state.history.append({
#                 "role": "assistant",
#                 "content": answer,
#                 "sources": list({d.metadata.get('source', 'Unknown')[:50] for d in retrieved_docs[:3]}) if has_relevant_context else [],
#                 "source_type": source_type
#             })

#             # Display response
#             with st.chat_message("assistant"):
#                 st.markdown(answer)
                
#                 # Show the source indicator
#                 st.caption(f"*{source_type}*")
                
#                 if has_relevant_context:
#                     with st.expander("Relevant Sources"):
#                         for doc in retrieved_docs[:3]:
#                             source = doc.metadata.get('source', 'Unknown')
#                             st.markdown(f"🔗 {source.split('/')[-1][:40]} - {doc.page_content[:100]}...")

#         except Exception as e:
#             st.error(f"Error: {str(e)}")
#             st.session_state.history.append({
#                 "role": "assistant",
#                 "content": f"Error processing request: {str(e)}"
#             })










#TESTING WITH SOURCE 







import streamlit as st
import os
import chromadb
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.document import Document
from embedding_client import EmbeddingClient
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import tempfile
import re


# Configuration
GOOGLE_API_KEY = "api"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
PERSIST_DIR = "./chroma_store"
COLLECTION_NAME = "my_story"
SEARCH_KWARGS = {"k": 6, "score_threshold": 0.4, "fetch_k": 20}


def clean_text(text: str) -> str:
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])([A-Z][a-z])', r'\1 \2', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def process_uploaded_files(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> List[Document]:
    """Process uploaded files into LangChain Documents with cleaned content and file metadata"""
    docs = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.getvalue())
            temp_path = temp_file.name

        try:
            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(temp_path)
                loaded_docs = loader.load()
            elif file.name.endswith((".txt", ".md")):
                loader = TextLoader(temp_path)
                loaded_docs = loader.load()
            else:
                st.warning(f"Unsupported file type: {file.name}")
                continue

            docs.extend([
                Document(
                    page_content=clean_text(doc.page_content),
                    metadata={**doc.metadata, "source": file.name}
                ) for doc in loaded_docs
            ])

        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
        finally:
            os.remove(temp_path)

    return docs


@st.cache_resource
def load_rag_system(uploaded_files: Optional[List] = None):
    embedding_model = EmbeddingClient(api_url="http://127.0.0.1:2000")
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)

    new_docs = process_uploaded_files(uploaded_files) if uploaded_files else []

    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
        existing_count = collection.count()

        if new_docs:
            ids = [str(existing_count + i) for i in range(len(new_docs))]
            collection.add(
                documents=[doc.page_content for doc in new_docs],
                ids=ids,
                metadatas=[doc.metadata for doc in new_docs]  # ✅ CRUCIAL FIX
            )

        vectordb = Chroma(
            client=chroma_client,
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"}
        )

    except:
        raw_docs = new_docs if new_docs else [Document(page_content="NovaMind SRAG default info")]
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

    return llm, retriever, vectordb._collection.count()


# --- Streamlit App ---
st.set_page_config(page_title="NovaMind SRAG Assistant", page_icon="🤖")
st.title("NovaMind SMART RAG")

if "history" not in st.session_state:
    st.session_state.history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Sidebar
with st.sidebar:
    st.header("📄 Document Management")
    uploaded_files = st.file_uploader("Upload documents", type=["pdf", "txt", "md"], accept_multiple_files=True)

    if uploaded_files and st.button("Process Documents"):
        st.session_state.uploaded_files = uploaded_files
        with st.spinner("Processing documents..."):
            try:
                llm, retriever, doc_count = load_rag_system(uploaded_files)
                st.success(f"Uploaded {len(uploaded_files)} files. Total documents: {doc_count}")
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")

    if st.button("🧹 Clear Chat History"):
        st.session_state.history = []
        st.rerun()

    if st.button("🧨 Reset Embedding Store"):
        chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
        
        try:
            collection_names = [col.name for col in chroma_client.list_collections()]
            if COLLECTION_NAME in collection_names:
                chroma_client.delete_collection(name=COLLECTION_NAME)
                st.success("Chroma store reset. Please upload files again.")
            else:
                st.info(f"Collection `{COLLECTION_NAME}` does not exist. Nothing to delete.")
        except Exception as e:
            st.error(f"Error resetting embedding store: {e}")

        st.session_state.history = []
        st.rerun()


    st.markdown("---")
    st.markdown("**Smart Features:**")
    st.markdown("- RAG with Gemini")
    st.markdown("- File-aware answers with sources")
    st.markdown("- Custom local embedding backend")


# Chat UI
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Relevant Sources"):
                for src in msg["sources"]:
                    st.markdown(f"📄 **{src}**")


if prompt := st.chat_input("Ask your question..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        try:
            llm, retriever, doc_count = load_rag_system(st.session_state.uploaded_files)

            query_prompt = f"""Based on this chat and the user's new question, generate an improved search query.

Chat:
{" ".join([msg['content'] for msg in st.session_state.history[-4:]])}

Question: {prompt}

Optimized search query:"""
            optimized_query = llm.invoke(query_prompt).content

            retrieved_docs = retriever.get_relevant_documents(optimized_query)
            context = "\n".join({doc.page_content for doc in retrieved_docs}) if retrieved_docs else ""

            final_prompt = f"""You are a helpful AI assistant. Use the following context (if available) to answer.

Context: {context or "No specific context found."}

Chat History:
{" ".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.history[-4:]])}

User Question: {prompt}

Answer concisely and clearly:"""

            response = llm.invoke(final_prompt)
            answer = response.content

            sources = list({doc.metadata.get('source', 'Unknown')[:50] for doc in retrieved_docs[:3]}) if context else []

            st.session_state.history.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

            with st.chat_message("assistant"):
                st.markdown(answer)
                if sources:
                    with st.expander("Relevant Sources"):
                        for doc in retrieved_docs[:3]:
                            source = doc.metadata.get("source", "Unknown")
                            snippet = doc.page_content[:100].replace("\n", " ")
                            st.markdown(f"🔗 **{source}** — {snippet}...")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.session_state.history.append({
                "role": "assistant",
                "content": f"Error occurred: {str(e)}"
            })
