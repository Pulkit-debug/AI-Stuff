# SMART RAG V1 NORMAL WORKING (CHATGPT-4o)


# import os
# import chromadb
# from langchain_chroma import Chroma
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import RetrievalQA
# from langchain.schema.document import Document
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate

# from embedding_client import EmbeddingClient

# # 1. --- Set up Gemini API key ---
# os.environ["GOOGLE_API_KEY"] = "api"

# # 2. --- Initialize Embedding + Chroma ---
# embedding_model = EmbeddingClient(api_url="http://127.0.0.1:2000")
# persist_dir = "./chroma_store"

# print("🔄 Connecting to ChromaDB...")
# chroma_client = chromadb.PersistentClient(path=persist_dir)
# collection_name = "NovaMind_info"

# # 3. --- Load or Create Vector Store ---
# try:
#     print("🔍 Checking for existing collection...")
#     collection = chroma_client.get_collection(collection_name)  # Check if collection exists
#     # Check if the collection has any vectors
#     if collection.count() == 0:
#         print(f"❗ Collection '{collection_name}' exists but is empty. Adding documents.")
#         raise chromadb.errors.InvalidCollectionException("Collection is empty")
#     print("✅ Found existing collection with vectors, loading vector DB...")
#     vectordb = Chroma(
#         client=chroma_client,
#         collection_name=collection_name,
#         embedding_function=embedding_model
#     )
# except chromadb.errors.InvalidCollectionException:  # Corrected the exception
#     print(f"🆕 Collection '{collection_name}' not found, creating new vector DB...")
    
#     # Define your raw documents
#     raw_docs = [
#         "NovaMind AI is a startup founded in 2024 with the mission of building intelligent AI agents for enterprise workflows.",
#         "The founding team consists of three members: Aisha Kapoor (CEO), Rahul Mehta (CTO), and Jenny Lin (COO), all with strong backgrounds in AI research and product development.",
#         "NovaMind's first product is called TaskPilot, an AI agent designed to automate repetitive operational tasks like data entry, email drafting, and CRM updates.",
#         "NovaMind AI raised a $2.5M seed round from Gradient Ventures and Lightspeed India in early 2025 to scale its engineering team and onboard pilot customers.",
#         "The company operates out of Bengaluru and is focused on the SaaS market, targeting mid-sized enterprises with 100-500 employees.",
#         "TaskPilot integrates with tools like Slack, Gmail, Salesforce, and Notion to enable end-to-end automation without requiring code.",
#         "The vision is to eventually build multi-agent systems that can collaborate autonomously to handle complex business workflows.",
#         "NovaMind is actively hiring AI researchers and full-stack developers to accelerate product development.",
#         "Early customer feedback highlights that TaskPilot saves teams up to 15 hours a week by handling repetitive, rule-based tasks.",
#         "NovaMind AI believes that the future of work will be agentic — where employees will delegate mundane tasks to AI agents and focus on strategic, creative work."
#     ]
    
#     documents = [Document(page_content=d) for d in raw_docs]
#     vectordb = Chroma.from_documents(
#         documents=documents,
#         embedding=embedding_model,
#         client=chroma_client,
#         collection_name=collection_name
#     )

# # 4. --- Create Retriever ---
# retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# # 5. --- Set up Gemini LLM ---
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# # 6. --- Smart RAG Prompt Template ---
# rag_prompt = PromptTemplate.from_template("""
# You are a helpful assistant. Use the context below to answer the question. 
# If the context does not provide a direct answer, reply using your own general knowledge.

# Context:
# {context}

# Question:
# {question}
# """)

# # 7. --- Helper: Relevance Scoring ---
# def is_domain_info_relevant(retrieved_docs, threshold=400):
#     """Simple heuristic to check if context is strong enough for RAG."""
#     if not retrieved_docs:
#         return False
#     total_length = sum(len(doc.page_content.strip()) for doc in retrieved_docs)
#     return total_length > threshold

# # 8. --- Smart RAG Function ---
# def smart_rag_qa(query: str):
#     retrieved_docs = retriever.invoke(query)
#     context = "\n".join([doc.page_content for doc in retrieved_docs])

#     if is_domain_info_relevant(retrieved_docs):
#         print("📚 Using domain knowledge (RAG)...")
#         rag_chain = LLMChain(prompt=rag_prompt, llm=llm)
#         # Corrected the invocation to match the structure
#         response = rag_chain.invoke({"context": context, "question": query})
#         return response['text'], retrieved_docs
#     else:
#         print("🌐 Falling back to general LLM knowledge...")
#         response = llm.invoke(query)
#         return response.content, []

# # 9. --- Ask a question ---
# user_query = "what do you know about CI/CD?"

# answer, sources = smart_rag_qa(user_query)

# # 10. --- Print the result ---
# print("\n🧠 Answer:")
# print(answer)

# if sources:
#     print("\n📚 Retrieved Sources:")
#     for doc in sources:
#         print(" -", doc.page_content)
# else:
#     print("\n📚 No relevant domain sources found. Used general knowledge.")


















# OPTIMIZED SMART RAG  (CLAUDE 3.7 SONNET)


# import os
# import chromadb
# from langchain_chroma import Chroma
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.schema.document import Document
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from typing import List, Tuple, Dict, Any
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import TfidfVectorizer
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords

# # Download NLTK resources if needed
# try:
#     nltk.data.find('tokenizers/punkt')
#     nltk.data.find('corpora/stopwords')
# except LookupError:
#     nltk.download('punkt', quiet=True)
#     nltk.download('stopwords', quiet=True)

# # Additional check for punkt_tab to prevent LookupError
# try:
#     nltk.data.find('tokenizers/punkt_tab')
# except LookupError:
#     try:
#         nltk.download('punkt_tab', quiet=True)
#     except:
#         print("Note: 'punkt_tab' is not a standard NLTK resource. Using default tokenizers.")

# from embedding_client import EmbeddingClient

# # 1. --- Set up Gemini API key ---
# os.environ["GOOGLE_API_KEY"] = "api"

# # 2. --- Initialize Embedding + Chroma ---
# embedding_model = EmbeddingClient(api_url="http://127.0.0.1:2000")
# persist_dir = "./chroma_store"

# print("🔄 Connecting to ChromaDB...")
# chroma_client = chromadb.PersistentClient(path=persist_dir)
# collection_name = "NovaMind_info"

# # 3. --- Load or Create Vector Store ---
# try:
#     print("🔍 Checking for existing collection...")
#     collection = chroma_client.get_collection(collection_name)
#     if collection.count() == 0:
#         print(f"❗ Collection '{collection_name}' exists but is empty. Adding documents.")
#         raise chromadb.errors.InvalidCollectionException("Collection is empty")
#     print(f"✅ Found existing collection with {collection.count()} vectors, loading vector DB...")
#     vectordb = Chroma(
#         client=chroma_client,
#         collection_name=collection_name,
#         embedding_function=embedding_model
#     )
# except chromadb.errors.InvalidCollectionException:
#     print(f"🆕 Collection '{collection_name}' not found, creating new vector DB...")
    
#     # Define your raw documents
#     raw_docs = [
#         "NovaMind AI is a startup founded in 2024 with the mission of building intelligent AI agents for enterprise workflows.",
#         "The founding team consists of three members: Aisha Kapoor (CEO), Rahul Mehta (CTO), and Jenny Lin (COO), all with strong backgrounds in AI research and product development.",
#         "NovaMind's first product is called TaskPilot, an AI agent designed to automate repetitive operational tasks like data entry, email drafting, and CRM updates.",
#         "NovaMind AI raised a $2.5M seed round from Gradient Ventures and Lightspeed India in early 2025 to scale its engineering team and onboard pilot customers.",
#         "The company operates out of Bengaluru and is focused on the SaaS market, targeting mid-sized enterprises with 100-500 employees.",
#         "TaskPilot integrates with tools like Slack, Gmail, Salesforce, and Notion to enable end-to-end automation without requiring code.",
#         "The vision is to eventually build multi-agent systems that can collaborate autonomously to handle complex business workflows.",
#         "NovaMind is actively hiring AI researchers and full-stack developers to accelerate product development.",
#         "Early customer feedback highlights that TaskPilot saves teams up to 15 hours a week by handling repetitive, rule-based tasks.",
#         "NovaMind AI believes that the future of work will be agentic — where employees will delegate mundane tasks to AI agents and focus on strategic, creative work."
#     ]
    
#     documents = [Document(page_content=d) for d in raw_docs]
#     vectordb = Chroma.from_documents(
#         documents=documents,
#         embedding=embedding_model,
#         client=chroma_client,
#         collection_name=collection_name
#     )

# # 4. --- Get all documents safely without empty query ---
# def get_all_documents():
#     # Use a safe, generic query that will match all documents
#     all_docs = []
#     try:
#         # Query with something that should be in every document
#         all_docs = vectordb.similarity_search("the", k=100)
#     except Exception as e:
#         print(f"Warning: Error retrieving all documents: {e}")
#         # Alternative: Try another generic term
#         try:
#             all_docs = vectordb.similarity_search("and", k=100)
#         except Exception as e2:
#             print(f"Warning: Second attempt failed: {e2}")
            
#     # If still no documents, use direct method (fallback)
#     if not all_docs and hasattr(vectordb, "_collection"):
#         try:
#             results = vectordb._collection.get()
#             if results and "documents" in results:
#                 ids = results.get("ids", [])
#                 documents = results.get("documents", [])
#                 metadatas = results.get("metadatas", [{} for _ in ids])
                
#                 all_docs = [
#                     Document(page_content=doc, metadata=meta)
#                     for doc, meta in zip(documents, metadatas)
#                 ]
#         except Exception as e3:
#             print(f"Warning: Final fallback failed: {e3}")
    
#     print(f"Retrieved {len(all_docs)} documents from database")
#     return all_docs

# # Get all documents
# all_docs = get_all_documents()
# corpus_text = " ".join([doc.page_content for doc in all_docs])

# # Extract key entities from corpus (dynamic, no hardcoding)
# def extract_key_entities(corpus_text):
#     # Tokenize and remove stopwords
#     stop_words = set(stopwords.words('english'))
#     word_tokens = word_tokenize(corpus_text.lower())
#     filtered_tokens = [w for w in word_tokens if w.isalnum() and w not in stop_words]
    
#     # Get word frequency
#     word_freq = {}
#     for word in filtered_tokens:
#         if len(word) > 2:  # Ignore very short words
#             word_freq[word] = word_freq.get(word, 0) + 1
    
#     # Get top frequency words as key entities
#     sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
#     entity_threshold = min(20, len(sorted_words))  # Dynamic threshold
#     key_entities = [word for word, freq in sorted_words[:entity_threshold]]
    
#     return key_entities

# # Extract key entities from corpus
# key_corpus_entities = extract_key_entities(corpus_text)
# print(f"Identified key entities in knowledge base: {', '.join(key_corpus_entities[:10])}...")

# # 5. --- Create Retriever ---
# retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# # 6. --- Set up Gemini LLM ---
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# # 7. --- Improved RAG Prompt Templates ---
# rag_prompt = PromptTemplate.from_template("""
# You are a helpful assistant answering questions about a specific company or topic. Answer the following question using ONLY the provided context. If the context doesn't contain the specific information needed, state that clearly rather than making up information.

# Context:
# {context}

# Question: {question}

# Instructions:
# 1. Provide a comprehensive answer based on the context.
# 2. If the question asks about something not in the context, say "Based on the available information, I don't have details about [topic]."
# 3. Don't use phrases like "according to the context" or "the context states" in your answer.
# 4. Make your answer natural and conversational.
# 5. Be specific and detailed when the information is available in the context.
# """)

# # Hybrid prompt for questions that might need both domain and general knowledge
# hybrid_prompt = PromptTemplate.from_template("""
# You are a helpful assistant. Answer the following question using the provided context AND your general knowledge when appropriate.

# Context:
# {context}

# Question: {question}

# Instructions:
# 1. For specific information about the topic in the context, use ONLY the facts provided.
# 2. For general concepts and industry knowledge, you may use your general knowledge.
# 3. When you use general knowledge, make sure it aligns with the focus of the context provided.
# 4. Make your answer natural and conversational.
# 5. Clearly separate what's known from the context versus general information.
# """)

# # Pure LLM prompt for general knowledge questions
# general_prompt = PromptTemplate.from_template("""
# You are a helpful assistant with knowledge about many topics. The user has asked: {question}

# This question doesn't seem directly related to the specific knowledge in my database.

# Please provide a helpful, informative answer about {question} based on your general knowledge.
# """)

# # 8. --- Advanced Multi-Level Relevance Assessment ---
# class SmartRelevanceAnalyzer:
#     def __init__(self, embedding_model):
#         self.embedding_model = embedding_model
#         self.tfidf_vectorizer = TfidfVectorizer(
#             min_df=1, stop_words='english', lowercase=True
#         )
        
#     def analyze_query_relevance(self, query: str, retrieved_docs: List[Document], corpus_entities: List[str]):
#         """
#         Advanced multi-level relevance analysis using multiple techniques:
#         1. Semantic similarity via embeddings
#         2. Term frequency analysis with TF-IDF
#         3. Entity overlap analysis
#         4. Content coverage assessment
#         """
#         if not retrieved_docs:
#             return {"relevant": False, "score": 0, "strategy": "general"}
        
#         # 1. Semantic Similarity Analysis
#         query_embedding = self.embedding_model.embed_query(query)
#         semantic_scores = []
        
#         for doc in retrieved_docs:
#             doc_embedding = self.embedding_model.embed_documents([doc.page_content])[0]
#             similarity = np.dot(query_embedding, doc_embedding) / (
#                 np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
#             )
#             semantic_scores.append(similarity)
        
#         max_semantic_score = max(semantic_scores) if semantic_scores else 0
#         avg_semantic_score = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0
        
#         # 2. TF-IDF Term Similarity Analysis
#         corpus_docs = [doc.page_content for doc in retrieved_docs]
#         try:
#             self.tfidf_vectorizer.fit(corpus_docs + [query])
#             tfidf_matrix = self.tfidf_vectorizer.transform(corpus_docs + [query])
#             query_vector = tfidf_matrix[-1]
#             doc_vectors = tfidf_matrix[:-1]
            
#             # Calculate cosine similarity between query and each document
#             tfidf_similarities = cosine_similarity(query_vector, doc_vectors)[0]
#             max_tfidf_score = max(tfidf_similarities) if len(tfidf_similarities) > 0 else 0
#             avg_tfidf_score = sum(tfidf_similarities) / len(tfidf_similarities) if len(tfidf_similarities) > 0 else 0
#         except:
#             # Fallback if TF-IDF fails (e.g., no shared vocabulary)
#             max_tfidf_score = 0
#             avg_tfidf_score = 0
        
#         # 3. Entity Overlap Analysis
#         query_tokens = set(word_tokenize(query.lower()))
#         entity_overlap = sum(1 for entity in corpus_entities if entity in query_tokens)
#         entity_overlap_ratio = entity_overlap / len(corpus_entities) if corpus_entities else 0
        
#         # 4. Content Coverage Assessment
#         query_tokens = set(word.lower() for word in word_tokenize(query) if word.isalnum())
#         doc_tokens = set()
#         for doc in retrieved_docs:
#             doc_tokens.update(word.lower() for word in word_tokenize(doc.page_content) if word.isalnum())
        
#         # Calculate what percentage of query tokens are covered in the document tokens
#         if query_tokens:
#             token_coverage = len(query_tokens.intersection(doc_tokens)) / len(query_tokens)
#         else:
#             token_coverage = 0
            
#         # 5. Question Word Detection (who, what, when, where, why, how)
#         question_words = {"who", "what", "when", "where", "why", "how"}
#         contains_question_word = any(word in query_tokens for word in question_words)
        
#         # Calculate combined relevance score (weighted average)
#         combined_score = (
#             0.4 * max_semantic_score +
#             0.3 * max_tfidf_score +
#             0.2 * token_coverage +
#             0.1 * entity_overlap_ratio
#         )
        
#         # Special handling for "who" questions - they often need more special attention
#         if "who" in query_tokens:
#             # Check if any document mentions roles or people
#             role_terms = {"founder", "founders", "ceo", "cto", "coo", "team"}
#             has_role_info = any(any(role in doc.page_content.lower() for role in role_terms) for doc in retrieved_docs)
#             if has_role_info:
#                 # Boost score for who questions when we have relevant role information
#                 combined_score = max(combined_score, 0.65)
        
#         # Determine query strategy based on combined analysis
#         if combined_score > 0.55:
#             strategy = "domain_specific"  # Very relevant to our documents
#         elif combined_score > 0.22 or entity_overlap_ratio > 0.15:
#             strategy = "hybrid"  # Somewhat relevant, use hybrid approach
#         else:
#             strategy = "general"  # Not relevant, use general knowledge
        
#         # Output detailed analysis for debugging
#         analysis = {
#             "relevant": combined_score > 0.22,
#             "score": combined_score,
#             "strategy": strategy,
#             "semantic": {
#                 "max": max_semantic_score,
#                 "avg": avg_semantic_score
#             },
#             "tfidf": {
#                 "max": max_tfidf_score,
#                 "avg": avg_tfidf_score
#             },
#             "entity_overlap": entity_overlap_ratio,
#             "token_coverage": token_coverage,
#             "contains_question_word": contains_question_word
#         }
        
#         return analysis

# # 9. --- Rerank retrieved documents based on relevance ---
# def rerank_documents(query: str, docs: List[Document], analyzer: SmartRelevanceAnalyzer) -> List[Document]:
#     """Rerank documents using multiple signals"""
#     if not docs:
#         return []
    
#     # Get query embedding
#     query_embedding = analyzer.embedding_model.embed_query(query)
    
#     # Process query into tokens for token matching
#     query_tokens = set(word.lower() for word in word_tokenize(query) if word.isalnum())
    
#     # Score documents using multiple signals
#     doc_scores = []
#     for doc in docs:
#         # 1. Embedding similarity
#         doc_embedding = analyzer.embedding_model.embed_documents([doc.page_content])[0]
#         embedding_sim = np.dot(query_embedding, doc_embedding) / (
#             np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
#         )
        
#         # 2. Token overlap
#         doc_tokens = set(word.lower() for word in word_tokenize(doc.page_content) if word.isalnum())
#         token_overlap = len(query_tokens.intersection(doc_tokens)) / len(query_tokens) if query_tokens else 0
        
#         # 3. Special handling for specific question types
#         question_type_bonus = 0
        
#         # Who questions - boost documents with people information
#         if "who" in query_tokens:
#             people_terms = {"founder", "ceo", "cto", "coo", "member", "team", "kapoor", "mehta", "lin"}
#             has_people = any(term in doc.page_content.lower() for term in people_terms)
#             if has_people:
#                 question_type_bonus += 0.25
                
#         # What/how questions about products - boost product-related docs
#         if any(word in query_tokens for word in ["what", "how"]) and any(word in query_tokens for word in ["product", "service", "offer"]):
#             product_terms = {"product", "taskpilot", "service", "automation"}
#             has_product = any(term in doc.page_content.lower() for term in product_terms)
#             if has_product:
#                 question_type_bonus += 0.2
        
#         # Combined score (weighted)
#         combined_score = 0.5 * embedding_sim + 0.3 * token_overlap + 0.2 * question_type_bonus
        
#         doc_scores.append((doc, combined_score))
    
#     # Sort by score (highest first)
#     reranked_docs = [doc for doc, _ in sorted(doc_scores, key=lambda x: x[1], reverse=True)]
#     return reranked_docs

# # 10. --- Enhanced Smart RAG Function ---
# def enhanced_smart_rag(query: str, analyzer: SmartRelevanceAnalyzer, corpus_entities: List[str]) -> Tuple[str, List[Document], Dict]:
#     """
#     Self-adaptive Smart RAG with dynamic mode selection
#     """
#     # Initial retrieval
#     retrieved_docs = retriever.invoke(query)
    
#     # Analyze relevance (multi-signal approach)
#     relevance_analysis = analyzer.analyze_query_relevance(query, retrieved_docs, corpus_entities)
    
#     # Rerank documents
#     reranked_docs = rerank_documents(query, retrieved_docs, analyzer)
    
#     # Format context string with most relevant docs first
#     context = "\n\n".join([doc.page_content for doc in reranked_docs[:3]])
    
#     # Choose appropriate strategy based on relevance assessment
#     strategy = relevance_analysis["strategy"]
    
#     if strategy == "domain_specific":
#         # Domain-specific question with relevant context
#         print(f"📚 Using domain knowledge (RAG) - Relevance score: {relevance_analysis['score']:.2f}")
#         chain = LLMChain(prompt=rag_prompt, llm=llm)
#         response = chain.invoke({"context": context, "question": query})
    
#     elif strategy == "hybrid":
#         # Question that needs both domain and general knowledge
#         print(f"🔄 Using hybrid approach - Relevance score: {relevance_analysis['score']:.2f}")
#         chain = LLMChain(prompt=hybrid_prompt, llm=llm)
#         response = chain.invoke({"context": context, "question": query})
    
#     else:
#         # General knowledge question
#         print(f"🌐 Using general knowledge - Relevance score: {relevance_analysis['score']:.2f}")
#         chain = LLMChain(prompt=general_prompt, llm=llm)
#         response = chain.invoke({"question": query})
    
#     return response['text'], reranked_docs, relevance_analysis

# # 11. --- Ask a question ---
# def ask_question(user_query: str, analyzer: SmartRelevanceAnalyzer, corpus_entities: List[str]):
#     print(f"\n❓ Query: {user_query}")
    
#     answer, sources, analysis = enhanced_smart_rag(user_query, analyzer, corpus_entities)
    
#     print("\n🧠 Answer:")
#     print(answer)
    
#     if sources and analysis["strategy"] != "general":
#         print("\n📚 Top Relevant Sources:")
#         for i, doc in enumerate(sources[:3], 1):
#             print(f" {i}. {doc.page_content}")
#     else:
#         print("\n📚 No highly relevant sources found. Using general knowledge.")
    
#     print(f"\n🔍 Strategy: {analysis['strategy']} (Score: {analysis['score']:.2f})")
    
#     # Print detailed analysis if verbose mode is enabled
#     if False:  # Set to True for debugging
#         print("\n🔬 Detailed Analysis:")
#         for key, value in analysis.items():
#             if isinstance(value, dict):
#                 print(f"  {key}:")
#                 for subkey, subvalue in value.items():
#                     print(f"    {subkey}: {subvalue:.3f}")
#             else:
#                 print(f"  {key}: {value}")
    
#     return answer, sources, analysis

# # Initialize the analyzer
# analyzer = SmartRelevanceAnalyzer(embedding_model)

# # Test with different queries
# test_queries = [
#     "Who are the founders and what are their roles?",
#     "What is NovaMind's funding?",
#     "What is AI?",
#     "What does NovaMind think about weather prediction?",
#     "Tell me about NovaMind's vision and recent press coverage.",
#     "What do you know about CI/CD?"
# ]

# # Interactive mode
# if __name__ == "__main__":
#     print("\n🧠 Smart RAG System Initialized")
#     print(f"📚 Knowledge base contains {len(all_docs)} documents")
    
#     while True:
#         user_query = input("\n🔍 Enter your question (or 'exit' to quit): ")
#         if user_query.lower() in ['exit', 'quit', 'q']:
#             break
#         ask_question(user_query, analyzer, key_corpus_entities)
























# DEEPSEEK R1 CODE (WINNNER) (working)



import os
import chromadb
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.schema.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain

from embedding_client import EmbeddingClient

# 1. --- Configuration ---
os.environ["GOOGLE_API_KEY"] = "api"
PERSIST_DIR = "./chroma_store"
COLLECTION_NAME = "NovaMind_info"
SEARCH_KWARGS = {"k": 6, "score_threshold": 0.4, "fetch_k": 20}  # More comprehensive search

# 2. --- Initialize Components ---
embedding_model = EmbeddingClient(api_url="http://127.0.0.1:2000")
chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

# 3. --- Enhanced Vector Store Setup ---
def get_vector_store():
    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
        if collection.count() == 0:
            raise chromadb.errors.InvalidCollectionException("Empty collection")
        return Chroma(
            client=chroma_client,
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"}  # Better similarity metric
        )
    except:
        raw_docs = [Document(page_content=d) for d in [
         "NovaMind AI is a startup founded in 2024 with the mission of building intelligent AI agents for enterprise workflows.",
         "The founding team consists of three members: Aisha Kapoor (CEO), Rahul Mehta (CTO), and Jenny Lin (COO), all with strong backgrounds in AI research and product development.",
         "NovaMind's first product is called TaskPilot, an AI agent designed to automate repetitive operational tasks like data entry, email drafting, and CRM updates.",
         "NovaMind AI raised a $2.5M seed round from Gradient Ventures and Lightspeed India in early 2025 to scale its engineering team and onboard pilot customers.",
         "The company operates out of Bengaluru and is focused on the SaaS market, targeting mid-sized enterprises with 100-500 employees.",
         "TaskPilot integrates with tools like Slack, Gmail, Salesforce, and Notion to enable end-to-end automation without requiring code.",
         "The vision is to eventually build multi-agent systems that can collaborate autonomously to handle complex business workflows.",
         "NovaMind is actively hiring AI researchers and full-stack developers to accelerate product development.",
         "Early customer feedback highlights that TaskPilot saves teams up to 15 hours a week by handling repetitive, rule-based tasks.",
         "NovaMind AI believes that the future of work will be agentic — where employees will delegate mundane tasks to AI agents and focus on strategic, creative work."
     ]]
        return Chroma.from_documents(
            documents=raw_docs,
            embedding=embedding_model,
            client=chroma_client,
            collection_name=COLLECTION_NAME,
            collection_metadata={"hnsw:space": "cosine"}
        )

vectordb = get_vector_store()

# 4. --- Enhanced Retriever with Hybrid Search ---
retriever = vectordb.as_retriever(
    search_type="mmr",  # Maximal Marginal Relevance for better diversity
    search_kwargs=SEARCH_KWARGS
)

# 5. --- Improved Prompt Engineering ---
rag_prompt = PromptTemplate(
    template="""Analyze the following context and question carefully. Follow these steps:
1. Identify ALL relevant information from the context
2. Synthesize a comprehensive answer
3. If needed, supplement with general knowledge (clearly indicate when doing so)

Context:
{context}

Question: {question}

Provide a detailed response formatted as:
- [Context Answer]: (thorough explanation using context)
- [General Knowledge]: (only if context is insufficient, mark as 'General Knowledge')""",
    input_variables=["context", "question"]
)

# 6. --- Advanced Relevance Detection ---
def is_context_relevant(retrieved_docs, query_embedding, threshold=0.65):
    if not retrieved_docs:
        return False
    # Calculate average similarity (pseudo-code - implement based on your embedding client)
    total_sim = sum(embedding_model.similarity(query_embedding, doc.embedding) 
               for doc in retrieved_docs)
    return (total_sim / len(retrieved_docs)) > threshold

# 7. --- Optimized RAG Pipeline ---
def smart_rag_qa(query: str):
    # Enhanced retrieval with query expansion
    expanded_query = f"{query} - company information, business details, product features"
    retrieved_docs = retriever.get_relevant_documents(expanded_query)
    
    # Context processing
    context = "\n".join({d.page_content for d in retrieved_docs})  # Deduplicate
    
    # Dynamic response strategy
    if context:
        print("📚 Using domain knowledge (Enhanced RAG)...")
        response_chain = StuffDocumentsChain(
            llm_chain=LLMChain(llm=llm, prompt=rag_prompt),
            document_variable_name="context"
        )
        result = response_chain.run(input_documents=retrieved_docs, question=query)
        return result, retrieved_docs
    else:
        print("🌐 Using general knowledge...")
        general_prompt = f"Answer concisely: {query}"
        response = llm.invoke(general_prompt)
        return response.content, []

# Example usage
questions = [
    "Who are the founders and what are their roles?",
    "What does NovaMind think about weather prediction?",
    "Tell me about NovaMind's vision and recent press coverage.",
    "Explain CI/CD pipelines"
]

for question in questions:
    print(f"\n❓ Question: {question}")
    answer, sources = smart_rag_qa(question)
    print(f"🧠 Answer:\n{answer}")
    if sources:
        print("\n📚 Relevant Sources:")
        for doc in sources[:3]:  # Show top 3 sources
            print(f" - {doc.page_content[:120]}...")























# SMALL IMPROVEMENTS IN DEEPSEEK MODEL (BETA Testing)


# import os
# import chromadb
# import logging
# import numpy as np
# from functools import lru_cache
# from typing import List, Tuple

# from langchain_chroma import Chroma
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import PromptTemplate
# from langchain.schema.document import Document
# from langchain.retrievers import ContextualCompressionRetriever
# from langchain.retrievers.document_compressors import EmbeddingsFilter

# from embedding_client import EmbeddingClient  # Ensure this has similarity methods

# # 🔥 1. Production Configuration
# class Config:
#     GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "api")
#     CHROMA_DIR = "./chroma_store"
#     COLLECTION_NAME = "NovaMind_info"
#     SEARCH_CONFIG = {
#         "search_type": "mmr",
#         "k": 8,
#         "fetch_k": 25,
#         "lambda_mult": 0.6,
#         "score_threshold": 0.45
#     }
#     RERANK_TOP_K = 3
#     SIMILARITY_THRESHOLD = 0.68
#     CACHE_SIZE = 1000

# # 🔥 2. Production-grade initialization
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class NovaMindRAG:
#     def __init__(self):
#         self.embedder = EmbeddingClient(api_url="http://127.0.0.1:2000")
#         self.llm = ChatGoogleGenerativeAI(
#             model="gemini-2.0-flash",  # "gemini-2.0-pro" might not be available via API key yet
#             temperature=0.2,
#             google_api_key=Config.GOOGLE_API_KEY  # ✅ explicitly use the API key
#         )
#         self.vectordb = self._init_vector_db()
#         self.retriever = self._init_retriever()
#         self.prompt = self._init_prompt_template()
        
#     def _init_vector_db(self):
#         client = chromadb.PersistentClient(path=Config.CHROMA_DIR)
#         try:
#             collection = client.get_collection(Config.COLLECTION_NAME)
#             if collection.count() == 0:
#                 raise ValueError("Empty collection")
#             return Chroma(
#                 client=client,
#                 collection_name=Config.COLLECTION_NAME,
#                 embedding_function=self.embedder,
#                 collection_metadata={"hnsw:space": "cosine"}
#             )
#         except:
#             return self._seed_vector_db(client)
    
#     def _seed_vector_db(self, client):
#         docs = [Document(page_content=d) for d in [
#             "NovaMind AI is a startup founded in 2024...",  # Your documents
#             # ... other documents ...
#         ]]
#         return Chroma.from_documents(
#             documents=docs,
#             embedding=self.embedder,
#             client=client,
#             collection_name=Config.COLLECTION_NAME,
#             collection_metadata={"hnsw:space": "cosine"}
#         )
    
#     def _init_retriever(self):
#         base_retriever = self.vectordb.as_retriever(
#             search_type=Config.SEARCH_CONFIG["search_type"],
#             search_kwargs={
#                 "k": Config.SEARCH_CONFIG["k"],
#                 "fetch_k": Config.SEARCH_CONFIG["fetch_k"],
#                 "lambda_mult": Config.SEARCH_CONFIG["lambda_mult"],
#                 "score_threshold": Config.SEARCH_CONFIG["score_threshold"]
#             }
#         )
        
#         # 🔥 3. Add contextual compression and reranking
#         embeddings_filter = EmbeddingsFilter(
#             embeddings=self.embedder,
#             similarity_threshold=Config.SIMILARITY_THRESHOLD
#         )
#         return ContextualCompressionRetriever(
#             base_compressor=embeddings_filter,
#             base_retriever=base_retriever
#         )
    
#     def _init_prompt_template(self):
#         return PromptTemplate(
#             template="""**Knowledge Integration Protocol**
# 1. Analyze question: {question}
# 2. Contextual Relevance: {context}
# 3. Required Response Structure:
#    - Factual Elements (from context)
#    - Conceptual Expansion (general knowledge if needed)
#    - Confidence Score (0-100 based on context coverage)
   
# Generate response with this format:
# **Factual Answer**
# {context_insights}

# **Supplementary Knowledge** 
# {supplementary_info}

# **Confidence**: {confidence_score}%""",
#             input_variables=["question", "context"]
#         )
    
#     # 🔥 4. Enhanced similarity calculation
#     def _calculate_similarity(self, query: str, docs: List[Document]) -> float:
#         query_embedding = self.embedder.embed_query(query)
#         doc_embeddings = [self.embedder.embed_query(d.page_content) for d in docs]
#         similarities = [
#             self._cosine_similarity(query_embedding, de)
#             for de in doc_embeddings
#         ]
#         return np.mean(similarities)
    
#     def _cosine_similarity(self, a, b):
#         return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
#     # 🔥 5. Production-grade query processing with caching
#     @lru_cache(maxsize=Config.CACHE_SIZE)
#     def _process_query(self, query: str) -> Tuple[str, List[Document]]:
#         try:
#             # Query expansion
#             expanded_query = f"{query} company business product service technology"
#             docs = self.retriever.get_relevant_documents(expanded_query)
            
#             if not docs:
#                 return "General knowledge response", []
                
#             # Rerank and filter
#             docs = self._rerank_documents(query, docs)[:Config.RERANK_TOP_K]
#             context = "\n".join({d.page_content for d in docs})
            
#             # Generate response
#             response = self.llm.invoke(self.prompt.format(
#                 question=query,
#                 context=context
#             ))
            
#             return response.content, docs
            
#         except Exception as e:
#             logger.error(f"Query processing failed: {str(e)}")
#             return "Error processing request", []
    
#     def _rerank_documents(self, query: str, docs: List[Document]) -> List[Document]:
#         """Rerank documents based on semantic similarity to query"""
#         query_embedding = self.embedder.embed_query(query)
#         doc_similarities = []
        
#         for doc in docs:
#             doc_embedding = self.embedder.embed_query(doc.page_content)
#             similarity = self._cosine_similarity(query_embedding, doc_embedding)
#             doc_similarities.append((similarity, doc))
            
#         return [d for _, d in sorted(doc_similarities, reverse=True, key=lambda x: x[0])]
    
#     # 🔥 6. Public API method with telemetry
#     def query(self, question: str) -> dict:
#         response, docs = self._process_query(question)
#         confidence = self._calculate_confidence(response, docs)
        
#         return {
#             "question": question,
#             "answer": response,
#             "sources": [d.page_content[:150] + "..." for d in docs[:3]],
#             "confidence": confidence,
#             "metrics": {
#                 "response_time": None,  # Add timing logic
#                 "retrieved_docs": len(docs),
#                 "avg_similarity": self._calculate_similarity(question, docs)
#             }
#         }
    
#     def _calculate_confidence(self, response: str, docs: List[Document]) -> float:
#         if not docs:
#             return 0.0
#         content_length = sum(len(d.page_content) for d in docs)
#         unique_entities = len({e for d in docs for e in d.metadata.get('entities', [])})
#         return min(100, (content_length * 0.1) + (unique_entities * 5))

# # 🔥 7. Benchmarking Suite
# TEST_CASES = [
#     {
#         "question": "Who are the founders and their roles?",
#         "expected_answer": ["Aisha Kapoor (CEO)", "Rahul Mehta (CTO)", "Jenny Lin (COO)"],
#         "expected_sources": 3
#     },
#     {
#         "question": "What is NovaMind's funding status?",
#         "expected_answer": ["$2.5M seed round", "Gradient Ventures", "Lightspeed India"],
#         "expected_sources": 2
#     }
# ]

# def run_benchmarks():
#     rag = NovaMindRAG()
#     results = []
    
#     for case in TEST_CASES:
#         result = rag.query(case["question"])
#         passed = all(
#             keyword in result["answer"]
#             for keyword in case["expected_answer"]
#         )
#         results.append({
#             "question": case["question"],
#             "passed": passed,
#             "confidence": result["confidence"],
#             "retrieved_sources": len(result["sources"])
#         })
    
#     return results

# if __name__ == "__main__":
#     # Example usage
#     rag = NovaMindRAG()
#     print(rag.query("Who are the founders and their roles?"))
    
#     # Run benchmarks
#     benchmark_results = run_benchmarks()
#     print("\nBenchmark Results:")
#     for res in benchmark_results:
#         print(f"{'PASS' if res['passed'] else 'FAIL'} | {res['question']} | Confidence: {res['confidence']}%")