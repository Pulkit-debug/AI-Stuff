# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.llms import Ollama
# from langchain.chains import RetrievalQA
# import chromadb
# import ollama

# # STEP 1: Load documents
# documents = [
#asdfsdfasfa
# ]

# # STEP 2: Create Chroma DB collection
# client = chromadb.Client()
# collection = client.create_collection(name="myDetails")

# # STEP 3: Embed and store each document
# for i, d in enumerate(documents):
#     response = ollama.embed(model="mxbai-embed-large", input=d)
#     embedding = response["embeddings"][0]  # <- FIX: unwrap the embedding
#     collection.add(
#         ids=[str(i)],
#         embeddings=[embedding],  # still needs to be wrapped in a list here
#         documents=[d]
#     )

# # STEP 4: Define the input query
# user_prompt = "Do you know someone named pulkit? what is he doing in his profession and how can i go in his line of work?"

# # STEP 5: Embed the prompt to query the collection
# embed_response = ollama.embed(model="mxbai-embed-large", input=user_prompt)
# prompt_embedding = embed_response["embeddings"][0]  # <- FIX: unwrap here too

# # STEP 6: Query
# results = collection.query(
#     query_embeddings=[prompt_embedding],  # still wrap in list
#     n_results=1
# )
# retrieved_data = results['documents'][0][0]  # First result

# # STEP 7: Generate a response using the retrieved context
# final_prompt = f"Using this data: {retrieved_data}. Respond to this prompt: {user_prompt}"

# output = ollama.generate(
#     model="qwen2.5:7b",
#     prompt=final_prompt
# )

# # STEP 8: Show the output
# print(output['response'])


# # # STEP 2: Split text into smaller chunks
# # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
# # docs = text_splitter.split_documents(documents)

# # # STEP 3: Create embeddings
# # embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # # STEP 4: Store in Chroma vector DB
# # db = Chroma.from_documents(docs, embedding, persist_directory="./chroma_db")
# # db.persist()

# # # STEP 5: Create retriever
# # retriever = db.as_retriever(search_kwargs={"k": 3})

# # # STEP 6: Load local LLM via Ollama (make sure it’s running)
# # llm = Ollama(model="qwen2.5:7b")  # or "llama2", "gemma", etc.

# # # STEP 7: Create QA chain
# # qa_chain = RetrievalQA.from_chain_type(
# #     llm=llm,
# #     retriever=retriever,
# #     return_source_documents=True
# # )

# # # STEP 8: Ask a question
# # query = "If i want to be like pulkit, what should i do in my career and where should i apply for job?"
# # response = qa_chain(query)

# # print("\n🧠 Answer:")
# # print(response['result'])

# # print("\n📚 Source Documents:")
# # for doc in response["source_documents"]:
# #     print("  -", doc.metadata.get("source"))
