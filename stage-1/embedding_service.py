import os
from langchain.embeddings import HuggingFaceEmbeddings
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# Create FastAPI app
app = FastAPI(title="Embedding Service")

# Define request model
class TextRequest(BaseModel):
    text: str

# Define response model
class EmbeddingResponse(BaseModel):
    embeddings: List[float]

# Load the model once when the server starts
print("🔄 Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("✅ Model loaded and ready to use")

@app.post("/embed", response_model=EmbeddingResponse)
async def embed(request: TextRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    # Get embeddings
    result = embedding_model.embed_documents([request.text])
    # Convert numpy array to list for JSON serialization
    embeddings = [float(x) for x in result[0]]
    
    return {"embeddings": embeddings}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=2000)