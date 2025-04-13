import httpx
from langchain.embeddings.base import Embeddings
from typing import List

class EmbeddingClient(Embeddings):
    def __init__(self, api_url="http://127.0.0.1:2000"):
        self.api_url = api_url
        self.client = httpx.Client(timeout=30.0)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embeddings = []
        for text in texts:
            response = self.client.post(
                f"{self.api_url}/embed", 
                json={"text": text}
            )
            if response.status_code == 200:
                embeddings.append(response.json()["embeddings"])
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        response = self.client.post(
            f"{self.api_url}/embed", 
            json={"text": text}
        )
        if response.status_code == 200:
            return response.json()["embeddings"]
        return []