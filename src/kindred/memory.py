import os
import chromadb
import pandas as pd
import pymupdf  # PyMuPDF


class KnowledgeBase:
    def __init__(self, agent_name: str, orchestrator):
        self.orchestrator = orchestrator
        self.client = chromadb.PersistentClient(path=f"./memory/{agent_name}")
        self.collection = self.client.get_or_create_collection("knowledge")

    def _extract_text(self, file_path: str) -> str:
        """Internal helper to pull text from various file types."""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".pdf":
            doc = pymupdf.open(file_path)
            return "\n".join([page.get_text() for page in doc])
        
        elif ext == ".csv":
            df = pd.read_csv(file_path)
            # Combine all columns into a single string per row
            return "\n".join(df.astype(str).agg(' | '.join, axis=1))
        
        elif ext == ".txt":
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        raise ValueError(f"Unsupported file type: {ext}")

    def add_document(self, file_path: str):
        content = self._extract_text(file_path)
        chunks = [c.strip() for c in content.split("\n\n") if len(c.strip()) > 10]
        
        # 2026 Check: Ensure you have the model pulled!
        # Run: ollama pull nomic-embed-text
        response = self.orchestrator.client.embed(
            model="nomic-embed-text", 
            input=chunks
        )
        
        # CRITICAL: In 2026, response.embeddings is a list of vectors.
        # ChromaDB needs these to match the IDs 1:1.
        self.collection.add(
            documents=chunks,
            embeddings=response.embeddings, # Access as attribute, not dict
            ids=[f"{os.path.basename(file_path)}_{i}" for i in range(len(chunks))]
        )