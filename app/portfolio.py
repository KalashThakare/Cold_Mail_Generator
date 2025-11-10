"""Portfolio management for ChromaDB vector storage."""

import os
import pandas as pd
import chromadb as db
import uuid


class Portfolio:
    """Manages portfolio projects with ChromaDB."""
    
    def __init__(self, file_path="app/resources/my_portfolio.csv"):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.data = pd.read_csv(file_path)
        if self.data.empty or not all(col in self.data.columns for col in ["Techstack", "Links"]):
            raise ValueError("Invalid CSV format")
        
        self.chroma_client = db.PersistentClient("VectorStore")
        self.collection = self.chroma_client.get_or_create_collection("portfolio")
    
    def load_portfolio(self):
        if self.collection.count():
            return
        
        for _, row in self.data.iterrows():
            tech = str(row["Techstack"]).strip()
            if tech and tech.lower() != 'nan':
                self.collection.add(
                    documents=[tech],
                    metadatas=[{"links": str(row["Links"]).strip()}],
                    ids=[str(uuid.uuid4())]
                )
    
    def query_links(self, skills, n_results=2):
        if not skills:
            return []
        
        valid = [s.strip() for s in skills if s and s.strip()]
        return self.collection.query(query_texts=valid, n_results=n_results).get("metadatas", []) if valid else []