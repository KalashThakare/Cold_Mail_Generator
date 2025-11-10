"""Portfolio Management Module for ChromaDB-based project storage."""

import os
import logging
from typing import List, Dict
import pandas as pd
import chromadb as db
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Portfolio:
    """Manages portfolio projects with ChromaDB vector storage."""
    
    REQUIRED_COLUMNS = ["Techstack", "Links"]
    
    def __init__(self, file_path: str = "app/resources/my_portfolio.csv"):
        """Initialize portfolio from CSV file."""
        self.file_path = file_path
        self._validate_and_load()
        self.chroma_client = db.PersistentClient("VectorStore")
        self.collection = self.chroma_client.get_or_create_collection("portfolio")
    
    def _validate_and_load(self) -> None:
        """Validate file exists and load CSV data."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Portfolio file not found: {self.file_path}")
        
        self.data = pd.read_csv(self.file_path)
        
        if self.data.empty:
            raise ValueError("Portfolio CSV is empty")
        
        missing = set(self.REQUIRED_COLUMNS) - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing columns: {', '.join(missing)}")
        
        logger.info(f"Loaded {len(self.data)} portfolio items")
    
    def load_portfolio(self) -> None:
        """Load portfolio items into ChromaDB if collection is empty."""
        if self.collection.count():
            logger.info(f"Portfolio already loaded ({self.collection.count()} items)")
            return
        
        added = 0
        for _, row in self.data.iterrows():
            techstack = str(row["Techstack"]).strip()
            if techstack and techstack.lower() != 'nan':
                self.collection.add(
                    documents=[techstack],
                    metadatas=[{"links": str(row["Links"]).strip()}],
                    ids=[str(uuid.uuid4())]
                )
                added += 1
        
        logger.info(f"Loaded {added} items into portfolio")
    
    def query_links(self, skills: List[str], n_results: int = 2) -> List[Dict]:
        """Query portfolio for projects matching skills."""
        if not skills:
            return []
        
        valid_skills = [s.strip() for s in skills if s and s.strip()]
        if not valid_skills:
            return []
        
        results = self.collection.query(
            query_texts=valid_skills,
            n_results=n_results
        )
        return results.get("metadatas", [])
    
    def get_portfolio_count(self) -> int:
        """Get number of items in portfolio."""
        return self.collection.count()
    
    def clear_portfolio(self) -> None:
        """Clear all portfolio items."""
        self.chroma_client.delete_collection("portfolio")
        self.collection = self.chroma_client.get_or_create_collection("portfolio")
        logger.info("Portfolio cleared")