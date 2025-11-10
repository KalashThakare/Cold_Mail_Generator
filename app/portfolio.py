"""Portfolio management for ChromaDB vector storage."""

import os
import uuid
from typing import List, Dict, Any
import pandas as pd
import chromadb as db

from logger_utils import get_logger, safe_execution

logger = get_logger("Portfolio")


class Portfolio:
    """Manages portfolio projects with ChromaDB."""

    def __init__(self, file_path: str = "app/resources/my_portfolio.csv"):
        self.file_path = file_path
        self.data = self._load_csv()
        self.chroma_client = self._init_client()
        self.collection = self._get_or_create_collection("portfolio")

    def _load_csv(self) -> pd.DataFrame:
        """Load and validate CSV portfolio file."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Portfolio file not found: {self.file_path}")

        data = pd.read_csv(self.file_path)
        if data.empty or not {"Techstack", "Links"}.issubset(data.columns):
            raise ValueError("Portfolio CSV must contain 'Techstack' and 'Links' columns.")

        return data.dropna(subset=["Techstack", "Links"])

    @staticmethod
    def _init_client() -> db.Client:
        """Initialize persistent ChromaDB client."""
        return db.PersistentClient(path="VectorStore")

    def _get_or_create_collection(self, name: str):
        """Retrieve or create a ChromaDB collection."""
        return self.chroma_client.get_or_create_collection(name)

    def load_portfolio(self) -> None:
        """Load portfolio data into vector DB if not already loaded."""
        if self.collection.count() > 0:
            return

        records = [
            (
                str(row["Techstack"]).strip(),
                {"links": str(row["Links"]).strip()},
                str(uuid.uuid4()),
            )
            for _, row in self.data.iterrows()
            if pd.notna(row["Techstack"]) and pd.notna(row["Links"])
        ]

        with safe_execution(logger, "load_portfolio"):
            self.collection.add(
                documents=[tech for tech, _, _ in records],
                metadatas=[meta for _, meta, _ in records],
                ids=[uid for _, _, uid in records],
            )
            logger.info(f"Loaded {len(records)} portfolio entries..")

    def query_links(self, skills: List[str], n_results: int = 2) -> List[Dict[str, Any]]:
        """Retrieve project links relevant to provided skills."""
        valid_skills = [s.strip() for s in skills if s and isinstance(s, str) and s.strip()]
        if not valid_skills:
            return []

        with safe_execution(logger, "query_links"):
            result = self.collection.query(query_texts=valid_skills, n_results=n_results)
            return result.get("metadatas", [])
