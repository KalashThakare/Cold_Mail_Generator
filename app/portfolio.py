"""Portfolio management for ChromaDB vector storage."""

import os
import uuid
from typing import List, Dict, Any

import pandas as pd
import chromadb as db


class Portfolio:
    """Handles portfolio projects using ChromaDB as a vector store."""

    def __init__(self, file_path: str = "app/resources/my_portfolio.csv"):
        self.file_path = file_path
        self.data = self._load_csv()
        self.chroma_client = self._init_chroma_client()
        self.collection = self._get_or_create_collection("portfolio")

    def _load_csv(self) -> pd.DataFrame:
        """Loads and validates the portfolio CSV file."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Portfolio file not found: {self.file_path}")

        data = pd.read_csv(self.file_path)
        required_columns = {"Techstack", "Links"}

        if data.empty:
            raise ValueError("Portfolio CSV is empty.")
        if not required_columns.issubset(data.columns):
            raise ValueError(f"Portfolio CSV must contain columns: {required_columns}")

        return data.dropna(subset=["Techstack", "Links"])

    @staticmethod
    def _init_chroma_client() -> db.Client:
        """Initializes the ChromaDB persistent client."""
        return db.PersistentClient(path="VectorStore")

    def _get_or_create_collection(self, name: str):
        """Fetches or creates a ChromaDB collection."""
        return self.chroma_client.get_or_create_collection(name)

    def load_portfolio(self) -> None:
        """Populates ChromaDB collection from CSV if empty."""
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

        if not records:
            raise ValueError("No valid portfolio records found to load.")

        self.collection.add(
            documents=[tech for tech, _, _ in records],
            metadatas=[meta for _, meta, _ in records],
            ids=[uid for _, _, uid in records],
        )

    def query_links(self, skills: List[str], n_results: int = 2) -> List[Dict[str, Any]]:
        """Queries project links matching given skills."""
        valid_skills = [s.strip() for s in skills if s and isinstance(s, str) and s.strip()]
        if not valid_skills:
            return []

        try:
            results = self.collection.query(query_texts=valid_skills, n_results=n_results)
            return results.get("metadatas", [])
        except Exception as e:
            print(f"ChromaDB query failed: {e}")
            return []
