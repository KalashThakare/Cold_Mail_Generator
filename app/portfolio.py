"""
Portfolio Management Module

This module provides functionality to manage a portfolio of projects with their
associated tech stacks and links, using ChromaDB for vector-based retrieval.
"""

import os
import logging
from typing import List, Dict, Optional
import pandas as pd
import chromadb as db
import uuid


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Portfolio:
    """
    Manages a portfolio of projects stored in CSV format and indexed in ChromaDB.
    
    Attributes:
        file_path (str): Path to the CSV file containing portfolio data
        data (pd.DataFrame): Loaded portfolio data
        chroma_client: ChromaDB persistent client instance
        collection: ChromaDB collection for storing portfolio items
    """
    
    REQUIRED_COLUMNS = ["Techstack", "Links"]
    DEFAULT_FILE_PATH = "app/resources/my_portfolio.csv"
    COLLECTION_NAME = "portfolio"
    VECTOR_STORE_PATH = "VectorStore"
    
    def __init__(self, file_path: str = DEFAULT_FILE_PATH):
        """
        Initialize the Portfolio with data from CSV file.
        
        Args:
            file_path (str): Path to the portfolio CSV file
            
        Raises:
            FileNotFoundError: If the CSV file doesn't exist
            ValueError: If CSV is empty or missing required columns
        """
        self.file_path = file_path
        self.data = None
        self.chroma_client = None
        self.collection = None
        
        try:
            self._validate_file_path()
            self._load_csv_data()
            self._initialize_chromadb()
            logger.info("Portfolio initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize portfolio: {str(e)}")
            raise
    
    def _validate_file_path(self) -> None:
        """
        Validate that the portfolio CSV file exists.
        
        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Portfolio file not found: {self.file_path}")
        logger.info(f"Portfolio file found: {self.file_path}")
    
    def _load_csv_data(self) -> None:
        """
        Load and validate data from the CSV file.
        
        Raises:
            ValueError: If CSV is empty or missing required columns
            pd.errors.EmptyDataError: If CSV file is empty
        """
        try:
            self.data = pd.read_csv(self.file_path)
            
            if self.data.empty:
                raise ValueError("Portfolio CSV file is empty")
            
            missing_cols = set(self.REQUIRED_COLUMNS) - set(self.data.columns)
            if missing_cols:
                raise ValueError(
                    f"Missing required columns: {', '.join(missing_cols)}"
                )
            
            logger.info(f"Loaded {len(self.data)} portfolio items from CSV")
            
        except pd.errors.EmptyDataError:
            raise ValueError("Portfolio CSV file is empty or corrupted")
    
    def _initialize_chromadb(self) -> None:
        """
        Initialize ChromaDB client and get or create the portfolio collection.
        """
        try:
            self.chroma_client = db.PersistentClient(self.VECTOR_STORE_PATH)
            self.collection = self._get_or_create_collection()
            logger.info("ChromaDB initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
    
    def _get_or_create_collection(self):
        """
        Get existing collection or create a new one.
        
        Returns:
            Collection: ChromaDB collection instance
        """
        return self.chroma_client.get_or_create_collection(
            name=self.COLLECTION_NAME
        )
    
    def load_portfolio(self) -> None:
        """
        Load portfolio items into ChromaDB collection if not already loaded.
        
        This method is idempotent - it only loads data if the collection is empty.
        """
        try:
            if self.collection.count() > 0:
                logger.info(
                    f"Portfolio already loaded with {self.collection.count()} items"
                )
                return
            
            items_added = 0
            for _, row in self.data.iterrows():
                if self._add_item_to_collection(row):
                    items_added += 1
            
            logger.info(f"Successfully loaded {items_added} items into portfolio")
            
        except Exception as e:
            logger.error(f"Error loading portfolio: {str(e)}")
            raise
    
    def _add_item_to_collection(self, row: pd.Series) -> bool:
        """
        Add a single portfolio item to the ChromaDB collection.
        
        Args:
            row (pd.Series): Row from the portfolio DataFrame
            
        Returns:
            bool: True if item was added successfully, False otherwise
        """
        try:
            techstack = str(row["Techstack"]).strip()
            links = str(row["Links"]).strip()
            
            if not techstack or techstack.lower() == 'nan':
                logger.warning("Skipping item with empty Techstack")
                return False
            
            self.collection.add(
                documents=[techstack],
                metadatas=[{"links": links}],
                ids=[str(uuid.uuid4())]
            )
            return True
            
        except Exception as e:
            logger.error(f"Error adding item to collection: {str(e)}")
            return False
    
    def query_links(
        self, 
        skills: List[str], 
        n_results: int = 2
    ) -> List[Dict[str, str]]:
        """
        Query the portfolio for projects matching the given skills.
        
        Args:
            skills (List[str]): List of skills/technologies to search for
            n_results (int): Number of results to return (default: 2)
            
        Returns:
            List[Dict[str, str]]: List of metadata dictionaries containing project links
            
        Raises:
            ValueError: If skills list is empty
        """
        if not skills or not isinstance(skills, list):
            logger.warning("Empty or invalid skills list provided")
            raise ValueError("Skills must be a non-empty list")
        
        try:
            # Filter out empty strings
            valid_skills = [s.strip() for s in skills if s and s.strip()]
            
            if not valid_skills:
                logger.warning("No valid skills after filtering")
                return []
            
            results = self.collection.query(
                query_texts=valid_skills,
                n_results=n_results
            )
            
            metadatas = results.get("metadatas", [])
            logger.info(f"Found {len(metadatas)} results for skills: {valid_skills}")
            return metadatas
            
        except Exception as e:
            logger.error(f"Error querying links: {str(e)}")
            return []
    
    def get_portfolio_count(self) -> int:
        """
        Get the number of items currently in the portfolio collection.
        
        Returns:
            int: Number of portfolio items
        """
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error getting portfolio count: {str(e)}")
            return 0
    
    def clear_portfolio(self) -> None:
        """
        Clear all items from the portfolio collection.
        
        Useful for testing or maintenance purposes.
        """
        try:
            self.chroma_client.delete_collection(name=self.COLLECTION_NAME)
            self.collection = self._get_or_create_collection()
            logger.info("Portfolio cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing portfolio: {str(e)}")
            raise