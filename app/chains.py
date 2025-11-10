"""
Chain module for LLM operations.

This module provides the Chain class for handling job extraction and email generation
using LangChain and Groq API.
"""

import os
import logging
from typing import List, Dict, Union

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class Chain:
    """Handles LLM operations for job extraction and email generation."""
    
    # Class-level constants
    DEFAULT_MODEL = "llama-3.1-8b-instant"
    DEFAULT_TEMPERATURE = 0
    
    JOB_EXTRACTION_TEMPLATE = """
### SCRAPED TEXT FROM WEBSITE:
{page_data}

### INSTRUCTION:
The scraped text is from the career's page of a website.
Your job is to extract the job postings and return them in JSON format containing 
the following keys: `role`, `experience`, `skills` and `description`.
Only return the valid JSON.

### VALID JSON (NO PREAMBLE):
"""
    
    EMAIL_TEMPLATE = """
### JOB DESCRIPTION:
{job_description}

### INSTRUCTION:
You are Kalash, a passionate student developer specializing in AI & Software solutions, 
eager to contribute your skills and enthusiasm to company. With hands-on experience from 
personal and academic projects, you have helped organizations and teams achieve process 
automation, improved efficiency, and cost reduction through innovative solutions.

Your task is to craft a cold email to the client (potential employer) for the job described 
above, highlighting your relevant skills, eagerness to learn, and concrete examples of your work.

Also, select the most relevant items from your portfolio or previous project links here: 
{link_list}, and add them as proof of your capabilities.

Remember you are Kalash, an enthusiastic student developer.
Do not provide a preamble.

### EMAIL (NO PREAMBLE):
"""

    def __init__(self):
        """Initialize the Chain with ChatGroq LLM."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.llm = ChatGroq(
            temperature=self.DEFAULT_TEMPERATURE,
            model=self.DEFAULT_MODEL,
            groq_api_key=api_key
        )
        logger.info("Chain initialized successfully")
    
    def _parse_json_response(self, response_content: str) -> List[Dict]:
        """
        Parse JSON response from LLM.
        
        Args:
            response_content: Raw response content from LLM
            
        Returns:
            List of parsed job dictionaries
            
        Raises:
            OutputParserException: If parsing fails
        """
        try:
            json_parser = JsonOutputParser()
            parsed_result = json_parser.parse(response_content)
            return parsed_result if isinstance(parsed_result, list) else [parsed_result]
        except OutputParserException as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            raise OutputParserException("Context too big. Unable to parse jobs.")
    
    def extract_jobs(self, cleaned_text: str) -> List[Dict]:
        """
        Extract job postings from cleaned webpage text.
        
        Args:
            cleaned_text: Cleaned text content from careers page
            
        Returns:
            List of job dictionaries containing role, experience, skills, and description
            
        Raises:
            OutputParserException: If unable to parse jobs from response
        """
        if not cleaned_text or not cleaned_text.strip():
            raise ValueError("Cleaned text cannot be empty")
        
        logger.info("Extracting jobs from text")
        
        prompt = PromptTemplate.from_template(self.JOB_EXTRACTION_TEMPLATE)
        chain = prompt | self.llm
        
        response = chain.invoke(input={"page_data": cleaned_text})
        jobs = self._parse_json_response(response.content)
        
        logger.info(f"Extracted {len(jobs)} job(s)")
        return jobs
    
    def write_mail(self, job: Dict, links: List[Dict]) -> str:
        """
        Generate a cold email for a specific job posting.
        
        Args:
            job: Job dictionary containing job details
            links: List of relevant portfolio links
            
        Returns:
            Generated email content as string
        """
        if not job:
            raise ValueError("Job dictionary cannot be empty")
        
        logger.info(f"Generating email for job: {job.get('role', 'Unknown')}")
        
        prompt = PromptTemplate.from_template(self.EMAIL_TEMPLATE)
        chain = prompt | self.llm
        
        response = chain.invoke({
            "job_description": str(job),
            "link_list": links
        })
        
        logger.info("Email generated successfully")
        return response.content


def main():
    """Test the Chain functionality."""
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        print(f"GROQ_API_KEY found: {api_key[:10]}...")
    else:
        print("GROQ_API_KEY not found!")


if __name__ == "__main__":
    main()