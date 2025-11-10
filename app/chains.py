"""Chain module for LLM operations."""

import os
import logging
from typing import List, Dict

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class Chain:
    """Handles LLM operations for job extraction and email generation."""
    
    JOB_TEMPLATE = """
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
        """Initialize Chain with ChatGroq LLM."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        
        self.llm = ChatGroq(
            temperature=0,
            model="llama-3.1-8b-instant",
            groq_api_key=api_key
        )
        logger.info("Chain initialized")
    
    def extract_jobs(self, cleaned_text: str) -> List[Dict]:
        """Extract job postings from cleaned webpage text."""
        if not cleaned_text or not cleaned_text.strip():
            raise ValueError("Cleaned text cannot be empty")
        
        prompt = PromptTemplate.from_template(self.JOB_TEMPLATE)
        chain = prompt | self.llm
        response = chain.invoke({"page_data": cleaned_text})
        
        try:
            parser = JsonOutputParser()
            result = parser.parse(response.content)
            jobs = result if isinstance(result, list) else [result]
            logger.info(f"Extracted {len(jobs)} job(s)")
            return jobs
        except OutputParserException as e:
            logger.error(f"Parse error: {str(e)}")
            raise OutputParserException("Context too big. Unable to parse jobs.")
    
    def write_mail(self, job: Dict, links: List[Dict]) -> str:
        """Generate cold email for job posting."""
        if not job:
            raise ValueError("Job dictionary cannot be empty")
        
        prompt = PromptTemplate.from_template(self.EMAIL_TEMPLATE)
        chain = prompt | self.llm
        response = chain.invoke({
            "job_description": str(job),
            "link_list": links
        })
        
        logger.info(f"Generated email for: {job.get('role', 'Unknown')}")
        return response.content