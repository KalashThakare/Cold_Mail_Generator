"""Chain module for LLM operations."""

import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

from logger_utils import get_logger, safe_execution

load_dotenv()
logger = get_logger("Chain")


class Chain:
    """Encapsulates LLM operations for job extraction and cold email generation.."""

    _JOB_PROMPT = PromptTemplate.from_template("""
### SCRAPED TEXT:
{page_data}

### INSTRUCTION:
Extract job postings from the text and return valid JSON containing:
`role`, `experience`, `skills`, and `description`.
Return only the JSON.
""")

    _EMAIL_PROMPT = PromptTemplate.from_template("""
### JOB DESCRIPTION:
{job_description}

### INSTRUCTION:
You are Kalash, a passionate student developer specializing in AI and software.
Write a concise, personalized cold email for the above job, showing enthusiasm,
relevant projects, and proof links: {link_list}. Output email only.
""")

    def __init__(self):
        """Initialize LLM connection."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("Missing GROQ_API_KEY in environment.")

        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            groq_api_key=api_key
        )
        self._parser = JsonOutputParser()
        logger.info("Chain initialized successfully.")

    def extract_jobs(self, cleaned_text: str) -> List[Dict[str, Any]]:
        """Extract structured job listings from text."""
        if not cleaned_text.strip():
            raise ValueError("Cleaned text cannot be empty.")

        with safe_execution(logger, "extract_jobs"):
            response = (self._JOB_PROMPT | self.llm).invoke({"page_data": cleaned_text})
            result = self._parser.parse(response.content)
            jobs = result if isinstance(result, list) else [result]
            logger.info(f"Extracted {len(jobs)} job(s).")
            return jobs

    def write_mail(self, job: Dict[str, Any], links: List[Dict[str, Any]]) -> str:
        """Generate a cold email for a given job posting."""
        if not job:
            raise ValueError("Job data is required to generate an email.")

        with safe_execution(logger, "write_mail"):
            response = (self._EMAIL_PROMPT | self.llm).invoke({
                "job_description": str(job),
                "link_list": links
            })
            logger.info(f"Generated email for: {job.get('role', 'Unknown')}")
            return response.content.strip()
