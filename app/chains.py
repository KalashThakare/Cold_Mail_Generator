"""Chain module for LLM operations."""

import os
import logging
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

load_dotenv()
logger = logging.getLogger("Chain")


class Chain:
    """Encapsulates all LLM-based operations: job extraction and email writing."""

    _JOB_PROMPT = PromptTemplate.from_template("""
### SCRAPED TEXT FROM WEBSITE:
{page_data}

### INSTRUCTION:
Extract job postings from the text above and return only valid JSON with fields:
`role`, `experience`, `skills`, and `description`.
Return **only** valid JSON (no explanation).
""")

    _EMAIL_PROMPT = PromptTemplate.from_template("""
### JOB DESCRIPTION:
{job_description}

### INSTRUCTION:
You are Kalash, a passionate student developer in AI & software solutions.
Write a cold email for the above job, showing enthusiasm, relevant projects, and
including proof links: {link_list}. Avoid preambles — output the email only.
""")

    def __init__(self):
        """Initialize LLM with Groq API key."""
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
        """Extract structured job data from webpage text."""
        if not cleaned_text.strip():
            raise ValueError("Cleaned text cannot be empty.")

        try:
            response = (self._JOB_PROMPT | self.llm).invoke({"page_data": cleaned_text})
            result = self._parser.parse(response.content)
            jobs = result if isinstance(result, list) else [result]
            logger.info(f"Extracted {len(jobs)} job(s).")
            return jobs
        except OutputParserException as e:
            logger.error(f"Job parsing failed: {e}")
            raise OutputParserException("Failed to parse job data from response.") from e
        except Exception as e:
            logger.error(f"Job extraction failed: {e}")
            raise RuntimeError("Unexpected error while extracting jobs.") from e

    def write_mail(self, job: Dict[str, Any], links: List[Dict[str, Any]]) -> str:
        """Generate cold email content for a job posting."""
        if not job:
            raise ValueError("Job details are required to generate email.")

        try:
            response = (self._EMAIL_PROMPT | self.llm).invoke({
                "job_description": str(job),
                "link_list": links
            })
            logger.info(f"Email generated for role: {job.get('role', 'Unknown')}")
            return response.content.strip()
        except Exception as e:
            logger.error(f"Email generation failed: {e}")
            raise RuntimeError("Failed to generate email.") from e
