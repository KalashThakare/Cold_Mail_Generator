"""
Cold Email Generator Application

This module provides a Streamlit-based UI for generating personalized cold emails
based on job postings extracted from web URLs.
"""

import logging
from typing import List, Dict, Optional

import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from portfolio import Portfolio
from utils import clean_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmailGenerator:
    """Handles the email generation workflow."""
    
    def __init__(self, llm: Chain, portfolio: Portfolio):
        """
        Initialize the EmailGenerator.
        
        Args:
            llm: Chain instance for LLM operations
            portfolio: Portfolio instance for querying relevant links
        """
        self.llm = llm
        self.portfolio = portfolio
        self._portfolio_loaded = False
    
    def load_web_content(self, url: str) -> str:
        """
        Fetch and clean webpage text from a given URL.
        
        Args:
            url: The URL to fetch content from
            
        Returns:
            Cleaned text content from the webpage
            
        Raises:
            ValueError: If URL is invalid or empty
            ConnectionError: If unable to fetch the webpage
        """
        if not url or not url.strip():
            raise ValueError("URL cannot be empty")
        
        try:
            logger.info(f"Loading content from URL: {url}")
            loader = WebBaseLoader([url])
            documents = loader.load()
            
            if not documents:
                raise ValueError("No content could be loaded from the URL")
            
            content = clean_text(documents[0].page_content)
            logger.info(f"Successfully loaded {len(content)} characters")
            return content
            
        except Exception as e:
            logger.error(f"Failed to load web content: {str(e)}")
            raise ConnectionError(f"Unable to fetch content from URL: {str(e)}")
    
    def generate_emails(self, jobs: List[Dict]) -> List[str]:
        """
        Generate cold emails for a list of extracted jobs.
        
        Args:
            jobs: List of job dictionaries containing job details
            
        Returns:
            List of generated email strings
            
        Raises:
            ValueError: If jobs list is empty
        """
        if not jobs:
            raise ValueError("No jobs found to generate emails")
        
        # Load portfolio only once
        if not self._portfolio_loaded:
            logger.info("Loading portfolio data")
            self.portfolio.load_portfolio()
            self._portfolio_loaded = True
        
        emails = []
        logger.info(f"Generating emails for {len(jobs)} job(s)")
        
        for idx, job in enumerate(jobs, 1):
            try:
                skills = job.get("skills", [])
                links = self.portfolio.query_links(skills)
                email = self.llm.write_mail(job, links)
                emails.append(email)
                logger.info(f"Generated email {idx}/{len(jobs)}")
            except Exception as e:
                logger.error(f"Failed to generate email for job {idx}: {str(e)}")
                emails.append(f"Error generating email: {str(e)}")
        
        return emails
    
    def process_url(self, url: str) -> List[str]:
        """
        Process a URL and generate emails for all found jobs.
        
        Args:
            url: The job posting URL to process
            
        Returns:
            List of generated emails
            
        Raises:
            ValueError: If URL is invalid or no jobs found
            ConnectionError: If unable to fetch the webpage
        """
        content = self.load_web_content(url)
        jobs = self.llm.extract_jobs(content)
        
        if not jobs:
            raise ValueError("No job postings found at the provided URL")
        
        return self.generate_emails(jobs)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'generator' not in st.session_state:
        st.session_state.generator = EmailGenerator(Chain(), Portfolio())
    if 'processing' not in st.session_state:
        st.session_state.processing = False


def render_header():
    """Render the application header."""
    st.title("📧 Cold Email Generator")
    st.markdown(
        """
        Generate personalized cold emails based on job postings.
        Simply paste a job posting URL and let AI create tailored emails for you.
        """
    )
    st.divider()


def render_input_section() -> Optional[str]:
    """
    Render the URL input section.
    
    Returns:
        The submitted URL or None
    """
    with st.form(key="url_form"):
        url_input = st.text_input(
            "Job Posting URL",
            placeholder="https://example.com/job-posting",
            help="Enter the full URL of the job posting page"
        )
        submit_button = st.form_submit_button(
            "Generate Emails",
            use_container_width=True,
            type="primary"
        )
        
        if submit_button:
            return url_input.strip() if url_input else None
    
    return None


def render_emails(emails: List[str]):
    """
    Render generated emails.
    
    Args:
        emails: List of email strings to display
    """
    st.success(f"✅ Successfully generated {len(emails)} email(s)")
    
    for idx, email in enumerate(emails, 1):
        with st.expander(f"📧 Email {idx}", expanded=(idx == 1)):
            st.markdown(email)
            st.download_button(
                label="Download Email",
                data=email,
                file_name=f"cold_email_{idx}.txt",
                mime="text/plain",
                key=f"download_{idx}"
            )


def main():
    """Main application entry point."""
    st.set_page_config(
        layout="wide",
        page_title="Cold Email Generator",
        page_icon="📧",
        initial_sidebar_state="collapsed"
    )
    
    initialize_session_state()
    render_header()
    
    # Input section
    url = render_input_section()
    
    # Process URL if submitted
    if url:
        if not st.session_state.processing:
            st.session_state.processing = True
            
            with st.spinner("🔄 Processing URL and generating emails..."):
                try:
                    emails = st.session_state.generator.process_url(url)
                    render_emails(emails)
                    
                except ValueError as ve:
                    logger.warning(f"Validation error: {str(ve)}")
                    st.error(f"⚠️ Invalid Input: {str(ve)}")
                    
                except ConnectionError as ce:
                    logger.error(f"Connection error: {str(ce)}")
                    st.error(f"🌐 Connection Error: {str(ve)}")
                    
                except Exception as e:
                    logger.exception("Unexpected error occurred")
                    st.error(f"❌ Unexpected Error: {str(e)}")
                    st.info("Please try again or contact support if the issue persists.")
            
            st.session_state.processing = False
    
    # Footer
    st.divider()
    st.caption("Built with Streamlit and LangChain | Powered by AI")


if __name__ == "__main__":
    main()