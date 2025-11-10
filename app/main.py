"""Cold Email Generator Application."""

import logging
from typing import List, Dict, Optional

import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from portfolio import Portfolio
from utils import clean_text

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmailGenerator:
    """Handles email generation workflow."""
    
    def __init__(self, llm: Chain, portfolio: Portfolio):
        self.llm = llm
        self.portfolio = portfolio
        self._portfolio_loaded = False
    
    def load_web_content(self, url: str) -> str:
        """Fetch and clean webpage content."""
        if not url or not url.strip():
            raise ValueError("URL cannot be empty")
        
        try:
            logger.info(f"Loading: {url}")
            loader = WebBaseLoader([url])
            docs = loader.load()
            
            if not docs:
                raise ValueError("No content loaded from URL")
            
            content = clean_text(docs[0].page_content)
            logger.info(f"Loaded {len(content)} characters")
            return content
        except Exception as e:
            logger.error(f"Load failed: {str(e)}")
            raise ConnectionError(f"Unable to fetch URL: {str(e)}")
    
    def generate_emails(self, jobs: List[Dict]) -> List[str]:
        """Generate cold emails for jobs."""
        if not jobs:
            raise ValueError("No jobs to process")
        
        if not self._portfolio_loaded:
            self.portfolio.load_portfolio()
            self._portfolio_loaded = True
        
        logger.info(f"Generating {len(jobs)} email(s)")
        emails = []
        
        for idx, job in enumerate(jobs, 1):
            try:
                skills = job.get("skills", [])
                links = self.portfolio.query_links(skills)
                email = self.llm.write_mail(job, links)
                emails.append(email)
                logger.info(f"Email {idx} generated")
            except Exception as e:
                logger.error(f"Email {idx} failed: {str(e)}")
                emails.append(f"Error: {str(e)}")
        
        return emails
    
    def process_url(self, url: str) -> List[str]:
        """Process URL and generate emails."""
        content = self.load_web_content(url)
        jobs = self.llm.extract_jobs(content)
        
        if not jobs:
            raise ValueError("No jobs found at URL")
        
        return self.generate_emails(jobs)


def init_session():
    """Initialize session state."""
    if 'generator' not in st.session_state:
        st.session_state.generator = EmailGenerator(Chain(), Portfolio())


def render_header():
    """Render app header."""
    st.title("📧 Cold Email Generator")
    st.markdown(
        "Generate personalized cold emails from job postings. "
        "Paste a URL and let AI create tailored emails."
    )
    st.divider()


def render_input() -> Optional[str]:
    """Render URL input."""
    with st.form("url_form"):
        url = st.text_input(
            "Job Posting URL",
            placeholder="https://example.com/careers",
            help="Enter job posting page URL"
        )
        submit = st.form_submit_button(
            "Generate Emails",
            use_container_width=True,
            type="primary"
        )
        return url.strip() if (submit and url) else None


def render_emails(emails: List[str]):
    """Display generated emails."""
    st.success(f"✅ Generated {len(emails)} email(s)")
    
    for idx, email in enumerate(emails, 1):
        with st.expander(f"📧 Email {idx}", expanded=(idx == 1)):
            st.markdown(email)
            st.download_button(
                "Download Email",
                email,
                f"email_{idx}.txt",
                "text/plain",
                key=f"dl_{idx}"
            )


def handle_error(error: Exception):
    """Handle and display errors."""
    error_msg = str(error)
    
    if isinstance(error, ValueError):
        logger.warning(f"Validation: {error_msg}")
        st.error(f"⚠️ Invalid Input: {error_msg}")
    elif isinstance(error, ConnectionError):
        logger.error(f"Connection: {error_msg}")
        st.error(f"🌐 Connection Error: {error_msg}")
    else:
        logger.exception("Unexpected error")
        st.error(f"❌ Error: {error_msg}")
        st.info("Please try again or contact support.")


def process_submission(url: str, generator: EmailGenerator):
    """Process URL and show results."""
    with st.spinner("🔄 Processing..."):
        try:
            emails = generator.process_url(url)
            render_emails(emails)
        except Exception as e:
            handle_error(e)


def main():
    """Main application entry."""
    st.set_page_config(
        layout="wide",
        page_title="Cold Email Generator",
        page_icon="📧",
        initial_sidebar_state="collapsed"
    )
    
    init_session()
    render_header()
    
    url = render_input()
    if url:
        process_submission(url, st.session_state.generator)
    
    st.divider()
    st.caption("Built with Streamlit & LangChain | Powered by AI")


if __name__ == "__main__":
    main()