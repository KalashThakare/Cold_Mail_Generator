"""Cold Email Generator Application."""

import logging
from typing import List, Dict, Optional

import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from portfolio import Portfolio
from utils import clean_text

# ──────────────────────────────────────────────
# Logging Configuration
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("ColdEmailApp")


# ──────────────────────────────────────────────
# Core Application Logic
# ──────────────────────────────────────────────
class EmailGenerator:
    """Manages email generation workflow from job URLs."""

    def __init__(self, llm: Chain, portfolio: Portfolio):
        self.llm = llm
        self.portfolio = portfolio
        self._portfolio_loaded = False

    def _ensure_portfolio_loaded(self) -> None:
        """Load portfolio into vector DB if not already loaded."""
        if not self._portfolio_loaded:
            logger.info("Loading portfolio into vector store...")
            self.portfolio.load_portfolio()
            self._portfolio_loaded = True

    def load_web_content(self, url: str) -> str:
        """Fetch and clean webpage content."""
        if not url or not url.strip():
            raise ValueError("URL cannot be empty.")

        try:
            loader = WebBaseLoader([url])
            docs = loader.load()
            if not docs:
                raise ValueError("No content fetched from the provided URL.")
            content = clean_text(docs[0].page_content)
            logger.info(f"Fetched {len(content)} characters from {url}")
            return content
        except Exception as e:
            logger.error(f"Error loading content from {url}: {e}")
            raise ConnectionError("Failed to fetch or clean webpage content.") from e

    def generate_emails(self, jobs: List[Dict]) -> List[str]:
        """Generate personalized cold emails for provided job data."""
        if not jobs:
            raise ValueError("Job list is empty or invalid.")

        self._ensure_portfolio_loaded()
        logger.info(f"Generating emails for {len(jobs)} job(s)...")

        emails = []
        for i, job in enumerate(jobs, start=1):
            try:
                links = self.portfolio.query_links(job.get("skills", []))
                email = self.llm.write_mail(job, links)
                emails.append(email)
                logger.info(f"Email {i} generated successfully.")
            except Exception as e:
                logger.error(f"Email generation failed for job {i}: {e}")
                emails.append(f"Error generating email: {e}")
        return emails

    def process_url(self, url: str) -> List[str]:
        """Process job postings from URL and generate emails."""
        content = self.load_web_content(url)
        jobs = self.llm.extract_jobs(content)
        if not jobs:
            raise ValueError("No valid jobs found on the given page.")
        return self.generate_emails(jobs)


# ──────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────
def init_session() -> None:
    """Initialize Streamlit session state."""
    if "generator" not in st.session_state:
        st.session_state.generator = EmailGenerator(Chain(), Portfolio())


def render_header() -> None:
    """Render the main header and description."""
    st.title("📧 Cold Email Generator")
    st.markdown(
        "AI-powered tool to create personalized cold emails from job postings.\n"
        "Just paste a job URL and let AI do the rest."
    )
    st.divider()


def render_input() -> Optional[str]:
    """Render and handle job posting URL input."""
    with st.form("url_form"):
        url = st.text_input(
            "Job Posting URL",
            placeholder="https://example.com/careers",
        )
        submitted = st.form_submit_button("Generate Emails", use_container_width=True)
        return url.strip() if submitted and url else None


def render_emails(emails: List[str]) -> None:
    """Display generated emails in an expandable format."""
    st.success(f"✅ Generated {len(emails)} email(s).")
    for i, email in enumerate(emails, start=1):
        with st.expander(f"📩 Email {i}", expanded=(i == 1)):
            st.markdown(email)
            st.download_button(
                "Download Email",
                email,
                file_name=f"email_{i}.txt",
                mime="text/plain",
                key=f"dl_{i}"
            )


def handle_error(error: Exception) -> None:
    """Handle user-friendly error messages."""
    msg = str(error)
    if isinstance(error, ValueError):
        st.error(f"⚠️ Input Error: {msg}")
    elif isinstance(error, ConnectionError):
        st.error(f"🌐 Connection Issue: {msg}")
    else:
        st.error(f"❌ Unexpected Error: {msg}")
        st.info("Try again later or contact support.")


def process_submission(url: str, generator: EmailGenerator) -> None:
    """Process user URL input."""
    with st.spinner("🔄 Processing..."):
        try:
            emails = generator.process_url(url)
            render_emails(emails)
        except Exception as e:
            handle_error(e)


def main() -> None:
    """Streamlit app entry point."""
    st.set_page_config(
        page_title="Cold Email Generator",
        page_icon="📧",
        layout="wide",
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
