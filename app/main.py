"""Cold Email Generator Application."""

from typing import List, Dict, Optional
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from portfolio import Portfolio
from utils import clean_text
from logger_utils import get_logger, safe_execution, validate_non_empty

logger = get_logger("ColdEmailApp")


class EmailGenerator:
    """Handles complete cold email generation workflow."""

    def __init__(self, llm: Chain, portfolio: Portfolio):
        self.llm = llm
        self.portfolio = portfolio
        self._portfolio_loaded = False

    def _ensure_portfolio_loaded(self):
        """Load portfolio once for session."""
        if not self._portfolio_loaded:
            logger.info("Loading portfolio into ChromaDB...")
            self.portfolio.load_portfolio()
            self._portfolio_loaded = True

    def load_web_content(self, url: str) -> str:
        """Fetch and clean webpage content."""
        url = validate_non_empty(url, "URL")
        with safe_execution(logger, "load_web_content"):
            loader = WebBaseLoader([url])
            docs = loader.load()
            if not docs:
                raise ValueError("No content fetched from the provided URL.")
            content = clean_text(docs[0].page_content)
            logger.info(f"Fetched {len(content)} characters from {url}")
            return content

    def generate_emails(self, jobs: List[Dict]) -> List[str]:
        """Generate emails for given jobs."""
        if not jobs:
            raise ValueError("No jobs provided for email generation.")
        self._ensure_portfolio_loaded()

        logger.info(f"Generating {len(jobs)} email(s)...")
        emails = []

        for i, job in enumerate(jobs, start=1):
            with safe_execution(logger, f"generate_email_{i}"):
                links = self.portfolio.query_links(job.get("skills", []))
                email = self.llm.write_mail(job, links)
                emails.append(email)
        return emails

    def process_url(self, url: str) -> List[str]:
        """Extract jobs and generate corresponding emails."""
        content = self.load_web_content(url)
        jobs = self.llm.extract_jobs(content)
        if not jobs:
            raise ValueError("No job listings found on the given page.")
        return self.generate_emails(jobs)


# ──────────────── Streamlit UI Layer ────────────────
def init_session():
    if "generator" not in st.session_state:
        st.session_state.generator = EmailGenerator(Chain(), Portfolio())


def render_header():
    st.title("📧 Cold Email Generator")
    st.markdown(
        "AI tool that generates personalized cold emails from job postings.\n"
        "Paste a job URL, and let AI craft the perfect outreach email."
    )
    st.divider()


def render_input() -> Optional[str]:
    with st.form("url_form"):
        url = st.text_input("Job Posting URL", placeholder="https://example.com/careers")
        submitted = st.form_submit_button("Generate Emails", use_container_width=True)
        return url.strip() if submitted and url else None


def render_emails(emails: List[str]):
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


def handle_error(error: Exception):
    msg = str(error)
    if isinstance(error, ValueError):
        st.error(f"⚠️ Input Error: {msg}")
    elif isinstance(error, ConnectionError):
        st.error(f"🌐 Connection Issue: {msg}")
    else:
        st.error(f"❌ Unexpected Error: {msg}")
        st.info("Please try again later.")


def process_submission(url: str, generator: EmailGenerator):
    with st.spinner("🔄 Processing..."):
        try:
            emails = generator.process_url(url)
            render_emails(emails)
        except Exception as e:
            handle_error(e)


def main():
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
