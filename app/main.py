import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from chains import Chain
from portfolio import Portfolio
from utils import clean_text
from typing import List, Dict


def load_web_content(url: str) -> str:
    """Fetch and clean webpage text from a given URL.."""
    "just checking "
    loader = WebBaseLoader([url])
    return clean_text(loader.load().pop().page_content)


def generate_emails(jobs: List[Dict], portfolio: Portfolio, llm: Chain) -> List[str]:
    """Generate cold emails for a list of extracted jobs."""
    emails = []
    for job in jobs:
        skills = job.get("skills", [])
        links = portfolio.query_links(skills)
        emails.append(llm.write_mail(job, links))
    return emails


def app_ui(llm: Chain, portfolio: Portfolio):
    """Streamlit UI for cold email generation."""
    st.title("Cold Mail Generator")
    url_input = st.text_input("Paste a URL here:")
    
    if st.button("Submit") and url_input:
        try:
            data = load_web_content(url_input)
            portfolio.load_portfolio()
            jobs = llm.extract_jobs(data)
            emails = generate_emails(jobs, portfolio, llm)

            for email in emails:
                st.code(email, language="markdown")

        except ValueError as ve:
            st.error(f"Invalid Input: {ve}")
        except Exception as e:
            st.error(f"Unexpected Error: {e}")


if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    app_ui(Chain(), Portfolio())
