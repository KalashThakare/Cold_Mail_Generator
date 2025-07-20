# Cold Mail Generator

An AI-powered application that automatically generates personalized cold emails for job applications by analyzing job postings and matching them with relevant portfolio projects.

## 🎯 Problem Statement

Writing personalized cold emails for job applications is time-consuming and often generic. This project solves that by automatically:
- Reading job requirements from web postings
- Identifying required skills and technologies
- Matching them with your portfolio projects
- Generating tailored cold emails with relevant portfolio attachments

## ✨ Features

- **Web Scraping**: Extract job requirements directly from job posting URLs
- **AI-Powered Analysis**: Uses LangChain and Groq to understand job requirements
- **Smart Portfolio Matching**: ChromaDB vector database queries to find best matching projects
- **Automated Email Generation**: Creates personalized cold emails with relevant portfolio links
- **Streamlit UI**: User-friendly web interface for easy interaction

## 🛠️ Technologies Used

- **LangChain**: Framework for building LLM applications
- **LangChain Community**: Additional LangChain integrations
- **LangChain-Groq**: Fast inference with Groq API
- **Unstructured**: Document processing and parsing
- **WebBaseLoader**: Web scraping and automation
- **ChromaDB**: Vector database for portfolio storage and retrieval
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Python-dotenv**: Environment variable management

## 📁 Project Structure

```
Cold_mail_generator/
├── .devcontainer/
├── app/
│   ├── __pycache__/
│   └── resources/
│       ├── my_portfolio.csv
│       ├── chains.py
│       ├── main.py
│       ├── portfolio.py
│       └── utils.py
├── VectorStore/
│   ├── 31259f34... (ChromaDB files)
│   └── chroma.sqlite3
├── .env
├── .gitignore
├── Cold_mail_generator.ipynb
└── requirement.txt
```

## 🚀 How It Works

1. **Input**: User pastes a job posting URL
2. **Scraping**: WebBaseLoader extracts the job description and requirements
3. **Analysis**: LangChain processes the text to identify key skills and technologies
4. **Matching**: ChromaDB finds the most relevant portfolio projects based on required skills
5. **Generation**: AI generates a personalized cold email incorporating matched portfolio items
6. **Output**: User receives a tailored email ready to send

## 📋 Prerequisites

- Python 3.8+
- Groq API key
- Chrome browser (for Selenium)

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/KalashThakare/Cold_Mail_Generator.git
cd Cold_Mail_Generator
```

2. Install dependencies:
```bash
pip install -r requirement.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Add your Groq API key to .env file
```

4. Run the application:
```bash
streamlit run app/main.py
```

## 💡 Usage

1. Open the Streamlit web interface
2. Paste the job posting URL in the input field
3. Click "Generate Cold Email"
4. Review the generated email with matched portfolio projects
5. Copy and customize as needed before sending

## 🎓 Learning Outcomes

This project provided hands-on experience with:
- Building end-to-end AI applications with LangChain
- Implementing vector databases for semantic search
- Web scraping with WebBaseLoader
- Creating interactive web apps with Streamlit
- Integrating multiple AI/ML technologies in a cohesive solution

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!
