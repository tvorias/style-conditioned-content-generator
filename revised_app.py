"""AbbVie Social Media Chatbot
- SQL agent for analytical Questions
- Semantic search for content generation
"""

import streamlit as st
import chromadb
import ollama
import sqlite3
import pandas as pd
from pathlib import Path
from langchain_community.utilities import SQLDatabase 
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.llms.ollama import Ollama

# Model selection for different tasks
INTENT_MODEL = 'llama3.1:8b'
SQL_MODEL = 'deepseek-coder:6.7b'
CONTENT_MODEL = 'qwen2.5:14b'


# Database paths
CHROMA_DB_PATH = './chroma_db'
SQL_DB_PATH = './abbvie_data.db'
EXCEL_FILE = 'ref/AbbVie AI Data Collection_cleaned.xlsx'

# System prompt to guide chatbot behavior based on type of text user wants generated
SYSTEM_PROMPTS = {
    "tweet": """You are a social media writer for AbbVie, a global, research-driven biopharmaceutical company that develops advanced therapies for complex and serious diseases.

Write a professional tween that:
- is under 280 characters
- Includes 1-4 relevant hashtags
- Has an accessible yet professional tone
- Focuses on heathcare innovation, patient care, or research
- Emphasizes patient impact

Write ONLY the tweet text, nothing else.
        """,

    "press_release": """You are a corporate communications writer for AbbVie, a global, research-driven biopharmaceutical company that develops advanced therapies for complex and serious diseases. 

Write a press release opening (2-3 paragraphs) that:
- uses formal, authoritative tone
- leads with key announcement
- emphasizes patient impact and clinical significance
- includes specific details
- Uses professional medical terminology

Write the press release opening ONLY, nothing else.
        """,

        'question': """You are a helpful data analyst.

Answer questions about AbbVie's social media dataset clearly and concisely.

The dataset contains:
- Press releases: titles, dates, topics, audiences, full text
- Tweets: text, dates, topics, audiences, engagement (likes, shares, comments)

Provide factual, data-driven answers. Cite specific numbers or examples when relevant. If you do not know the answer, do not guess.
"""
        }

@st.cache_resource
def initialize_sql_db():
    """Create SQLite database from cleaned data if it doesn't already exist"""

    db_path = Path(SQL_DB_PATH)

    if db_path.exists():
        return SQLDatabase.from_uri(f"sqlite:///{SQL_DB_PATH}")
    
    # Load cleaned Excel data
    xls = pd.ExcelFile(EXCEL_FILE)
    df_pr = pd.read_excel(xls, 'Press releases')
    df_tw = pd.read_excel(xls, 'Twitter')

    # Standardize column and clean names
    df_pr.columns = [col.strip().lower().replace(' ', '_') for col in df_pr.columns]
    df_tw.columns = [col.strip().lower().replace(' ', '_') for col in df_tw.columns]

    df_pr = df_pr.rename(columns={'verbatim': 'text'})
    df_tw = df_tw.rename(columns={'post_copy': 'text'})

    df_pr['content_type'] = 'press_release'
    df_tw['content_type'] = 'tweet'


    # Create db
    conn = sqlite3.connect(SQL_DB_PATH)

    df_pr.to_sql('press_release', conn, if_exists='replace', index=False)
    df_tw.to_sql('tweets', conn, if_exists='replace', index=False)

    conn.close()

    return SQLDatabase.from_uri(f"sqlite:///{SQL_DB_PATH}")
