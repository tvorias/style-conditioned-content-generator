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

@st.cache_resource
def initialize_connections():
    """Initialize ChromaDb, Ollama, and SQL db"""

    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = chroma_client.get_collection('abbvie_social_media')

        ollama_client = ollama.Client()

        sql_db = initialize_sql_db()

        # LangChain Ollama LLM for SQL questies
        sql_llm = Ollama(model=SQL_MODEL, temperature=0)

        sql_agent = create_sql_agent(
            llm=sql_llm,
            db=sql_db,
            agent_type="openai-tools",
            verbose=True,
            handle_parsing_errors=True
        )

        return collection, ollama_client, sql_agent
    
    except Exception as e:
        st.error(f"Error initializing connextions: {e}")
        st.stop()
    
def detect_intent(ollama_client, user_input):
    """Detect if user want to do content generation or has a question about the data"""

    intent_prompt = f"""Classify the following user input into exactly ONE of these categories:
    - tweet: User wants to generate/create a short social media post (Twitter/X post)
    - press_release: User wants to generate/create a formal press release or announcement
    - question: User is asking a question about data, wants information, or wants analysis based on the dataset

    User input: "{user_input}"

    Respond with ONLY one word: tweet, press_release, or question"""

    try:
        response = ollama_client.generate(
            model=INTENT_MODEL,
            prompt=intent_prompt,
            options={'temperature': 0.1, 'max_tokens': 10}
        )

        intent = response['response'].strip().lower()

        if 'tweet' in intent:
            return 'tweet'
        elif 'press_release' in intent or 'press release' in intent:
            return 'press_release'
        else:
            return 'question'
    
    except Exception as e:
        return 'question'


def generate_content(collection, ollama_client, intent, user_input):
    """Generate content using semantic search with Chromadb for style examples"""

    system_prompt = SYSTEM_PROMPTS[intent]

    # Apply content_filter so only relevant examples are pulled
    try:
        results = collection.query(
            query_texts=[user_input],
            where={"content_type": intent},
            n_results=10
        )

        if not results or not results['documents'] or not results['documents'][0]:
            st.warning(f"No {intent} examples found")
            return f"Sorry, I couldn't find relevant {'tweets' if intent == 'tweet' else 'press releases'} examples to learn from"
        
        
        # Build content from retrieved examples
        examples = []
        for i, doc in enumerate(results['documents'][0][:15]):
            examples.append(f"Example {i+1}: \n{doc[:750]}")

        context_str = "\n\n---\n\n".join(examples)

        full_prompt = f"""{system_prompt}

Here are examples of AbbVie's style for {'tweets' if intent == 'tweet' else 'press releases'}

{context_str}

User request: {user_input}

Generate content:"""
        
        response = ollama_client.generate(
            model=CONTENT_MODEL,
            prompt=full_prompt,
            options={
                'temperature': 0.5,
                'top_p': 0.9
            }
        )

        return response['response']
    
    except Exception as e:
        return "Error generating content: {e}"
    
def answer_question(sql_agent, user_input):
    """Answer analytical questions using SQL agent"""

    try:
        prompt = f"""Answer this question about AbbVie's social media data:

"{user_input}"

Database schema:
- tweets table: id, date, text, link, likes, shares, comments, total_engagement, topics, audience, content_type
- press_releases table: id, title, date, text, link, topics, audience, content_type

Use SQL queries to find the answer. Be specific and include relevant details. If you don't know the answer, DO NOT make up answers.answer_question
"""
        response = sql_agent.invoke(prompt)

        if isinstance(response, dict) and 'output' in response:
            return response['output']
        else:
            return str(response)
        
    except Exception as e:
        return f"Error querying database: {e}"
    
def main():
    st.set_page_config(
        page_title="AbbVie Content Generator",
        layout="wide"
    )

    collection, ollama_client, sql_agent = initialize_connections()

    st.title("AbbVie Social Media Content Generator and Chatbot")
    st.markdown("""
    Ask me to:
    - **Generate a tweet** (e.g. "Write a tweet about clinical trial diversity")
    - **Generate a press release** (e.g. "Write a press release announcing new research results")
    - **Answer questions about AbbVie's social media content** (e.g. "What tweets have the highest engagement?")
                """)
    
    # Begin chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    
    if prompt := st.chat_input("Ask a question or request content generation..."):
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                intent = detect_intent(ollama_client, prompt)

                if intent in ['tweet', 'press_release']:
                    st.caption(f"Using semantic search to find {'tweet' if intent == 'tweet' else 'press release'} examples")
                    
                    response = generate_content(
                        collection,
                        ollama_client,
                        intent,
                        prompt
                    )

                else:
                    # If not content generation, proceed to answer question
                    st.caption("Querying database...")
                    
                    response = answer_question(sql_agent, prompt)

                st.markdown(response)

                metadata = {'intent': intent}

                st.session_state.messages.append({
                    "role": 'assistant',
                    "content": response,
                    "metadata": metadata
                })

    
    # add side bar with clear button
    with st.sidebar:
        if st.button('Clear Chat'):
            st.session_state.message = []
            st.rerun()

if __name__ == '__main__':
    main()



