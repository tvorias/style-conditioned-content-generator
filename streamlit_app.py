""" Social Media Chatbot (tailored to the style of AbbVie's social media presence)
"""

import streamlit as st
import chromadb
import ollama
from datetime import datetime

# Configuration
CHROMA_DB_PATH = './chroma_db'
MODEL_NAME = 'llama3.1:8b'
PASSWORD = 'abbvie'


# System prompt to guide chatbot behavior based on type of text user wants generated
SYSTEM_PROMPT = {
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

                        Provide factual, data-driven answers. Cite specific numbers or examples when relevant.
                        """
        }

@st.cache_data
def get_unique_topics_and_audiences(_collection):
    """Get all unique topics and audiences from database"""
    try:
        all_docs = _collection.get(limit=1500)
        topics = set()
        audiences = set()

        for metadata in all_docs['metadatas']:
            if metadata.get('topics'):
                topic_list = [t.strip() for t in metadata['topics'].split(',')]
                topics.update(topic_list)
            if metadata.get('audiences'):
                audiences.add(metadata['audiences'])
        
        return sorted(list(topics)), sorted(list(audiences))
    except Exception as e:
        return [], []


def initialize_connection():
    """Initialize Chroma and Ollama"""
    
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection('abbvie_social_media')

        ollama_client = ollama.Client()

        return collection, ollama_client
    except Exception as e:
        st.error(f"Error initializing connections: {e}")
        st.info("Please ensure ChromaDB is running and Ollama is accessible.")
        st.stop()

def detect_intent(ollama_client, user_input):
    """Use LLM to detect what the user wants to do: generate a tweet, press release, or ask a question about the dataset"""

    intent_prompt = f"""Classify the following user input into exactly ONE of these categories:
    - tweet: User wants to generate/create a short social media post (Twitter/X post)
    - press_release: User want to generate/create a formal press release or announcement
    - question: User is asking a question about data, wants information, or wants analysis based on the dataset

    User input: "{user_input}"

    Respond with ONLY one word: tweet, press_release, or question"""

    try:
        response = ollama_client.generate(
            model=MODEL_NAME,
            prompt=intent_prompt,
            options={
                'temperature': 0.1,
                'max_tokens': 10
            }
        )

        intent = response['response'].strip().lower()

        if 'tweet' in intent:
            return 'tweet'
        elif 'press_release' in intent or 'press release' in intent:
            return 'press_release'
        else:
            return 'question'
    except Exception as e:
        st.warning(f"Intent detection using LLM failed, using fallback method")
        message_lower = user_input.lower()
        if 'tweet' in message_lower or 'post' in message_lower:
            return 'tweet'  
        elif 'press release' in message_lower or 'announcement' in message_lower:
            return 'press_release'
        else:
            return 'question'
        

def retrieve_relevant_content(collection, query, content_type=None, n_results=5):
    """Retrieve relevant content from Chroma"""

    try:
        where_filter = {}
        if content_type == 'tweet':
            where_filter = {'content_type': 'tweet'}
        elif content_type == 'press_release':
            where_filter = {'content_type': 'press_release'}
        
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where_filter=where_filter if where_filter else None
        )
        return results
    except Exception as e:
        st.error(f"Error retrieving content from ChromaDB: {e}")
        return None

def generate_response(ollama_client, intent, user_message, retrieved_context)