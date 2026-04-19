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


# Model selection for different tasks
INTENT_MODEL = 'llama3.1:8b'
SQL_MODEL = 'deepseek-coder:6.7b'
CONTENT_MODEL = 'llama3.1:8b'  # 'qwen2.5:14b'


# Database paths
CHROMA_DB_PATH = './chroma_db'
SQL_DB_PATH = './abbvie_data.db'
EXCEL_FILE = 'AbbVie_AI_Data_Collection_cleaned.xlsx'

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
        try:
            conn = sqlite3.connect(SQL_DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table AND name='documents'")
            has_documents = cursor.fetchone() is not None
            conn.close()

            if has_documents:
                return True
            
            st.warning("Old database schema. Recreating...")
            db_path.unlink()
        except Exception:
            if db_path.exists():
                db_path.unlink()


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

    if 'total_engagement' not in df_tw.columns:
        df_tw['total_engagement'] = df_tw['likes'].fillna(0) + df_tw['shares'].fillna(0) + df_tw['comments'].fillna(0)

    df_pr['likes'] = None
    df_pr['shares'] = None
    df_pr['comments'] = None
    df_pr['total_engagement'] = None

    df_tw['title'] = None

    common_columns = ['content_type', 'title', 'text', 'link', 'likes', 'shares', 'comments',
                      'total_engagement', 'topics', 'audience']
    
    df_combined = pd.concat([
        df_pr[common_columns], 
        df_tw[common_columns]
    ], ignore_index=True)


    # Create db
    conn = sqlite3.connect(SQL_DB_PATH)

    df_combined.to_sql('documents', conn, if_exists='replace', index=False)

    conn.close()

    return True

@st.cache_resource
def initialize_connections():
    """Initialize ChromaDb, Ollama"""

    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = chroma_client.get_collection('abbvie_social_media')

        ollama_client = ollama.Client()

        initialize_sql_db()

        return collection, ollama_client
    
    except Exception as e:
        st.error(f"Error initializing connections: {e}")
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
            where={"content_type": intent} #,
            # n_results=10
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
                'temperature': 0.3,
                'top_p': 0.9
            }
        )

        return response['response']
    
    except Exception as e:
        return "Error generating content: {e}"
    
def answer_question(ollama_client, user_input):
    """Answer analytical questions using SQL agent"""

    try:
        prompt = f"""You are a SQL expert. Generate a SQLite query to answer this question:

Question: "{user_input}"

Database schema:
- Table: documents
- columns:
    * content_type (TEXT): 'tweet' or 'press_release'
    * date (TEXT): publication date
    * title (TEXT): press release title (NULL for tweets)
    * text (TEXT): content (tweet text or press release body)
    * link (TEXT): URL
    * likes (INTEGER): number of likes (NULL for press releases)
    * shares (INTEGER): number of shares (NULL for press releases)
    * comments (INTEGER): number of comments (NULL for press releases)
    * total_engagement (INTEGER): sum of engagement metrics (NULL for press releases)
    * topics (TEXT): PRE-CATEGORIZED comma-separated topic tags - USE THIS for "about" queries
    * audience (TEXT): target audience

Important:
- "COUNT(*) FROM documents" should only be used when getting counts or percentages of ALL documents
- "document" means 'tweet' AND 'press_release'
- Use 'WHERE content_type = 'tweet'" ONLY when you need to filter tweets
- Use 'WHERE content_type = 'press_release'" ONLY when you need to filter press releases
- For percentages, the content_filter MUST be the same for calculating the numerator and denominator. There should never be two separate values of WHERE content_filter = X
- Example: "What percent of press releases are about patents?" -> SELECT (*) * 100 / (SELECT COUNT(*) FROM documents WHERE content_filter = "press_release") FROM documents WHERE content_type = "press_release" AND topics LIKE '%patents%'
- To count all documents: SELECT COUNT(*) FROM documents
- Use LIKE for text matching (e.g., topics LIKE '%clinical trial%', text LIKE '%cancer%')
- When users asks "about [topic]" -> search the topics and text fields
- Example: "How many about patents?" -> WHERE topics LIKE '%patents%'
- To count tweets: SELECT COUNT(*) FROM documents WHERE content_type = 'tweet'
- For engagement metrics, always filter tweets only
- Limit results to 10 unless user asks for more

Respond with ONLY the SQL query, no explanations, no markdown formatting.
"""
        sql_response = ollama_client.generate(
            model=SQL_MODEL,
            prompt=prompt,
            options={'temperature': 0, 'max_tokens': 200}
        )

        sql_query = sql_response['response'].strip()

        sql_query = sql_query.replace('```sql', '').replace('```', '').strip()

        sql_query = sql_query.rstrip(';')

        # Display SQL query
        with st.expander("Generated SQL query:", expanded=False):
            st.code(sql_query, language='sql')

        conn = sqlite3.connect(SQL_DB_PATH)
        cursor = conn.cursor()

        cursor.execute(sql_query)
        results = cursor.fetchall()
        column_names = [description[0] for description in cursor.description] if cursor.description else []
        
        conn.close()

        # Format the results
        if not results:
            return "No results found for your query"
    
        if len(results) == 1 and len(results[0]) == 1 and column_names[0].lower().startswith('count'):
            count = results[0][0]
            return f"There are {count} documents matching your query"
        
        if len(results) <= 10:
            results_text = ""
            for row in results:
                row_dict = dict(zip(column_names, row))
                results_text += f"\n- {row_dict}"

            
            interpretation_prompt = f"""User asked: "{user_input}

Query results:
{results_text}

Format this as a clear, concise answer. Include specific numbers and details."""
            
            answer_response = ollama_client.generate(
                model=CONTENT_MODEL,
                prompt=interpretation_prompt,
                options={'temperature': 0.3, 'max_tokens': 400}
            )

            return answer_response['response']
        else:
            return f"Found {len(results)} results. Top 10:\n" + "\n".join([str(dict(zip(column_names, row))) for row in results[:10]])

        
    except sqlite3.Error as e:
        error_message = f"SQL execution error: {e}"
        if 'no such table' in str(e).lower():
            error_message += "\n\nDatabase may need to be recreated"
        return error_message
    except Exception as e:
        return f"Error processing query: {e}"
    
def main():
    st.set_page_config(
        page_title="AbbVie Content Generator",
        layout="wide"
    )

    collection, ollama_client = initialize_connections()

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

    # Check if demo prompt was clicked
    if 'example_prompt' in st.session_state:
        prompt = st.session_state.example_prompt
        del st.session_state.example_prompt
    else:
        prompt = st.chat_input("Ask a question or request content generation...")

    
    if prompt:
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
                    
                    response = answer_question(ollama_client, prompt)

                st.markdown(response)

                metadata = {'intent': intent}

                st.session_state.messages.append({
                    "role": 'assistant',
                    "content": response,
                    "metadata": metadata
                })

    
    # add side bar with clear button
    with st.sidebar:
        st.markdown("---")
        
        # section for example prompts for demo
        st.header("Example Prompts")
        st.caption("Click to try these examples")

        st.subheader("Analytic Questions")
        if st.button("Prevalence of press releases about patents", use_container_width=True):
            st.session_state.example_prompt = "What percent of press releases are about patents?"
            st.rerun()

        if st.button("Top 5 tweets about cancer by total engagement", use_container_width=True):
            st.session_state.example_prompt = 'What are the top 5 tweets about cancer by total engagement?'
            st.rerun()
        
        if st.button("Average engagement for tweets about clinical trials", use_container_width=True):
            st.session_state.example_prompt = 'What is the average engagement for tweets about clinical trials?'
            st.rerun()

        st.subheader("Tweet Generation")

        if st.button("Tweet about sleep issues among those with Parkinson's Disease", use_container_width=True):
            st.session_state.example_prompt = "Write an informational tweet about how 90 percent of people with Parkinson's Disease have trouble sleeping"
            st.rerun()

        if st.button("Tweet about new White House agreement for accessibility and affordability", use_container_width=True):
            st.session_state.example_prompt = "Create a tweet about an agreement with the White House about increasing access and affordability in healthcare in the US."
            st.rerun()

        st.subheader("Press Release Generation")

        if st.button("Clinical trial results", use_container_width=True):
            st.session_state.example_prompt = "Write a press release announcing positive Phase 3 clinical trial results for an oncology therapeutic"
            st.rerun()

        if st.button("Research Partnership", use_container_width=True):
            st.session_state.example_prompt = "Write a press release about a new research partnership with an aesthetics company to evaluate the long-term effects of using Botox as a preventative anti-aging treatment"
            st.rerun()

        st.markdown("---")

        if st.button('Clear Chat', use_container_width=True):
            st.session_state.message = []
            if 'example_prompt' in st.session_state:
                del st.session_state.example_prompt
            st.rerun()

if __name__ == '__main__':
    main()



