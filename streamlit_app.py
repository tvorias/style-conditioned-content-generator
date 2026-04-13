""" Social Media Chatbot (tailored to the style of AbbVie's social media presence)
"""

import streamlit as st
import chromadb
import ollama
from datetime import datetime
import json

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

FILTER_DATABASE_TOOL = {
    "name": "filter_database",
    "description": "Filter the AbbVie social media database by topics and/or audiences to get relevant examples. Use this when you find content about specific topics or for specific audiences.",
    "parameters": {
        "type": "object",
        "properties": {
            "topics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of topics to filter by (eg. ['clinical trial', 'research & publications'])"
            },
            "audience": {
                "type": "array",
                "items": {'type': 'string'},
                "description": "List of audiences to filter by (e.g., ['patients/caregivers', 'investor'])"
            },
            "content_type": {
                "type": "string",
                "enum": ["tweet", "press_release", "any"],
                "description": "Type of content to retrieve"
            }
        },
        "required": []
    }
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

        # Get unique topics and audiences 
        unique_topics, unique_audiences = get_unique_topics_and_audiences(collection)

        return collection, ollama_client, unique_topics, unique_audiences
    except Exception as e:
        st.error(f"Error initializing connections: {e}")
        st.info("Please ensure ChromaDB is running and Ollama is accessible.")
        st.stop()

def filter_database_tool(collection, topics=None, audiences=None, content_type="any", n_results=100):
    """Tool function to filter database by topics and audiences"""
    try:
        where_filter_base = {}

        if content_type == "tweet":
            where_filter_base = {"content_type": "tweet"}
        elif content_type == "press_release":
            where_filter_base = {"content_type": "press_release"}

        # retrieve relevant documents based on base filter
        all_docs = collection.get(
            limit=1500, 
            where=where_filter_base if where_filter_base else None
            )
        
        filtered_indices = []
        for i, metadata in enumerate(all_docs['metadatas']):
            match = False

            if topics:
                db_topics = [t.strip().lower() for t in metadata.get('topics', '').split(',')]
                
                # Check if any of the provided topics match the document's topics
                if any(req_topic.lower() in db_topics for req_topic in topics):
                    match = True

            if audiences:
                db_audience = metadata.get('audiences', '').strip().lower()
                if db_audience in audiences:
                    match = True

            # If no filters specified, include all documents
            if not topics and not audiences:
                match = True

            if match:
                filtered_indices.append(i)
            
            # Get filtered results
        if filtered_indices:
            filtered_docs = [all_docs['documents'][i] for i in filtered_indices[:n_results]]
            filtered_metas = [all_docs['metadatas'][i] for i in filtered_indices[:n_results]]
            filtered_ids = [all_docs['ids'][i] for i in filtered_indices[:n_results]]

            results = {
                'documents': [filtered_docs],
                'metadatas': [filtered_metas],
                'ids': [filtered_ids]
            }
        else:
            results = collection.get(
                limit=n_results, 
                where=where_filter_base if where_filter_base else None
            )

            if results and results['documents']:
                results = {
                    'documents': [results['documents']],
                    'metadatas': [results['metadatas']],
                    'ids': [results['ids']]
                }
            
        return results
        
    except Exception as e:
        st.error(f"Error in filter_database_tool: {e}")
        return None


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
        
def use_tool_calling(ollama_client, collection, user_input, intent, unique_topics, unique_audiences):
    """Use LLM with tool calling to filter database and generate response."""
    
    content_type = 'tweet' if intent == 'tweet' else 'press_release' if intent == 'press_release' else 'any'

    tool_prompt = f"""You are an AI assistant that can filter the AbbVie social media database to find relevant examples based on topics and audiences.

        User request: "{user_input}"
        
        Available topics in database
        {', '.join(unique_topics)}

        Available audiences in database:
        {', '.join(unique_audiences)}

        You have access to a tool called "filter_database" that can filter content by topics and audiences.

        Task:
        1. Determine which topics and audience are relevant to the user's request
        2. Call the filter_database tool with appropriate parameters
        3. Use the retrieved content to fulfill the user's request

        To call the tool, respond in this JSON format:
        {{
            "tool": "filter_database",
            "parameters": {{
                "topics": [list of relevant topics from database],
                "audience": [relevant audience from database],
                "content_type": "{content_type}"
            }}
        }}

        If no specific topics/audiences are relevant, use empty arrays: "topics": [], "audience": []"""
    
    try:
        response = ollama_client.generate(
            model=MODEL_NAME,
            prompt=tool_prompt,
            options={'temperature': 0.2}
        )

        response_text = response['response'].strip()

        # Attempt to parse the response as JSON
        try:
            json_start = response_text.find('{')
            json_end  = response_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                tool_call = json.loads(json_str)

                params = tool_call.get('parameters', {})
                topics = params.get('topics', [])
                audiences = params.get('audience', [])

                st.info(f"LLM identified relevant topics: {topics} and audiences: {audiences}")

                filtered_results  = filter_database_tool(
                    collection, 
                    topics=topics if topics else None, 
                    audiences=audiences if audiences else None, 
                    content_type=content_type,
                    n_results=100
                )

                return filtered_results, topics, audiences
            
        except json.JSONDecodeError:
            st.warning("LLM response could not be parsed as JSON, using default retrieval without filters")
            
        filtered_results = filter_database_tool(collection, content_type=content_type, n_results=100) 
        return filtered_results, [], []
    
    except Exception as e:
        st.error(f"Error during tool calling: {e}")
        return None, [], []


def generate_response(ollama_client, intent, user_input, retrieved_context):
    """Generate final response using retrieved context"""

    system_promt = SYSTEM_PROMPT.get(intent, SYSTEM_PROMPT['question'])

    if intent in ['tweet', 'press_release']:
        context_str = "\n\n---\n\n".join([
            f"Example {i+1}:\n{doc[:500]}"
            for i, doc in enumerate(retrieved_context['documents'][0][:10])
        ])

        full_prompt = f"""{system_promt}

        Here are examples of AbbVie's style:
        {context_str}

        User request: "{user_input}"

        Generated content:"""

    else:
        context_parts = []

        for i, (doc, meta) in enumerate(zip(retrieved_context['documents'][0][:10], 
                                            retrieved_context['metadatas'][0][:10])):
            content_type = meta.get('content_type', 'unknown')

            if content_type == 'tweet':
                context_parts.append(
                    f"Tweet {i+1}:\n"
                    f"Date: {meta.get('date', 'N/A')}, Topics: {meta.get('topics', 'N/A')}, "
                    f"Audience: {meta.get('audiences', 'N/A')}, "
                    f"Engagement: {meta.get('total_engagement', 0)} "
                    f"(likes: {meta.get('likes', 0)}, shares: {meta.get('shares', 0)}, comments: {meta.get('comments', 0)})\n"
                    f"Text: {doc[:500]}"
                )
            else:
                context_parts.append(
                    f"Press Release {i+1}:\n"
                    f"Title: {meta.get('title', 'N/A')}, Date: {meta.get('date', 'N/A')}, "
                    f"Topics: {meta.get('topics', 'N/A')}, Audience: {meta.get('audiences', 'N/A')}\n"
                    f"Text: {doc[:1000]}"
                )
            
        context_str = "\n\n".join(context_parts)
        full_prompt = f"""{system_promt}

        Database context:
        {context_str}

        User question: {user_input}
        Answer:"""

    try:
        response = ollama_client.generate(
            model=MODEL_NAME,
            prompt=full_prompt,
            options={'temperature': 0.2,
                     'top_p': 0.9,}
        )

        return response['response'].strip()
    except Exception as e:
        return f"Error generating response: {e}"
    
def main():
    st.set_page_config(
        page_title='AbbVie Social Media Content Generator',
        layout="wide"
    )

    collection, ollama_client, unique_topics, unique_audiences = initialize_connection()

    st.title("AbbVie Social Media Content Generator")
    st.markdown("""
                Ask me to:
                - **Generate a tweet** (e.g. "Write a tweet about clinical trial diversity")
                - **Generate a press release** (e.g. "Write a press release announcing new research results")
                - **Answer questions about AbbVie's social media content** (e.g. "What tweets have the highest engagement?")
                
                """)
    
    if 'message' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if "metadata" in message and message["metadata"]:
                meta = message["metadata"]
                if meta.get("intent") == "tweet":
                    char_count = len(message["content"])
                    if char_count <= 280:
                        st.caption(f"{char_count}/280 characters")
                    else:
                        st.caption(f"{char_count}/280 characters (exceeds limit!)")
                
                if meta.get("filtered_topics") or meta.get("filtered_audiences"):
                    st.caption(f"Filtered by topics: {meta.get('filtered_topics', [])} and audiences: {meta.get('filtered_audiences', [])}")
    
    if prompt := st.chat_input("Ask a question or request content generation..."):
        st.session_start.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                intent = detect_intent(ollama_client, prompt)
                retrieved_context, filtered_topics, filtered_audiences = use_tool_calling(
                    ollama_client, 
                    collection, 
                    prompt, 
                    intent, 
                    unique_topics, 
                    unique_audiences
                )

                if retrieved_context and retrieved_context['documents'][0]:
                    response = generate_response(ollama_client, intent, prompt, retrieved_context)
                    st.markdown(response)

                    metadata = {
                        "intent": intent,
                        "filtered_topics": filtered_topics,
                        "filtered_audiences": filtered_audiences
                    }

                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response, 
                        "metadata": metadata
                    })

                    if intent == "tweet":
                        char_count = len(message["content"])
                        if char_count <= 280:
                            st.caption(f"{char_count}/280 characters")
                        else:
                            st.caption(f"{char_count}/280 characters (exceeds limit!)")

                    if filtered_topics or filtered_audiences:
                        st.caption(f"Filtered by topics: {filtered_topics} and audiences: {filtered_audiences}")

                else:
                    error = "Sorry, I couldn't retrieve relevant information from the database."
                    st.error(error)
                    st.session_state.messages.append({"role": "assistant", "content": error})

    # Additional Information
    with st.sidebar:
        st.header("About")
        st.markdown(f"""
                    **Model:** {MODEL_NAME}

                    **Database:** {collection.count()} documents

                    **Common topics:**
                    {chr(10).join(f'- {t}' for t in unique_topics)}

                    **Common audiences to target:**
                    {chr(10).join(f'- {a}' for a in unique_audiences)}
                    """)
        
        st.markdown("---")

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")

if __name__ == "__main__":
    main()