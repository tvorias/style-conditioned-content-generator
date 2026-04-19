# AbbVie Social Media Chatbot

A hybrid AI-powered chatbot that generates social media content in AbbVie's voice and answers analytical questions about their social media dataset.

## Overview

This project uses a **dual-mode approach:**
1. **Semantic Search** (ChromaDB) for content generation - learns from similar examples
2. **Direct SQL Generation** (SQLite) for analytical queries

The system intelligently routes user requests based on detected intent, providing both creative content generation and data-driven insights.

---

## Project Scripts

### 1. `clean_data.ipynb`
**Purposes:** Standardizes and cleans topic labels in the source Excel data.

**What it does:**
- Loads `ref/AbbVie AI Data Collection.xlsx`
- Fixes typos and inconsistencies in topic labels (e.g. "compant" -> "company", "conferenes" -> "conferences")
- Standardizes terminology (e.g., "and" -> "&")
- Preserves all topics tags as comma-separated lists
- Outputs cleaned data to `ref/AbbVie_AI_Data_Collection_cleaned.xlsx

**Output:** Cleaned Excel file with standardized topics across Press Releases and Twitter sheets.

### 2. `build_chroma_db.py`
**Purpose:** Creates a ChromaDB vector database for semantic search and content generation.

**What it does:**
- Loads cleaned Excel data
- Creates a ChromaDB collection names `abbvie_social_media`
- Stores press releases with title + body text
- Stores tweets engagement metrics (like, shares, comments)
- Adds metadata: content_type, date, topics, audience, links
- Enables semantic similarity search for finding style examples

**Output:** `./chroma_db/` directory containing the vector database (tweets + press releases).

### 3. `test_ollama_models.ipynb`
**Purposes:** Evaluates differeent Ollama models to find the best fit for AbbVie content generation.

**What it does:**
- Tests multiple LLM models: `qwen2.5:14b`, `llama3.1:8b`, `mistral:7b`
- Runs test prompts for tweets and press releases
- Evaluates outputs on:
  - Length appropriateness
  - Professional tone
  - Hashtag usage (for tweets)
- Loads high-engagement tweets from ChromaDB as reference examples
- Generates comparison scores and recommendations

### 4. `revised_app.py`
**Purpose:** The primary Streamlit chatbot interface combining semantic search and SQL query generation.

## Detailed Functionality

### Architecture

The app uses a **hybrid intelligent routing system:**
```
                                                  Tweet/Press Release -> Semantic Search (ChromaDB) -> Content generation (llama3.1:8b) -> generated content
User Input -> Intent Detection (llama3.1:8b) ->
                                                  Question -> SQL Generation (deepseek-coder:6.7b) -> Query Execution (SQLite) -> Formatted Answer


### Core Components

#### 1. **Intent Detextion** (`detect_intent()`)
- Uses `llama3.1:8b` with low temperature (0.1) for consistent classification
- Routes requests to one of three modes:
  - `tweet` - Generate short social media post
  - `press_release` - Generate formal press release
  - `question` - Answer analytical query

#### 2. **Semantic Search Content Generation** (`generate_content()`)
- **When used:** Tweet or press release generation requests
- **How it works:**
  1. Queries ChromaDB for similar content by content_type
  2. Retrieves top 15 most similar examples
  3. Uses top 10 examples as style reference
  4. Feeds examples + user request to  `llama3.1:8b`
  5. Generates content matching AbbVie's voice and style

- **System prompts ensure:**
  - Tweets: <280 characters, 1-4 hashtags, professional yet accessible tone
  - Press releases: Formal tone, patient impact focus, clinical significance

#### 3. **SQL Query Generation (`answer_question()`)
- **When used:** Analytical questions about the dataset
- **How it works:**
  1. Generate SQL using `deepseek-coder:6.7b` (code-specialized model)
  2. Provides detailed schema with critical rules
  3. Executes generated SQL against SQLite database
  4. Formats results using LLM for natural language response
  5. Shows generated SQL query in expandable section

#### 4. **Database Initialization** (`initialize_sql_db()`)
- Auto-creates SQLite database from cleaned Excel file
- Combines Press Releases and Twitter sheets into single  `documents` table
- Handles schema migrations (recreates if old schema detected)
- Cached using `@st.cache_resource` for fast app startup

### User Interface

#### Main Chat Interface
- Real-time message streaming
- Intent indicator show routing decision
- Expandable SQL query viewer for transparency
- Persistent chat history across interactions

#### Sidebar Features
- **Example prompts:** One-click buttons for common user prompts to support demo presentation






