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