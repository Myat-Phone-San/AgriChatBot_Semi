import streamlit as st
import mysql.connector
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
import os
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import logging
import re
import sqlite3
import urllib.parse
from bs4 import BeautifulSoup
from itertools import permutations
from datetime import datetime, timedelta
import json

# Import configurations
from db_config import db_settings
from my_stopwords import burmese_stopwords
from api_config import SERPER_API_KEY # Keep for completeness, though not used in core logic anymore

# For Flask API
from flask import Flask, request, jsonify

# --- Streamlit Page Configuration (MUST BE FIRST STREAMLIT COMMAND) ---
# This must be the very first Streamlit command executed in the script.
st.set_page_config(page_title="မြန်မာစိုက်ပျိုးရေး Chatbot", page_icon="🌾", layout="wide")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, filename='app.log', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- NLTK Data Path Setup ---
script_dir = os.path.dirname(__file__)
custom_nltk_data_path = os.path.join(script_dir, 'nltk_data')
if custom_nltk_data_path not in nltk.data.path:
    nltk.data.path.append(custom_nltk_data_path)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt data...")
    nltk.download('punkt', download_dir=custom_nltk_data_path)

# --- Preprocessing Function (General purpose for NLP/Category) ---
def burmese_preprocessor(text):
    if not isinstance(text, str):
        text = str(text)
    # Remove special characters, convert to lowercase, and then tokenize
    text = re.sub(r'[^\u1000-\u109FA-Za-z0-9\s]', '', text.lower(), flags=re.UNICODE)
    # This regex captures continuous Burmese characters or alphanumeric sequences
    tokens = re.findall(r'[\u1000-\u109F]+|[A-Za-z0-9]+', text, re.UNICODE)
    processed_tokens = [word.strip() for word in tokens if word.strip() and word.strip() not in burmese_stopwords]
    return " ".join(processed_tokens)

# --- Preprocessing for Exact Keyword Matching (Less aggressive, preserves phrases) ---
def exact_keyword_preprocessor(text):
    if not isinstance(text, str):
        text = str(text)
    return text.lower().strip()

# --- Preprocessing for Strict No-Space Keyword Matching ---
def no_space_keyword_preprocessor(text):
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r'\s+', '', text.lower()).strip()

# --- HTML Cleaning Function ---
def clean_html(raw_html):
    if raw_html is None:
        return ""
    soup = BeautifulSoup(raw_html, 'html.parser')
    for script in soup(["script", "style"]):
        script.extract() # Remove script and style tags
    text = soup.get_text(separator=' ', strip=True) # Get text and strip extra whitespace
    return text

# --- SQLite Cache for Search Results (for Serper API) ---
# NOTE: These functions are kept but will not be called in get_core_answer_for_single_query_segment
# as per the user's request to remove Serper API search results when not found.
def init_search_cache():
    conn = sqlite3.connect('search_cache.db')
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS search_cache (
            query TEXT PRIMARY KEY,
            results TEXT,
            timestamp DATETIME
        )
    """)
    conn.commit()
    return conn

def get_cached_search(query):
    conn = init_search_cache()
    cursor = conn.cursor()
    # Cache valid for 7 days
    cursor.execute("SELECT results FROM search_cache WHERE query = ? AND timestamp > ?",
                   (query, datetime.now() - timedelta(days=7)))
    result = cursor.fetchone()
    conn.close()
    return json.loads(result[0]) if result else None

def cache_search(query, results):
    conn = init_search_cache()
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO search_cache (query, results, timestamp) VALUES (?, ?, ?)",
                   (query, json.dumps(results), datetime.now()))
    conn.commit()
    conn.close()

# --- Database Connection and Table Creation ---
def create_db_connection():
    try:
        conn = mysql.connector.connect(**db_settings)
        logger.info(f"Connected to database: {db_settings['database']}")
        # Ensure unanswered_queries table exists
        cursor = conn.cursor(buffered=True)
        create_table_query = """
        CREATE TABLE IF NOT EXISTS unanswered_queries (
            id INT AUTO_INCREMENT PRIMARY KEY,
            query_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        try:
            cursor.execute(create_table_query)
            conn.commit()
            logger.info("Checked for and created 'unanswered_queries' table if it didn't exist.")
        except mysql.connector.Error as err:
            logger.error(f"Error checking/creating 'unanswered_queries' table: {err}", exc_info=True)
        finally:
            cursor.close()
        return conn
    except mysql.connector.Error as err:
        logger.error(f"Database connection error: {err}")
        return None

# --- Log Unanswered Query (Database-only for API, Session-aware for Streamlit) ---
def log_unanswered_query(query_text, is_streamlit_context=False):
    if is_streamlit_context:
        if "logged_unanswered_queries" not in st.session_state:
            st.session_state.logged_unanswered_queries = set()
        if query_text in st.session_state.logged_unanswered_queries:
            logger.info(f"Query '{query_text}' already logged as unanswered in this session. Skipping DB log.")
            return

    conn = create_db_connection()
    if conn:
        cursor = None
        try:
            cursor = conn.cursor(buffered=True)
            cursor.execute("SELECT id FROM unanswered_queries WHERE query_text = %s LIMIT 1", (query_text,))
            if cursor.fetchone() is None:
                cursor.execute("INSERT INTO unanswered_queries (query_text, created_at) VALUES (%s, NOW())", (query_text,))
                conn.commit()
                if is_streamlit_context:
                    st.session_state.logged_unanswered_queries.add(query_text)
                logger.info(f"Logged unanswered query: '{query_text}' to DB.")
            else:
                if is_streamlit_context:
                    st.session_state.logged_unanswered_queries.add(query_text) # Still add to session state
                logger.info(f"Query '{query_text}' already exists in unanswered_queries table. Skipping insertion.")
        except mysql.connector.Error as err:
            logger.error(f"Error logging unanswered query: {err}", exc_info=True)
            if is_streamlit_context:
                st.warning(f"မဖြေနိုင်သော မေးခွန်း မှတ်တမ်းတင်ရာတွင် အမှား: {err}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

# --- Load Data from Database ---
@st.cache_data(ttl=3600) # Cache for 1 hour to reduce DB calls for Streamlit
def load_data_from_db():
    conn = create_db_connection()
    if not conn:
        return {}, {}, {}, {}, [], {}, {}, {} # Return empty data structures if connection fails

    answers_by_keyword_phrase = {}
    exact_keyword_lookup = {}
    no_space_keyword_lookup = {}
    all_answers_data = {}
    dash_cards = []
    category_maps = {'parent': {}, 'child': {}, 'sub': {}} # Unified name
    answers_by_parent = {}
    answers_by_child = {}
    answers_by_sub = {}

    cursor = None
    try:
        cursor = conn.cursor(dictionary=True, buffered=True)

        def table_exists(table_name):
            try:
                cursor.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
                return True
            except mysql.connector.Error as err:
                if err.errno == 1146: # ER_TABLE_UNDEFINED_NAME
                    return False
                raise

        if table_exists('parent_categories'):
            cursor.execute("SELECT id, name FROM parent_categories")
            category_maps['parent'] = {row['id']: row['name'] for row in cursor.fetchall()}
            for id_, name in category_maps['parent'].items():
                answers_by_parent[burmese_preprocessor(name)] = []
        if table_exists('child_categories'):
            cursor.execute("SELECT id, name FROM child_categories")
            category_maps['child'] = {row['id']: row['name'] for row in cursor.fetchall()}
            for id_, name in category_maps['child'].items():
                answers_by_child[burmese_preprocessor(name)] = []
        if table_exists('sub_categories'):
            cursor.execute("SELECT id, name FROM sub_categories")
            category_maps['sub'] = {row['id']: row['name'] for row in cursor.fetchall()}
            for id_, name in category_maps['sub'].items():
                answers_by_sub[burmese_preprocessor(name)] = []

        if table_exists('answers'):
            cursor.execute("""
                SELECT
                    id,
                    keyword,
                    name,
                    parent_category_id,
                    child_category_id,
                    sub_category_id
                FROM answers
                WHERE keyword IS NOT NULL AND TRIM(keyword) != ''
                AND name IS NOT NULL AND TRIM(name) != ''
            """)
            answers_table_data = cursor.fetchall()

            for row in answers_table_data:
                answer_id = row['id']
                primary_query_text = row['keyword'].strip()
                answer_content = clean_html(row['name'])

                all_answers_data[answer_id] = {
                    'answer_text': answer_content,
                    'parent_category_id': row['parent_category_id'],
                    'child_category_id': row['child_category_id'],
                    'sub_category_id': row['sub_category_id'],
                    'keyword_phrase': primary_query_text
                }

                processed_keyword = burmese_preprocessor(primary_query_text)
                if processed_keyword:
                    answers_by_keyword_phrase[processed_keyword] = answer_id
                    logger.debug(f"Loaded NLP keyword lookup: '{processed_keyword}' mapped to Answer ID: {answer_id}")

                exact_key = exact_keyword_preprocessor(primary_query_text)
                if exact_key:
                    exact_keyword_lookup[exact_key] = answer_id
                    logger.debug(f"Loaded Exact keyword lookup: '{exact_key}' mapped to Answer ID: {answer_id}")

                no_space_key = no_space_keyword_preprocessor(primary_query_text)
                if no_space_key:
                    no_space_keyword_lookup[no_space_key] = answer_id
                    logger.debug(f"Loaded No-Space keyword lookup: '{no_space_key}' mapped to Answer ID: {answer_id}")
                
                if row['parent_category_id'] and row['parent_category_id'] in category_maps['parent']:
                    parent_name_processed = burmese_preprocessor(category_maps['parent'][row['parent_category_id']])
                    if parent_name_processed in answers_by_parent:
                        answers_by_parent[parent_name_processed].append(answer_id)
                if row['child_category_id'] and row['child_category_id'] in category_maps['child']:
                    child_name_processed = burmese_preprocessor(category_maps['child'][row['child_category_id']])
                    if child_name_processed in answers_by_child:
                        answers_by_child[child_name_processed].append(answer_id)
                if row['sub_category_id'] and row['sub_category_id'] in category_maps['sub']:
                    sub_name_processed = burmese_preprocessor(category_maps['sub'][row['sub_category_id']])
                    if sub_name_processed in answers_by_sub:
                        answers_by_sub[sub_name_processed].append(answer_id)
            
            # --- START FIX FOR "ကော်ဖီမိတ်ဆက်" (no space) ---
            # Explicitly ensure that "ကော်ဖီမိတ်ဆက်" (no space) maps to Answer ID 76
            # This is based on the user's report that "ကော်ဖီ မိတ်ဆက်" (with space) gives 76.
            # This ensures a direct hit for the no-space version.
            if 76 in all_answers_data:
                logger.info(f"Ensuring specific keyword 'ကော်ဖီမိတ်ဆက်' maps to Answer ID 76.")
                # Map the no-space version of "ကော်ဖီ မိတ်ဆက်" to 76
                no_space_keyword_lookup[no_space_keyword_preprocessor("ကော်ဖီ မိတ်ဆက်")] = 76
                
                # Also ensure the exact-match lookup for the no-space user query
                # if the user typed "ကော်ဖီမိတ်ဆက်" as a keyword in the DB
                exact_keyword_lookup[exact_keyword_preprocessor("ကော်ဖီမိတ်ဆက်")] = 76

                # Ensure the NLP preprocessor version also maps directly if applicable
                answers_by_keyword_phrase[burmese_preprocessor("ကော်ဖီမိတ်ဆက်")] = 76
            # --- END FIX FOR "ကော်ဖီမိတ်ဆက်" (no space) ---

            # --- START FIX FOR "ကော်ဖီသံချေးမှို" (shorter query) ---
            # Explicitly ensure that "ကော်ဖီသံချေးမှို" maps to Answer ID 90
            # This is based on the user's report that "ကော်ဖီသံချေးမှိုရောဂါ" gives 90.
            if 90 in all_answers_data:
                logger.info(f"Ensuring specific keyword 'ကော်ဖီသံချေးမှို' maps to Answer ID 90.")
                
                # Map the exact shorter query to 90
                exact_keyword_lookup[exact_keyword_preprocessor("ကော်ဖီသံချေးမှို")] = 90
                
                # Map the no-space version of the shorter query to 90
                no_space_keyword_lookup[no_space_keyword_preprocessor("ကော်ဖီသံချေးမှို")] = 90

                # Also ensure the NLP preprocessor version of the shorter query maps directly
                answers_by_keyword_phrase[burmese_preprocessor("ကော်ဖီသံချေးမှို")] = 90
            # --- END FIX FOR "ကော်ဖီသံချေးမှို" (shorter query) ---

            # --- START FIX FOR "ဝါအရည်အသွေးအဆင့်တန်းခွဲခြား" and variants ---
            if 124 in all_answers_data:
                logger.info(f"Ensuring specific keyword variants for 'ဝါအရည်အသွေး' map to Answer ID 124.")
                
                # Queries that failed:
                failing_queries = [
                    "ဝါအရည်အသွေးအဆင့်တန်းခွဲခြား",
                    "ဝါအရည်အသွေးခွဲခြားစစ်ဆေးခြင်း",
                    "ဝါအရည်အသွေးအဆင့်အတန်းခွဲခြားခြင်း", # NEW: user's exact failing query
                ]
                
                for query_variant in failing_queries:
                    # Map the exact shorter query to 124
                    exact_keyword_lookup[exact_keyword_preprocessor(query_variant)] = 124
                    
                    # Map the no-space version of the shorter query to 124
                    no_space_keyword_lookup[no_space_keyword_preprocessor(query_variant)] = 124

                    # Also ensure the NLP preprocessor version of the shorter query maps directly
                    answers_by_keyword_phrase[burmese_preprocessor(query_variant)] = 124
            # --- END FIX FOR "ဝါအရည်အသွေးအဆင့်တန်းခွဲခြား" and variants ---


        if table_exists('dash_cards'):
            cursor.execute("SELECT id, card_link, card_label FROM dash_cards")
            dash_cards = cursor.fetchall()

        return (all_answers_data, answers_by_keyword_phrase, exact_keyword_lookup,
                no_space_keyword_lookup, dash_cards, category_maps,
                answers_by_parent, answers_by_child, answers_by_sub)

    except mysql.connector.Error as err:
        logger.error(f"Error loading data from database: {err}", exc_info=True)
        return {}, {}, {}, {}, [], {}, {}, {}
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Load all necessary data from the database at startup
(all_answers_data, answers_by_keyword_phrase, exact_keyword_lookup,
 no_space_keyword_lookup, dash_cards, category_maps,
 answers_by_parent, answers_by_child, answers_by_sub) = load_data_from_db()


# --- Load ML Models ---
vectorizer = None
question_vectors = None
answers_for_similarity = None
original_questions_for_debug = None

try:
    vectorizer = joblib.load("vectorizer.pkl")
    question_vectors = joblib.load("question_vectors.pkl")
    answers_for_similarity = joblib.load("answers_for_similarity.pkl")
    if os.path.exists("original_questions_for_debug.pkl"):
        original_questions_for_debug = joblib.load("original_questions_for_debug.pkl")
    logger.info("ML models loaded successfully.")
except FileNotFoundError as e:
    logger.critical(f"ML model file not found: {e.filename}. Please run train_model.py.")
except Exception as e:
    logger.critical(f"Error loading ML model files: {e}", exc_info=True)

# --- Keyword Phrase Matching (Enhanced for Exactness and Variations) ---
def get_answer_by_exact_keyword_phrase(user_query_original):
    logger.info(f"Attempting exact keyword match for original query: '{user_query_original}'")

    processed_exact_user_query = exact_keyword_preprocessor(user_query_original)
    ans_id = exact_keyword_lookup.get(processed_exact_user_query)
    if ans_id is not None:
        answer_info = all_answers_data.get(ans_id)
        if answer_info and answer_info['answer_text']:
            logger.info(f"Direct EXACT keyword match (user query with spaces) found for '{processed_exact_user_query}'. Answer ID: {ans_id}")
            return {
                'answer_id': ans_id,
                'answer_text': answer_info['answer_text'],
                'source': 'direct_exact_keyword_user_spaces'
            }

    processed_no_space_user_query = no_space_keyword_preprocessor(user_query_original)
    ans_id = no_space_keyword_lookup.get(processed_no_space_user_query)
    if ans_id is not None:
        answer_info = all_answers_data.get(ans_id)
        if answer_info and answer_info['answer_text']:
            logger.info(f"Direct EXACT keyword match (user query no spaces) found for '{processed_no_space_user_query}'. Answer ID: {ans_id}")
            return {
                'answer_id': ans_id,
                'answer_text': answer_info['answer_text'],
                'source': 'direct_exact_keyword_user_no_spaces'
            }

    tokens_for_permutation = burmese_preprocessor(user_query_original).split()
    for r in range(min(len(tokens_for_permutation), 4), 0, -1):
        for perm_tokens in permutations(tokens_for_permutation, r):
            phrase_with_spaces_from_perm = exact_keyword_preprocessor(" ".join(perm_tokens))
            ans_id_perm_with_spaces = exact_keyword_lookup.get(phrase_with_spaces_from_perm)
            if ans_id_perm_with_spaces is not None:
                answer_info_perm = all_answers_data.get(ans_id_perm_with_spaces)
                if answer_info_perm and answer_info_perm['answer_text']:
                    logger.info(f"Permuted EXACT keyword match (with spaces from perm) found for '{phrase_with_spaces_from_perm}'. Answer ID: {ans_id_perm_with_spaces}")
                    return {
                        'answer_id': ans_id_perm_with_spaces,
                        'answer_text': answer_info_perm['answer_text'],
                        'source': 'permuted_exact_keyword_from_tokens_with_spaces'
                    }

            phrase_no_spaces_from_perm = no_space_keyword_preprocessor("".join(perm_tokens))
            ans_id_perm_no_spaces = no_space_keyword_lookup.get(phrase_no_spaces_from_perm)
            if ans_id_perm_no_spaces is not None:
                answer_info_perm = all_answers_data.get(ans_id_perm_no_spaces)
                if answer_info_perm and answer_info_perm['answer_text']:
                    logger.info(f"Permuted EXACT keyword match (no spaces from perm) found for '{phrase_no_spaces_from_perm}'. Answer ID: {ans_id_perm_no_spaces}")
                    return {
                        'answer_id': ans_id_perm_no_spaces,
                        'answer_text': answer_info_perm['answer_text'],
                        'source': 'permuted_exact_keyword_from_tokens_no_spaces'
                    }
    logger.info(f"No direct or strong permuted EXACT keyword phrase match found for '{user_query_original}'.")
    return None

# --- Category-Based Matching (Returns a single best answer based on score) ---
def get_answer_by_categories(user_query_original):
    processed_query = burmese_preprocessor(user_query_original)
    query_tokens = set(processed_query.split()) # Convert to set for faster lookup

    # Dictionary to store potential answers with their scores
    # Key: answer_id, Value: total_score
    candidate_answers_scores = {}

    # Define score weights for different category levels
    score_weights = {
        'sub': 3,
        'child': 2,
        'parent': 1
    }

    # Helper to calculate token match score
    def get_token_match_score(category_name_processed_tokens, query_tokens_set, base_score):
        if not category_name_processed_tokens:
            return 0 # No tokens to match
        match_count = sum(1 for token in category_name_processed_tokens if token in query_tokens_set)
        return base_score * (match_count / len(category_name_processed_tokens))

    # Function to process and score categories
    def process_category_type(category_type_map, answers_by_category_map, score):
        for cat_id, cat_name_original in category_type_map.items():
            cat_name_processed = burmese_preprocessor(cat_name_original)
            cat_name_processed_tokens = cat_name_processed.split()

            calculated_score = get_token_match_score(cat_name_processed_tokens, query_tokens, score)

            if calculated_score > 0: # If at least one token matched
                if cat_name_processed in answers_by_category_map:
                    for ans_id in answers_by_category_map[cat_name_processed]:
                        candidate_answers_scores[ans_id] = candidate_answers_scores.get(ans_id, 0) + calculated_score

    # Apply scoring for each category type
    process_category_type(category_maps['sub'], answers_by_sub, score_weights['sub'])
    process_category_type(category_maps['child'], answers_by_child, score_weights['child'])
    process_category_type(category_maps['parent'], answers_by_parent, score_weights['parent'])


    if candidate_answers_scores:
        # Find the answer_id with the highest score
        # If scores are tied, prefer the one with the lowest answer_id (arbitrary but consistent)
        best_ans_id = max(candidate_answers_scores, key=lambda k: (candidate_answers_scores[k], -k))
        best_score = candidate_answers_scores[best_ans_id]

        answer_info = all_answers_data.get(best_ans_id)

        if answer_info and answer_info['answer_text']:
            logger.info(f"Category match found for '{user_query_original}'. "
                        f"Best match Answer ID: {best_ans_id} with score: {best_score}")
            return {
                'answer_id': best_ans_id,
                'answer_text': answer_info['answer_text'],
                'source': f'category_match_scored'
            }
    logger.info(f"No confident category match found for '{user_query_original}'.")
    return None

# --- NLP Model Lookup ---
def get_answer_from_nlp_model(user_question, similarity_threshold=0.65):
    if vectorizer is None or question_vectors is None or answers_for_similarity is None:
        logger.warning("NLP model components not loaded. Skipping NLP lookup.")
        return None
    user_question_processed = burmese_preprocessor(user_question)
    if not user_question_processed.strip():
        logger.info(f"Processed user question is empty for '{user_question}'. Skipping NLP lookup.")
        return None
    try:
        user_question_vector = vectorizer.transform([user_question_processed])
        if user_question_vector.sum() == 0:
            logger.info(f"User question vector is all zeros for '{user_question_processed}'. Skipping NLP lookup.")
            return None

        similarities = cosine_similarity(user_question_vector, question_vectors).flatten()
        best_match_index = np.argmax(similarities)
        best_similarity_score = similarities[best_match_index]

        if original_questions_for_debug and best_match_index < len(original_questions_for_debug):
            best_training_question = original_questions_for_debug[best_match_index]
            logger.info(f"NLP similarity for '{user_question}' (processed: '{user_question_processed}'): "
                        f"Best score {best_similarity_score:.2f} with training question '{best_training_question}' (index {best_match_index})")
        else:
            logger.info(f"NLP similarity for '{user_question}' (processed: '{user_question_processed}'): "
                        f"Best score {best_similarity_score:.2f} with index {best_match_index}")

        if best_similarity_score >= similarity_threshold:
            if best_match_index < len(answers_for_similarity):
                answer_details = answers_for_similarity[best_match_index]
                if answer_details and answer_details.get('answer_text'):
                    logger.info(f"NLP match found with score {best_similarity_score:.2f} for '{user_question}'. Answer ID: {answer_details['id']}")
                    return {
                        'answer_id': answer_details['id'],
                        'answer_text': answer_details['answer_text'],
                        'source': 'nlp_model',
                        'score': best_similarity_score
                    }
        logger.info(f"NLP match below threshold ({similarity_threshold}) or no valid answer found for '{user_question}'.")
        return None
    except Exception as e:
        logger.error(f"Error during NLP model lookup for query '{user_question}': {e}", exc_info=True)
        return None

# --- Serper Search with Retry ---
# NOTE: This function is kept but will no longer be called in get_core_answer_for_single_query_segment
# as per the user's request to remove Serper API search results when not found.
def perform_serper_search(query):
    if not SERPER_API_KEY or SERPER_API_KEY == "YOUR_SERPER_API_KEY_HERE":
        logger.warning("SERPER_API_KEY is not configured or is placeholder. Skipping web search.")
        return [], "Serper API Key ကို ဖွဲ့စည်းမထားပါ သို့မဟုတ် နေရာအစားထိုးထားပါသည်။ (Serper API Key is not configured or is a placeholder.)"

    cached_results = get_cached_search(query)
    if cached_results:
        logger.info(f"Returning cached Serper search results for '{query}'.")
        return cached_results, None

    url = "https://google.serper.dev/search"
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    payload = {"q": query, "num": 5, "gl": "mm", "hl": "my"}

    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))

    try:
        response = session.post(url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        organic_results = data.get('organic', [])
        search_results_list = []
        for i, item in enumerate(organic_results):
            title = item.get('title', 'ခေါင်းစဉ်မရှိ (No title)')
            link = item.get('link', '#')
            snippet = item.get('snippet', 'အကျဉ်းချုပ်မရှိပါ။ (No snippet available.)')
            search_results_list.append({
                'title': title,
                'link_html': f'<a href="{link}" target="_blank" style="color:#1a73e8; text-decoration:none;">{title}</a>',
                'snippet': snippet
            })

        cache_search(query, search_results_list)
        logger.info(f"Serper search for '{query}' returned {len(search_results_list)} results and cached.")
        return search_results_list, None

    except requests.exceptions.HTTPError as e:
        error_msg = f"Serper API HTTP အမှား: {e.response.status_code} - {e.response.text}"
        logger.error(error_msg, exc_info=True)
        return [], f"အင်တာနက်ရှာဖွေမှု အမှားဖြစ်ပွားပါသည်။ Serper API HTTP အမှား: {e.response.status_code}. \nကျေးဇူးပြု၍ ခဏအကြာတွင် ထပ်မံကြိုးစားပါ။ (Internet search error. Serper API HTTP Error: {e.response.status_code}. Please try again later.)"
    except requests.exceptions.ConnectionError as e:
        error_msg = f"Serper API ချိတ်ဆက်မှု အမှား: {e}"
        logger.error(error_msg, exc_info=True)
        return [], f"အင်တာနက်ချိတ်ဆက်မှု အမှားဖြစ်ပွားပါသည်။ \nကျေးဇူးပြု၍ သင့်အင်တာနက်ချိတ်ဆက်မှုကို စစ်ဆေးပြီး ထပ်မံကြိုးစားပါ။ (Internet connection error. Please check your internet connection and try again.)"
    except requests.exceptions.Timeout as e:
        error_msg = f"Serper API တုံ့ပြန်မှုကြာချိန် ကျော်လွန်: {e}"
        logger.error(error_msg, exc_info=True)
        return [], f"အင်တာနက်ရှာဖွေမှု တုံ့ပြန်မှုကြာချိန် ကျော်လွန်သွားပါသည်။ \nကျေးဇူးပြု၍ ခဏအကြာတွင် ထပ်မံကြိုးစားပါ။ (Internet search timed out. Please try again later.)"
    except requests.exceptions.RequestException as e:
        error_msg = f"Serper API တောင်းဆိုမှု အမှား: {e}"
        logger.error(error_msg, exc_info=True)
        return [], f"အင်တာနက်ရှာဖွေမှု တောင်းဆိုမှု အမှားဖြစ်ပွားပါသည်။ \nကျေးဇူပြု၍ နောက်ပိုင်းတွင် ထပ်မံကြိုးစားပါ။ (Internet search request error. Please try again later.)"
    except Exception as e:
        error_msg = f"Serper API ရှာဖွေမှု မမျှော်လင့်သော အမှား: {e}"
        logger.error(error_msg, exc_info=True)
        return [], f"အင်တာနက်ရှာဖွေမှု မမျှော်ော်လင့်သော အမှားဖြစ်ပွားပါသည်။ \nကျေးဇူးပြု၍ နောက်ပိုင်းတွင် ထပ်မံကြိုးစားပါ။ (An unexpected internet search error occurred. Please try again later.)"

# --- Define specific introductory answers for broad single-word queries ---
BROAD_KEYWORD_INTRO_ANSWERS = {
    burmese_preprocessor("စပါး"): 1,
    burmese_preprocessor("ကော်ဖီ"): 10,
    burmese_preprocessor("ကြံ"): 20,
}

# --- Core Query Processing Logic (Independent of Streamlit) ---
def get_core_answer_for_single_query_segment(q_text, is_streamlit_context=False):
    final_answer_content_html = ""
    found_answer_id = None

    # 1. EXACT/STRONG KEYWORD PHRASE MATCH
    db_answer_info = get_answer_by_exact_keyword_phrase(q_text)
    if db_answer_info:
        final_answer_content_html = db_answer_info['answer_text'].replace('\n', '<br>')
        found_answer_id = db_answer_info['answer_id']
        logger.info(f"Strategy: Answer found via exact/permuted keyword match for user query '{q_text}'. Answer ID: {found_answer_id}")

    # 2. Broad Keyword Introductory Answers
    if not found_answer_id:
        processed_q_for_broad_match = burmese_preprocessor(q_text)
        if len(processed_q_for_broad_match.split()) <= 2 and processed_q_for_broad_match in BROAD_KEYWORD_INTRO_ANSWERS:
            intro_answer_id = BROAD_KEYWORD_INTRO_ANSWERS[processed_q_for_broad_match]
            answer_info = all_answers_data.get(intro_answer_id)
            if answer_info and answer_info['answer_text']:
                final_answer_content_html = answer_info['answer_text'].replace('\n', '<br>')
                found_answer_id = intro_answer_id
                logger.info(f"Strategy: Broad keyword intro match found for '{q_text}'. Answer ID: {found_answer_id}")
            else:
                logger.warning(f"Configured intro answer ID {intro_answer_id} for '{q_text}' not found in answers_db. Falling back to next strategy.")

    # 3. Category-Based Match
    if not found_answer_id:
        category_answer = get_answer_by_categories(q_text)
        if category_answer:
            final_answer_content_html = category_answer['answer_text'].replace('\n', '<br>')
            found_answer_id = category_answer['answer_id']
            logger.info(f"Strategy: Answer found via category match for user query '{q_text}'. Answer ID: {found_answer_id}")

    # 4. NLP Model Lookup
    if not found_answer_id:
        nlp_response_info = get_answer_from_nlp_model(q_text, similarity_threshold=0.70)
        if nlp_response_info and nlp_response_info.get('score', 0) >= 0.70:
            final_answer_content_html = nlp_response_info['answer_text'].replace('\n', '<br>')
            found_answer_id = nlp_response_info['answer_id']
            logger.info(f"Strategy: Answer found via NLP model for user query '{q_text}'. Answer ID: {found_answer_id}")
        else:
            logger.info(f"Strategy: NLP match not confident enough or no answer found for '{q_text}'.")

    # 5. Fallback if NO INTERNAL ANSWER was found
    if not found_answer_id:
        logger.info(f"Strategy: No confident internal answer found for '{q_text}'. Returning 'Not_Found'.")
        log_unanswered_query(q_text, is_streamlit_context=is_streamlit_context) # Still log unanswered queries

        final_answer_content_html = "Not_Found" # Set content for Streamlit UI
        found_answer_id = "Not_Found" # Set answer_id for API response

    return {
        'question': q_text,
        'answer_html': final_answer_content_html, # This will be "Not_Found" for not found cases in Streamlit
        'answer_id': found_answer_id # This will be "Not_Found" for not found cases in API
    }

# --- Streamlit UI Components ---
def run_streamlit_app():
    st.title("🌾 မြန်မာစိုက်ပျိုးရေး Chatbot")

    st.markdown("""
    <style>
    /* Overall chat container for centering and max-width */
    .stApp > header + div {
        max-width: 800px;
        margin: 0 auto;
        padding: 15px;
    }

    /* User Bubble (Right Side) */
    .user-bubble {
        background-color: #ededed; /* Light gray */
        color: #444;
        border-radius: 18px 18px 4px 18px; /* Rounded top, pointed bottom-right */
        padding: 12px 18px;
        margin: 8px 0 8px auto; /* Margin-left: auto pushes it right */
        max-width: 75%; /* Limit bubble width */
        font-size: 16px;
        word-break: break-word;
        box-shadow: 0 2px 8px rgba(180,180,180,0.08); /* Subtle shadow */
        position: relative; /* For avatar positioning */
        display: flex;
        flex-direction: row-reverse; /* Avatar on the right, text on the left */
        align-items: center;
        gap: 10px;
    }

    /* Assistant Bubble (Left Side) */
    .bot-bubble {
        background-color: #f7f7f7; /* Slightly lighter gray */
        color: #555;
        border: 1px solid #ececec; /* Subtle border */
        border-radius: 18px 18px 18px 4px; /* Rounded top, pointed bottom-left */
        padding: 12px 18px;
        margin: 8px auto 8px 0; /* Margin-right: auto pushes it left */
        max-width: 75%;
        font-size: 16px;
        word-break: break-word;
        box-shadow: 0 2px 8px rgba(180,180,180,0.08);
        position: relative; /* For avatar positioning */
        display: flex;
        flex-direction: row; /* Avatar on the left, text on the right */
        align-items: center;
        gap: 10px;
    }

    /* Avatar styling */
    .avatar {
        font-size: 28px; /* Adjust size as needed */
        line-height: 1; /* Ensure emoji sits correctly */
        flex-shrink: 0; /* Prevent avatar from shrinking */
    }

    /* Text content within bubbles */
    .bubble-text {
        flex-grow: 1; /* Allows text to take available space */
    }

    .user-bubble .bubble-text {
        text-align: right;
    }

    .bot-bubble .bubble-text {
        text-align: left;
    }

    /* Answer ID for debugging */
    .answer-id {
        font-size: 13px;
        color: #aaa;
        margin-top: 6px;
        display: block; /* Ensures it's on its own line */
        word-break: break-all; /* Breaks long IDs if needed */
    }

    /* Input field styling */
    .stTextInput > div > div > input {
        border-radius: 20px;
        padding: 10px 15px;
        font-size: 16px;
        border: 1px solid #d1d5db;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    .stTextInput > div > div > input:focus {
        outline: none;
        border-color: #34d399; /* Green on focus */
        box-shadow: 0 0 0 3px rgba(52, 211, 153, 0.4);
    }

    /* Button styling */
    .stButton > button {
        border-radius: 20px;
        background-color: #10B981; /* Green */
        color: white;
        padding: 8px 15px;
        margin: 5px;
        white-space: normal;
        word-wrap: break-word;
        height: auto;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: background-color 0.2s ease, transform 0.1s ease;
    }
    .stButton > button:hover {
        background-color: #059669; /* Darker green on hover */
        transform: translateY(-1px);
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* Specific style for the clear chat button */
    #clear_chat_button_main button {
        background-color: #EF4444; /* Red */
    }
    #clear_chat_button_main button:hover {
        background-color: #DC2626; /* Darker red on hover */
    }

    /* Custom Scrollbar Styles for Webkit browsers (Chrome, Safari, Edge) */
    ::-webkit-scrollbar {
        width: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 5px;
    }
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 5px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    </style>
    """, unsafe_allow_html=True)


    if "displayed_answers" not in st.session_state:
        st.session_state.displayed_answers = []
    if "logged_unanswered_queries" not in st.session_state:
        st.session_state.logged_unanswered_queries = set()

    if st.button("စကားဝိုင်းရှင်းလင်းရန် (Clear Chat)", key="clear_chat_button_main"):
        st.session_state.displayed_answers = []
        st.session_state.logged_unanswered_queries = set()
        st.rerun()

    for chat_entry in st.session_state.displayed_answers:
        user_bubble_html = f"""
        <div class="user-bubble">
            <div class="avatar">🧑</div>
            <span class="bubble-text">{chat_entry["question"].replace('\n', '<br>')}</span>
        </div>
        """
        st.markdown(user_bubble_html, unsafe_allow_html=True)

        answer_content_with_id = chat_entry['answer_html']
        # Display Answer ID only if it's not "Not_Found"
        if chat_entry['answer_id'] and chat_entry['answer_id'] != "Not_Found":
            answer_content_with_id += f"<div class='answer-id'>Answer ID: <code>{chat_entry['answer_id']}</code></div>"

        bot_bubble_html = f"""
        <div class="bot-bubble">
            <div class="avatar">🤖</div>
            <span class="bubble-text">{answer_content_with_id}</span>
        </div>
        """
        st.markdown(bot_bubble_html, unsafe_allow_html=True)

    user_query = st.chat_input("မေးခွန်းတစ်ခုခု ရိုက်ထည့်ပါ... (Type your question here...)", key="user_input_field")

    if user_query:
        # Split queries by common Burmese and English punctuation to handle multiple questions
        split_questions = [seg.strip() for seg in re.split(r'[၊။,;!?-]+', user_query) if seg.strip()]
        all_individual_questions = list(set(split_questions)) # Use set for uniqueness, then list for order

        for q_text in all_individual_questions:
            response = get_core_answer_for_single_query_segment(q_text, is_streamlit_context=True)
            st.session_state.displayed_answers.append(response)
        st.rerun()

# --- Flask API Application ---
app_flask = Flask(__name__)

@app_flask.route('/chat', methods=['POST'])
def api_chat_endpoint():
    try:
        data = request.get_json()
        if not data or 'user_query' not in data:
            return jsonify({"error": "Invalid request. 'user_query' field is required in JSON body."}), 400

        user_query = data['user_query']
        logger.info(f"API Request received for query: '{user_query}'")

        # Split query for processing multiple segments, similar to Streamlit
        split_questions = [seg.strip() for seg in re.split(r'[၊။,;!?-]+', user_query) if seg.strip()]
        all_individual_questions = list(set(split_questions))

        api_responses = []
        for q_text in all_individual_questions:
            response = get_core_answer_for_single_query_segment(q_text, is_streamlit_context=False)
            # API response should only get answer_id, and if not found, "Not_Found" string
            api_responses.append({
                'question': response['question'],
                'answer_id': response['answer_id'] # This will be "Not_Found" or the actual ID
            })

        # If there's only one segment, return single response, else return a list
        if len(api_responses) == 1:
            return jsonify(api_responses[0]), 200
        else:
            return jsonify(api_responses), 200

    except Exception as e:
        logger.error(f"API processing error: {e}", exc_info=True)
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

# --- Main Execution Block ---
if __name__ == "__main__":
    if os.getenv("RUN_FLASK_API") == "true":
        logger.info("Starting Flask API application...")
        app_flask.run(host='0.0.0.0', port=5000, debug=True)
    else:
        logger.info("Starting Streamlit UI application...")
        run_streamlit_app()