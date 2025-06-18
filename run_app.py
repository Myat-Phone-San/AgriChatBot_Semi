import streamlit as st
import mysql.connector
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import os
import time
import requests

# Assumes these files exist and contain necessary configurations/data
# Make sure db_config.py and api_config.py are in the same directory
# db_config.py should contain db_settings dictionary
# api_config.py should contain SERPER_API_KEY = "YOUR_API_KEY_HERE"
from db_config import db_settings
from my_stopwords import burmese_stopwords # Make sure this file exists with your stopwords
from api_config import SERPER_API_KEY # Modified: Import SERPER_API_KEY

# --- NLTK Data Path Setup ---
script_dir = os.path.dirname(__file__)
custom_nltk_data_path = os.path.join(script_dir, 'nltk_data')
if custom_nltk_data_path not in nltk.data.path:
    nltk.data.path.append(custom_nltk_data_path)

# Ensure NLTK punkt tokenizer is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt', download_dir=custom_nltk_data_path)
    print("NLTK 'punkt' tokenizer downloaded.")
# Ensure punkt_tab (often used internally by punkt) is also available if needed
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("NLTK 'punkt_tab' tokenizer not found. Downloading...")
    nltk.download('punkt_tab', download_dir=custom_nltk_data_path)
    print("NLTK 'punkt_tab' tokenizer downloaded.")


# --- Streamlit App Title ---
st.title("🌾 မြန်မာစိုက်ပျိုးရေး Chatbot 🌱")

# --- Database Helper Functions ---
def create_db_connection_attempt():
    """Attempts to establish a single database connection."""
    try:
        conn = mysql.connector.connect(**db_settings)
        if conn.is_connected():
            return conn
        else:
            conn.close()
            return None
    except mysql.connector.Error as err:
        print(f"Database connection attempt failed: {err}")
        return None
    except Exception as e:
        print(f"Unexpected error during DB connection attempt: {e}")
        return None

def log_unanswered_query(query_text):
    """Logs a user query to the unanswered_queries table."""
    if "logged_unanswered_queries" not in st.session_state:
        st.session_state.logged_unanswered_queries = set()
    if query_text in st.session_state.logged_unanswered_queries:
        print(f"Query '{query_text}' already logged in this session. Skipping.")
        return

    print(f"Attempting to log query: {query_text}")
    conn = None
    cursor = None
    max_retries = 3
    retry_delay = 1 # seconds

    for attempt in range(max_retries):
        conn = create_db_connection_attempt()
        if conn:
            try:
                cursor = conn.cursor()
                sql = "INSERT INTO unanswered_queries (query_text) VALUES (%s)"
                cursor.execute(sql, (query_text,))
                conn.commit()
                st.session_state.logged_unanswered_queries.add(query_text) # Mark as logged
                print(f"Successfully logged query: {query_text}")
                return
            except mysql.connector.Error as err:
                error_msg = f"Error logging query '{query_text}' on attempt {attempt + 1}: {err}"
                print(error_msg)
                st.warning(f"မဖြေနိုင်သော မေးခွန်း မှတ်တမ်းတင်ရာတွင် အမှားအယွင်းရှိခဲ့ပါသည်: {err}")
            except Exception as e:
                error_msg = f"Unexpected error logging query '{query_text}' on attempt {attempt + 1}: {e}"
                print(error_msg)
                st.error(f"မမျှော်လင့်ထားသော အမှား: {e}")
            finally:
                if cursor:
                    cursor.close()
                if conn and conn.is_connected():
                    conn.close()
        else:
            print(f"Failed to get DB connection for logging on attempt {attempt + 1}.")

        if attempt < max_retries - 1:
            time.sleep(retry_delay)

    error_msg = "Failed to log query after multiple retries. Check DB connection details and MySQL server status."
    print(error_msg)
    st.error(error_msg)


@st.cache_data(ttl=3600)
def load_data_from_db():
    """Loads keywords, questions, and answers from the database."""
    conn = None
    cursor = None
    try:
        conn = create_db_connection_attempt()
        if not conn:
            st.error("ဒေတာဘေ့စ်မှ ဒေတာများ မဖတ်ရှုနိုင်ပါ။ ချိတ်ဆက်မှုပြဿနာရှိပါသည်။")
            print("Failed to load data due to database connection issue.")
            return {}, {}, {}, {}

        cursor = conn.cursor(dictionary=True)

        keywords_data = {}
        cursor.execute("SELECT id, keyword FROM keywords")
        for row in cursor.fetchall():
            keywords_data[row['keyword']] = row['id']

        questions_by_keyword_id = {}
        questions_by_text = {}
        cursor.execute("SELECT id, keyword_id, question_text FROM questions")
        for row in cursor.fetchall():
            if row['keyword_id'] not in questions_by_keyword_id:
                questions_by_keyword_id[row['keyword_id']] = []
            questions_by_keyword_id[row['keyword_id']].append({'id': row['id'], 'question_text': row['question_text']})
            questions_by_text[row['question_text']] = row['id']

        answers_data = {}
        cursor.execute("SELECT question_id, answer_text FROM answers")
        for row in cursor.fetchall():
            answers_data[row['question_id']] = row['answer_text']

        return keywords_data, questions_by_keyword_id, questions_by_text, answers_data
    except mysql.connector.Error as err:
        error_msg = f"Error loading data from database: {err}"
        print(error_msg)
        st.error(f"ဒေတာဘေ့စ်မှ ဒေတာဖတ်ရာတွင် အမှား: {err}")
        return {}, {}, {}, {}
    except Exception as e:
        error_msg = f"Unexpected error loading data: {e}"
        print(error_msg)
        st.error(f"ဒေတာများဖတ်ရှုရာတွင် မမျှော်လင့်ထားသော အမှား: {e}")
        return {}, {}, {}, {}
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

keywords_db, questions_by_keyword_id, questions_by_text, answers_db = load_data_from_db()

# --- Clarification Map for Keyword-based Questions ---
clarification_map = {}
for keyword_name, keyword_id in keywords_db.items():
    options = {}
    if keyword_id in questions_by_keyword_id:
        for q_data in questions_by_keyword_id[keyword_id]:
            options[q_data['question_text']] = answers_db.get(q_data['id'], "ဤမေးခွန်းအတွက် အဖြေမတွေ့ပါ။")
    clarification_map[keyword_name] = {
        "title": f"**{keyword_name} နှင့်ပတ်သက်လို့ ဘာများသိချင်ပါသလဲ?**",
        "options": options
    }

# --- Load ML Model and Vectorizer ---
def burmese_preprocessor(text):
    """Custom preprocessor for Burmese text using NLTK word_tokenize and stopword removal."""
    text = str(text) # Ensure text is a string
    tokens = word_tokenize(text)
    processed_tokens = [word.strip() for word in tokens if word.strip() and word.strip() not in burmese_stopwords]
    return " ".join(processed_tokens)

try:
    vectorizer = joblib.load("vectorizer.pkl")
    question_vectors = joblib.load("question_vectors.pkl")
    answers_for_similarity = joblib.load("answers_for_similarity.pkl")
except FileNotFoundError:
    st.error("Error: ML model files (vectorizer.pkl, question_vectors.pkl, answers_for_similarity.pkl) not found.")
    st.error("ကျေးဇူးပြု၍ train.py ကိုအရင်ဆုံး run ပြီး ML model များကို ဖန်တီးပါ။")
    st.stop()
except Exception as e:
    st.error(f"Error loading ML model components: {e}")
    st.stop()

# --- Keyword Aliases ---
# Only include aliases for keywords that should trigger a clarification menu.
# Other queries will go through NLP model and then Serper.
keyword_aliases = {
    "စ": "စပါး", "စပ": "စပါး", "စပါ": "စပါး", "စပါး": "စပါး",
    "သ": "သစ်ခွ", "သစ်": "သစ်ခွ", "သစ်ခ": "သစ်ခွ", "သစ်ခွ": "သစ်ခွ",
    # Removed "ခရမ်း": "ခရမ်းပင်", "ခရမ်းပင်": "ခရမ်းပင်",
    # Removed "အာလူး": "အာလူးသီး", "အာလူးသီး": "အာလူးသီး"
}

# --- Answer Retrieval Functions ---
def get_answer_by_question_text(question_text):
    """Fetches exact answer for a given question text from DB."""
    q_id = questions_by_text.get(question_text.strip())
    if q_id:
        return answers_db.get(q_id)
    return None

def match_keyword_input(text):
    """Matches user input to a predefined keyword or its alias."""
    text = text.strip()
    aliased_keyword = keyword_aliases.get(text)
    # Check if the aliased keyword exists in the clarification_map (i.e., has associated questions)
    if aliased_keyword and aliased_keyword in clarification_map:
        return aliased_keyword
    # Also check if the raw text is a keyword that has associated questions
    if text in keywords_db and text in clarification_map:
        return text
    return None

def get_answer_from_nlp_model(user_question, similarity_threshold=0.7):
    """Finds the most similar answer using TF-IDF and cosine similarity."""
    user_question_processed = burmese_preprocessor(user_question)
    if not user_question_processed.strip():
        print(f"No valid tokens after preprocessing: {user_question}")
        return None
    
    # Ensure the vectorizer vocabulary is not empty before transforming
    if not hasattr(vectorizer, 'vocabulary_') or not vectorizer.vocabulary_:
        print("Vectorizer vocabulary is empty. Cannot transform user question.")
        return None

    # Check if user_question_processed results in an empty vector based on vocabulary
    # This can happen if the processed query contains no words from the vocabulary
    try:
        user_question_vector = vectorizer.transform([user_question_processed])
    except ValueError as e:
        print(f"Error transforming user question '{user_question_processed}': {e}")
        return None # Cannot process if transformation fails

    # Check if the user_question_vector is empty (e.g., if no common words with vocabulary)
    if user_question_vector.shape[1] == 0:
        print("User question vector is empty after transformation.")
        return None

    similarities = cosine_similarity(user_question_vector, question_vectors).flatten()
    best_match_index = np.argmax(similarities)
    best_similarity_score = similarities[best_match_index]
    print(f"Similarity score for '{user_question}': {best_similarity_score}")

    if best_similarity_score >= similarity_threshold:
        return answers_for_similarity[best_match_index]
    return None

# --- Serper Search Function ---
def perform_serper_search(query):
    """Performs a search using Serper API and returns formatted results."""
    # CORRECTED: New URL and HTTP method is POST
    url = "https://google.serper.dev/search"
    headers = {
        'X-API-KEY': SERPER_API_KEY, # API Key is passed in header
        'Content-Type': 'application/json' # IMPORTANT: Specify content type for POST request
    }
    # CORRECTED: Parameters are passed as a JSON body, not URL params
    payload = {
        "q": query,
        "num": 3 # Request top 3 results
    }

    try:
        # CORRECTED: Changed to requests.post and passing 'json' data
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status() # Raise an exception for bad status codes (e.g., 4xx or 5xx)
        data = response.json()

        organic_results = data.get('organic', [])

        if not organic_results:
            return "အင်တာနက်မှ ရှာဖွေမှုရလဒ်များကို ရှာမတွေ့ပါ။"

        search_results_markdown = "အင်တာနက်ရှာဖွေမှုမှ ရလဒ်များ (Serper API မှ): \n\n"
        for i, item in enumerate(organic_results):
            title = item.get('title', 'No Title')
            link = item.get('link', '#')
            snippet = item.get('snippet', 'No snippet available.')
            search_results_markdown += (
                f"**{i+1}. [{title}]({link})**\n" # Formatted to show clickable link
                f"{snippet}\n\n"
            )
        return search_results_markdown
    except requests.exceptions.RequestException as e:
        print(f"Error during Serper API search (RequestException): {e}")
        return f"အင်တာနက်မှ ရှာဖွေမှုရလဒ်များကို ရယူရာတွင် ပြဿနာဖြစ်ပွားခဲ့ပါသည်။ (Network Error: {e})"
    except Exception as e:
        print(f"Error during Serper API search (General Error): {e}")
        return f"အင်တာနက်မှ ရှာဖွေမှုရလဒ်များကို ရယူရာတွင် မမျှော်လင့်ထားသော ပြဿနာဖြစ်ပွားခဲ့ပါသည်။ ({e})"


# --- Process Multiple Questions ---
def process_multiple_questions(full_user_input):
    """Splits user input into individual questions and retrieves answers."""
    print(f"Processing input: {full_user_input}")
    split_questions = []
    # Split by Burmese full stop and comma, then filter empty strings
    temp_segments = full_user_input.split('။')
    for segment in temp_segments:
        sub_segments = segment.split('၊')
        for sub_seg in sub_segments:
            # Further split by English comma if present, just in case
            if ',' in sub_seg:
                split_questions.extend([q.strip() for q in sub_seg.split(',') if q.strip()])
            elif sub_seg.strip(): # Add if not empty after stripping
                split_questions.append(sub_seg.strip())

    # If no delimiters found but there's input, treat as a single question
    if not split_questions and full_user_input.strip():
        split_questions = [full_user_input.strip()]

    # Remove duplicates while preserving order for processing
    all_individual_questions = list(dict.fromkeys(split_questions))
    responses = []

    for q_text in all_individual_questions:
        print(f"Processing question: {q_text}")
        final_answer = ""

        # 1. Try to find exact match in DB
        db_answer = get_answer_by_question_text(q_text)
        if db_answer:
            print(f"Found DB answer for '{q_text}': {db_answer}")
            final_answer = db_answer
        else:
            # 2. Try to find similar answer using NLP model
            nlp_response = get_answer_from_nlp_model(q_text)
            if nlp_response:
                print(f"Found NLP answer for '{q_text}': {nlp_response}")
                final_answer = nlp_response
            else:
                # 3. If no answer found, perform Serper Search
                print(f"No answer found for '{q_text}' in DB or NLP, attempting Serper Search...")
                search_results = perform_serper_search(q_text)
                
                final_answer = f"သင့်မေးခွန်းအတွက် အဖြေကို ကျွန်ုပ်၏ ဒေတာဘေ့စ်တွင် ရှာမတွေ့ပါ။ အောက်ပါ အင်တာနက် ရှာဖွေမှု ရလဒ်များက အထောက်အကူ ဖြစ်နိုင်ပါသည်။\n\n{search_results}"

                # Still log to unanswered queries, even if search results are provided,
                # as it means the internal knowledge base didn't cover it.
                log_unanswered_query(q_text)

        responses.append((q_text, final_answer))

    return responses

# --- Streamlit Session State Initialization ---
if "selected_topic" not in st.session_state:
    st.session_state.selected_topic = None
if "displayed_answers" not in st.session_state:
    st.session_state.displayed_answers = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "logged_unanswered_queries" not in st.session_state:
    st.session_state.logged_unanswered_queries = set()

# --- UI Styling ---
st.markdown("""
<style>
body {
    font-family: 'Pyidaungsu', sans-serif; /* Assuming Pyidaungsu is available or a system font */
}
div.stForm {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: #f0f2f6;
    padding: 10px 20px;
    box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
    z-index: 1000;
    box-sizing: border-box;
    border-top: 1px solid #e0e0e0;
}
.stTextInput > div > div > input {
    border-radius: 20px;
    padding: 10px 15px;
    min-height: 40px;
    font-size: 16px;
    border: 1px solid #e0e0e0;
    box-shadow: none;
    width: 100%;
    box-sizing: border-box;
}
.stButton > button {
    border-radius: 20px;
    padding: 10px 15px;
    background-color: #4CAF50;
    color: white;
    border: none;
    cursor: pointer;
    font-size: 18px;
    font-weight: bold;
    width: 100%;
    transition: background-color 0.3s;
}
.stButton > button:hover {
    background-color: #45a049;
}
#clear_chat_button_main button {
    background-color: #f44336;
    color: white;
    border-radius: 5px;
    padding: 8px 15px;
    font-size: 14px;
}
#clear_chat_button_main button:hover {
    background-color: #da190b;
}
.chat-message {
    background-color: #f9f9f9;
    padding: 10px 15px;
    border-radius: 10px;
    margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.user-message {
    text-align: right;
    background-color: #e6f7ff;
    border-bottom-right-radius: 0;
}
.bot-message {
    text-align: left;
    background-color: #f0f0f0;
    border-bottom-left-radius: 0;
}
.block-container {
    padding-bottom: 100px; /* Adjust this to make space for fixed input form */
}
</style>
""", unsafe_allow_html=True)

# --- Display Chat History ---
# Iterate through displayed_answers to show chat history
for i, (q, a) in enumerate(st.session_state.displayed_answers):
    st.markdown(f"<div class='chat-message user-message'>**မေးခွန်း:** {q}</div>", unsafe_allow_html=True)
    # Use st.markdown with unsafe_allow_html for rendering links in search results
    st.markdown(f"<div class='chat-message bot-message'>📝🤖 **ဖြေချက်**: {a}</div>", unsafe_allow_html=True)
    
    # Feedback buttons should only appear for answers from your internal DB/NLP, not raw search results
    # Check if the answer contains specific strings indicating it's an external search result or a "not found" message
    if "စိတ်မရှိပါနဲ့" not in a and "အင်တာနက်ရှာဖွေမှုမှ ရလဒ်များ" not in a:
        col_feedback_yes, col_feedback_no, _ = st.columns([0.1, 0.1, 0.8])
        with col_feedback_yes:
            if st.button("👍", key=f"feedback_yes_{i}"):
                st.toast("ကျေးဇူးတင်ပါတယ်။ သင့်ရဲ့အကြံပြုချက်ကို မှတ်သားထားပါတယ်။")
        with col_feedback_no:
            if st.button("👎", key=f"feedback_no_{i}"):
                st.toast("စိတ်မကောင်းပါဘူး။ သင့်ရဲ့မေးခွန်းကို ပိုမိုကောင်းမွန်အောင်ဖြေဆိုနိုင်ဖို့ ကြိုးစားပါ့မယ်။")
    st.markdown("---") # Separator between turns

# --- Display Keyword Clarification Options (if a topic was selected) ---
if st.session_state.selected_topic:
    topic_data = clarification_map.get(st.session_state.selected_topic)
    if topic_data:
        st.markdown(f"🤖 {topic_data['title']}")
        # Create columns for buttons to make them appear more organized
        cols = st.columns(3) # Adjust number of columns as needed
        btn_idx = 0
        for option_question_text, option_answer_text in topic_data["options"].items():
            with cols[btn_idx % 3]: # Cycle through columns
                if st.button(option_question_text, key=f"btn_{option_question_text}"):
                    # When a clarification button is clicked, display that specific answer
                    st.session_state.displayed_answers.append((option_question_text, option_answer_text))
                    st.session_state.selected_topic = None # Reset selected topic
                    st.session_state.user_input = "" # Clear input
                    st.rerun()
            btn_idx += 1
    else:
        st.error(f"Topic '{st.session_state.selected_topic}' ကို မတွေ့ပါ။")
        st.session_state.selected_topic = None

# --- Clear Chat Button (positioned below chat history, above input form) ---
st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True) # Spacer
col_clear_button = st.columns([1, 2, 1]) # Center the button somewhat
with col_clear_button[2]: # Place in the third column
    if st.button("အစအဆုံးပြန်စရန်", key="clear_chat_button_main"):
        st.session_state.selected_topic = None
        st.session_state.displayed_answers = []
        st.session_state.user_input = ""
        st.session_state.logged_unanswered_queries = set() # Clear logged queries on full reset
        st.rerun()

# --- Input Form (fixed at the bottom) ---
with st.form("chat_input_form", clear_on_submit=True):
    col1, col2 = st.columns([0.9, 0.1])
    with col1:
        user_query = st.text_input(
            "မေးခွန်းထည့်ပါ...",
            key="user_query_input",
            value=st.session_state.user_input, # Keeps the input if rerunning without submitting
            placeholder="ဥပမာ: စပါးမိတ်ဆက်၊ စပါးတွင်ကျရောက်တတ်သောရောဂါများ",
            label_visibility="collapsed" # Hides the default label above the text input
        )
    with col2:
        send_button = st.form_submit_button("➤", use_container_width=True)

    if send_button and user_query:
        # Clear previous answers if a new query is submitted, unless it's a clarification
        if not st.session_state.selected_topic: # Only clear if not in a topic clarification flow
            st.session_state.displayed_answers = [] 
        
        st.session_state.selected_topic = None # Reset topic selection for new input
        st.session_state.user_input = user_query # Store current input for persistence if needed

        # Process the user's input
        processed_qa_pairs = process_multiple_questions(user_query)
        
        is_single_keyword_query = False
        if len(processed_qa_pairs) == 1:
            # Check if the exact user_query (before splitting) matches a direct keyword or alias
            # This is important for triggering the keyword-based clarification menu.
            # Example: If user types "စပါး" and "စပါး" is a keyword.
            matched_keyword = match_keyword_input(user_query.strip())
            if matched_keyword:
                # This condition ensures that complex sentences containing a keyword
                # (e.g., "စပါး စိုက်ပျိုးနည်း") do NOT trigger the clarification menu.
                # Only direct keyword inputs (like "စပါး") should.
                # It checks if the original user input exactly matches the alias or the keyword itself.
                if user_query.strip() == matched_keyword or \
                   any(alias_key == user_query.strip() and alias_value == matched_keyword 
                       for alias_key, alias_value in keyword_aliases.items()):
                    is_single_keyword_query = True
                    st.session_state.selected_topic = matched_keyword
        
        if is_single_keyword_query:
            # If it's a single keyword query, let the clarification section handle display.
            # The clarification menu will be shown at the top of the chat area.
            pass 
        else:
            # For multi-question or non-keyword-only queries, display the answers directly.
            # These answers can come from NLP or Serper search.
            st.session_state.displayed_answers.extend(processed_qa_pairs)
        
        st.rerun() # Rerun to update the display