# train_model.py
import mysql.connector
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import os
import re
import nltk
import time

# Import database settings and stopwords
from db_config import db_settings
from my_stopwords import burmese_stopwords

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, filename='train.log', format='%(asctime)s - %(levelname)s - %(message)s')
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

# --- Preprocessing Functions ---
# This one is for general NLP tokenization and stop word removal (used by TF-IDF)
def burmese_preprocessor(text):
    if not isinstance(text, str):
        text = str(text)
    # This regex captures continuous Burmese characters or alphanumeric sequences
    text = re.sub(r'[^\u1000-\u109FA-Za-z0-9\s]', '', text.lower(), flags=re.UNICODE)
    tokens = re.findall(r'[\u1000-\u109F]+|[A-Za-z0-9]+', text, re.UNICODE)
    processed_tokens = [word.strip() for word in tokens if word.strip() and word.strip() not in burmese_stopwords]
    return " ".join(processed_tokens)

# This one is for exact keyword matching (preserves spaces, only lowercases)
def exact_keyword_preprocessor(text):
    if not isinstance(text, str):
        text = str(text)
    return text.lower().strip()

# This one is for no-space keyword matching (removes all spaces, only lowercases)
def no_space_keyword_preprocessor(text):
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r'\s+', '', text.lower()).strip()


# --- Database Connection ---
def create_db_connection():
    try:
        conn = mysql.connector.connect(**db_settings)
        logger.info("Connected to database successfully.")
        return conn
    except mysql.connector.Error as err:
        logger.error(f"Database connection error: {err}")
        return None

# --- Load Data from Database ---
def load_data_from_db():
    start_time = time.time()
    conn = create_db_connection()
    if not conn:
        logger.error("Failed to connect to database. Returning empty data.")
        return [], []
    
    answers_data = [] # To store answer IDs and text for NLP similarity mapping
    questions_data = [] # To store keywords for TF-IDF vectorization
    cursor = None
    try:
        cursor = conn.cursor(dictionary=True, buffered=True)
        # Fetch data from 'answers' table
        cursor.execute("""
            SELECT id, keyword, name
            FROM answers
            WHERE keyword IS NOT NULL AND TRIM(keyword) != ''
            AND name IS NOT NULL AND TRIM(name) != ''
        """)
        for row in cursor.fetchall():
            question_text = row['keyword'].strip()
            answer_text = row['name'].strip()
            answers_data.append({
                'id': row['id'],
                'answer_text': answer_text
            })
            questions_data.append(question_text) # Collect original keywords for vectorizer training
        
        end_time = time.time()
        logger.info(f"Loaded {len(questions_data)} questions and answers from database in {end_time - start_time:.2f} seconds.")
        return questions_data, answers_data
    except mysql.connector.Error as err:
        logger.error(f"Error loading data from database: {err}", exc_info=True)
        return [], []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# --- Train Model ---
def train_model():
    logger.info("Starting data loading for training...")
    questions, answers = load_data_from_db()
    if not questions or not answers:
        logger.error("No data loaded from database. Exiting model training.")
        return
    
    # Initialize TF-IDF Vectorizer
    # Reduced max_features for potentially faster training if your dataset is not extremely diverse
    # You can experiment with this value (e.g., 2000, 3000, 4000, 5000)
    # A smaller value means fewer features, smaller matrix, faster computation.
    # Too small might reduce accuracy.
    vectorizer = TfidfVectorizer(
        preprocessor=burmese_preprocessor, # Use the custom preprocessor for TF-IDF
        token_pattern=r'(?u)\b[\w\u1000-\u109F]+\b', # Ensures Burmese characters are treated as word characters
        max_features=3000 # OPTIMIZATION: Reduced from 5000 for faster training
    )
    
    try:
        logger.info("Starting TF-IDF vectorization...")
        vectorization_start_time = time.time()
        # Fit the vectorizer on the loaded questions and transform them into TF-IDF vectors
        question_vectors = vectorizer.fit_transform(questions)
        vectorization_end_time = time.time()
        logger.info(f"Vectorized {len(questions)} questions with {question_vectors.shape[1]} features in {vectorization_end_time - vectorization_start_time:.2f} seconds.")
        
        logger.info("Starting model and data saving...")
        saving_start_time = time.time()
        # Save the trained vectorizer, question vectors, and corresponding answers
        joblib.dump(vectorizer, 'vectorizer.pkl')
        joblib.dump(question_vectors, 'question_vectors.pkl')
        joblib.dump(answers, 'answers_for_similarity.pkl')
        joblib.dump(questions, 'original_questions_for_debug.pkl') # Save original questions for debugging
        saving_end_time = time.time()
        logger.info(f"Model and data files saved successfully in {saving_end_time - saving_start_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"Error during model training or saving: {e}", exc_info=True)

if __name__ == "__main__":
    logger.info("Starting model training process...")
    total_start_time = time.time()
    train_model()
    total_end_time = time.time()
    logger.info(f"Total model training process finished in {total_end_time - total_start_time:.2f} seconds.")