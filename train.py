import mysql.connector
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import nltk
from nltk.tokenize import word_tokenize
import os
from db_config import db_settings  # Assumes this contains DB credentials
from my_stopwords import burmese_stopwords  # Assumes this contains Burmese stopwords

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
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("NLTK 'punkt_tab' tokenizer not found. Downloading...")
    nltk.download('punkt_tab', download_dir=custom_nltk_data_path)
    print("NLTK 'punkt_tab' tokenizer downloaded.")

def burmese_preprocessor(text):
    """Custom preprocessor for Burmese text using NLTK word_tokenize and stopword removal."""
    text = str(text)
    tokens = word_tokenize(text)
    processed_tokens = [word.strip() for word in tokens if word.strip() and word.strip() not in burmese_stopwords]
    return " ".join(processed_tokens)

def load_data():
    """Loads questions and answers from the database."""
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(**db_settings)
        if not conn.is_connected():
            raise Exception("Database connection established but not active.")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT q.question_text, a.answer_text
            FROM questions q
            JOIN answers a ON q.id = a.question_id
        """)
        data = cursor.fetchall()
        if not data:
            raise ValueError("No training data found in DB. Please add questions and answers.")
        questions, answers = zip(*data)
        return list(questions), list(answers)
    except mysql.connector.Error as e:
        error_msg = f"Database error: {e}"
        print(error_msg)
        raise Exception(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error loading data: {e}"
        print(error_msg)
        raise Exception(error_msg)
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

def train_and_save():
    """Trains the TF-IDF vectorizer and saves question vectors."""
    questions, answers = load_data()
    print(f"Training TF-IDF model with {len(questions)} question-answer pairs...")

    vectorizer = TfidfVectorizer(
        preprocessor=burmese_preprocessor,
        stop_words=burmese_stopwords,
        max_features=2000,
        min_df=2,
        max_df=0.9
    )

    try:
        question_vectors = vectorizer.fit_transform(questions)
        joblib.dump(vectorizer, "vectorizer.pkl")
        joblib.dump(question_vectors, "question_vectors.pkl")
        joblib.dump(answers, "answers_for_similarity.pkl")
        print("✅ TF-IDF Vectorizer and question vectors trained and saved.")
        print("Run 'streamlit run run_app.py' to test the chatbot.")
    except Exception as e:
        error_msg = f"Error during model training: {e}"
        print(error_msg)
        raise Exception(error_msg)

if __name__ == "__main__":
    try:
        train_and_save()
    except ValueError as e:
        print(f"Error: {e}")
        print("Suggestion: Ensure your 'questions' and 'answers' tables in the database are populated.")
    except Exception as e:
        print(f"An unexpected error occurred during training: {e}")