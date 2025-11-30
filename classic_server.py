import webbrowser
from flask import Flask, request, jsonify, render_template

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import os
import logging
import colorama
from colorama import Fore, Back, Style
import time
from flask_cors import CORS  # Add this import

# Initialize colorama for colored terminal output
colorama.init()

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

# Custom logger with colors
class ColorLogger:
    @staticmethod
    def info(message):
        print(f"{Fore.CYAN}[INFO] {message}{Style.RESET_ALL}")
    
    @staticmethod
    def success(message):
        print(f"{Fore.GREEN}[SUCCESS] {message}{Style.RESET_ALL}")
    
    @staticmethod
    def warning(message):
        print(f"{Fore.YELLOW}[WARNING] {message}{Style.RESET_ALL}")
    
    @staticmethod
    def error(message):
        print(f"{Fore.RED}[ERROR] {message}{Style.RESET_ALL}")
    
    @staticmethod
    def debug(message):
        print(f"{Fore.MAGENTA}[DEBUG] {message}{Style.RESET_ALL}")
    
    @staticmethod
    def request(method, endpoint, status_code=None):
        status = f" - {status_code}" if status_code else ""
        print(f"{Fore.BLUE}[REQUEST] {method} {endpoint}{status}{Style.RESET_ALL}")

# Initialize Flask app
app = Flask(__name__, 
            static_folder='visualization',  # Add visualization as a static folder
            static_url_path='/static/visualization')
CORS(app)  # Enable CORS for all routes
logger = ColorLogger()

# Download NLTK resources if needed
try:
    logger.info("Checking for NLTK resources...")
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    logger.success("NLTK resources found!")
except LookupError:
    logger.warning("NLTK resources not found. Downloading...")
    nltk.download('stopwords')
    nltk.download('wordnet')
    logger.success("NLTK resources downloaded successfully!")

# Global variables
MODEL_PATH = 'best_models\\fake_news_detector_optimized.h5'
TOKENIZER_PATH = 'best_models\\tokenizer.pickle'
MAX_LEN = 200

# Initialize model and tokenizer variables
model = None
tokenizer = None



def load_model_and_tokenizer():
    global model, tokenizer
    
    try:
        # Load model
        logger.info(f"Loading model from {MODEL_PATH}...")
        start_time = time.time()
        model = tf.keras.models.load_model(MODEL_PATH)
        model_load_time = time.time() - start_time
        logger.success(f"Model loaded successfully in {model_load_time:.2f} seconds!")
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {TOKENIZER_PATH}...")
        start_time = time.time()
        with open(TOKENIZER_PATH, 'rb') as handle:
            tokenizer = pickle.load(handle)
        tokenizer_load_time = time.time() - start_time
        logger.success(f"Tokenizer loaded successfully in {tokenizer_load_time:.2f} seconds!")
        
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model or tokenizer: {str(e)}")
        raise

@app.before_request
def before_request():
    global model, tokenizer
    if model is None or tokenizer is None:
        logger.warning("Model or tokenizer not loaded. Loading now...")
        model, tokenizer = load_model_and_tokenizer()
    
    # Log request details
    if request.endpoint != 'static':
        logger.request(request.method, request.path)

# Text preprocessing function
def preprocess_text(text, lemmatize=True):
    """
    Preprocess text data
    """
    if pd.isna(text):
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back to text
    text = ' '.join(tokens)
    
    return text

# Prediction function
def predict_fake_news(title, text):
    """
    Make prediction on whether an article is fake or real
    """
    logger.info(f"Processing prediction request - Title: '{title[:30]}...' Text length: {len(text)} chars")
    
    # Combine title and text
    content = title + ' ' + text
    
    # Preprocess
    logger.debug("Preprocessing text...")
    start_time = time.time()
    processed_content = preprocess_text(content)
    preprocess_time = time.time() - start_time
    logger.debug(f"Text preprocessing completed in {preprocess_time:.2f} seconds")
    
    # Tokenize
    logger.debug("Tokenizing text...")
    start_time = time.time()
    sequence = tokenizer.texts_to_sequences([processed_content])
    tokenize_time = time.time() - start_time
    logger.debug(f"Tokenization completed in {tokenize_time:.2f} seconds")
    
    # Pad sequence
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Predict
    logger.debug("Running prediction...")
    start_time = time.time()
    prediction = model.predict(padded_sequence, verbose=0)[0][0]
    predict_time = time.time() - start_time
    logger.debug(f"Prediction completed in {predict_time:.2f} seconds")
    
    # Return result
    result = {
        'prediction': float(prediction),
        'label': 'FAKE' if prediction > 0.5 else 'REAL',
        'confidence': float(prediction) if prediction > 0.5 else float(1 - prediction)
    }
    
    logger.success(f"Prediction result: {result['label']} with {result['confidence']:.2%} confidence")
    
    return result

# Routes
@app.route('/')
def home():
    logger.info("Serving home page")
    webbrowser.open("C:\\Users\\jkrut\\OneDrive\\Desktop\\Fake_News\\templates\\index.html")


def open_browser():
    chrome_path = 'C:/Program Files/Google/Chrome/Application/chrome.exe %s'
    webbrowser.get(chrome_path).open("http://127.0.0.1:5000")
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        # Get data from request
        data = request.get_json()
        title = data.get('title', '')
        text = data.get('text', '')
        
        # Check if text is empty
        if not text:
            logger.warning("API request received with empty text")
            return jsonify({
                'error': 'No text provided',
                'status': 'error'
            }), 400
        
        # Make prediction
        result = predict_fake_news(title, text)
        
        # Return result
        logger.request(request.method, request.path, 200)
        return jsonify({
            'result': result,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify the API is working
    """
    model_status = "LOADED" if model is not None else "NOT LOADED"
    tokenizer_status = "LOADED" if tokenizer is not None else "NOT LOADED"
    
    logger.info(f"Health check - Model: {model_status}, Tokenizer: {tokenizer_status}")
    
    return jsonify({
        'status': 'UP',
        'model_loaded': model is not None,
        'tokenizer_loaded': tokenizer is not None
    })

if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info("Starting Fake News Detection API Server")
    logger.info("=" * 50)
    logger.info(f"Server will be available at http://localhost:5000")
    logger.info(f"Model path: {MODEL_PATH}")
    logger.info(f"Tokenizer path: {TOKENIZER_PATH}")
    logger.info("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)