# Fix collections deprecation issues
import collections.abc
collections.Sequence = collections.abc.Sequence
collections.Iterable = collections.abc.Iterable

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
# Rest of your imports...import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import os
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create a directory for saving visualizations
os.makedirs('visualizations', exist_ok=True)

# Data Loading
def load_data(fake_path, real_path):
    """
    Load and combine fake and real news datasets
    """
    # Load datasets
    fake_df = pd.read_csv(fake_path)
    real_df = pd.read_csv(real_path)
    
    # Add labels (1 for fake, 0 for real)
    fake_df['label'] = 1
    real_df['label'] = 0
    
    # Combine datasets
    df = pd.concat([fake_df, real_df], axis=0, ignore_index=True)
    
    return df

# Text Preprocessing
def preprocess_text(text, stem=False, lemmatize=False):
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
    
    # Stemming
    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    
    # Lemmatization
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back to text
    text = ' '.join(tokens)
    
    return text

# Feature Engineering
def prepare_features(df, max_words=10000, max_len=200, use_title=True):
    """
    Prepare features for model training
    """
    # Combine title and text if required
    if use_title:
        df['content'] = df['title'] + ' ' + df['text']
    else:
        df['content'] = df['text']
    
    # Apply preprocessing
    df['processed_content'] = df['content'].apply(
        lambda x: preprocess_text(x, stem=False, lemmatize=True)
    )
    
    # Create tokenizer
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['processed_content'])
    
    # Convert text to sequences
    sequences = tokenizer.texts_to_sequences(df['processed_content'])
    
    # Pad sequences
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    
    return padded_sequences, tokenizer

# Model Building Functions
def build_baseline_lstm(vocab_size, embedding_dim=100, max_len=200):
    """
    Build baseline LSTM model as described in the methodology
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
        Dropout(0.2),
        LSTM(150, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    return model

def build_enhanced_lstm(vocab_size, embedding_dim=100, max_len=200):
    """
    Build enhanced LSTM model with regularization
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
        Dropout(0.3),
        LSTM(150, return_sequences=False, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )
    
    return model

def build_optimized_model(vocab_size, embedding_dim=100, max_len=200):
    """
    Build optimized deep learning model
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
        Dropout(0.3),
        LSTM(150, return_sequences=True, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        LSTM(100, return_sequences=False, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1, activation='sigmoid')
    ])
    
    # AdamW optimizer from tensorflow_addons can be used but requires extra dependency
    # For simplicity, using Adam with weight decay
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model

# Model Evaluation
def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model and return metrics
    """
    # Predict
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Print results
    print(f"\n{model_name} Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], 
                yticklabels=['Real', 'Fake'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'visualizations/confusion_matrix_{model_name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

# Visualization Functions
def plot_training_history(histories, model_names):
    """
    Plot training history for multiple models
    """
    # Create a figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot accuracy
    for i, (history, name) in enumerate(zip(histories, model_names)):
        axes[0].plot(history.history['accuracy'], label=f'{name} - Train', linestyle='-')
        axes[0].plot(history.history['val_accuracy'], label=f'{name} - Validation', linestyle='--')
    
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Plot loss
    for i, (history, name) in enumerate(zip(histories, model_names)):
        axes[1].plot(history.history['loss'], label=f'{name} - Train', linestyle='-')
        axes[1].plot(history.history['val_loss'], label=f'{name} - Validation', linestyle='--')
    
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig('visualizations/training_history_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_metrics_comparison(metrics_list, model_names):
    """
    Plot comparison of metrics for different models
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Create a figure with 4 subplots (2x2 grid)
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2)
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = plt.subplot(gs[i])
        values = [m[metric] for m in metrics_list]
        
        # Create bar plot
        bars = ax.bar(model_names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_title(f'{metric.capitalize()} Comparison')
        ax.set_ylim(0, 1.1)
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('visualizations/metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a single bar chart with all metrics
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(model_names))
    width = 0.2
    
    # Plot bars for each metric
    for i, metric in enumerate(metrics):
        values = [m[metric] for m in metrics_list]
        ax.bar(x + (i - 1.5) * width, values, width, label=metric.capitalize())
    
    ax.set_title('Performance Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('visualizations/combined_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_metrics_table(metrics_list, model_names):
    """
    Create and display a table of metrics
    """
    # Create DataFrame
    metrics_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': [m['accuracy'] for m in metrics_list],
        'Precision': [m['precision'] for m in metrics_list],
        'Recall': [m['recall'] for m in metrics_list],
        'F1-Score': [m['f1'] for m in metrics_list]
    })
    
    # Style the DataFrame for better visualization
    styled_df = metrics_df.style.format({
        'Accuracy': '{:.4f}',
        'Precision': '{:.4f}',
        'Recall': '{:.4f}',
        'F1-Score': '{:.4f}'
    }).background_gradient(cmap='Blues', subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    
    # Save to CSV
    metrics_df.to_csv('visualizations/model_performance_metrics.csv', index=False)
    
    # Display the table
    print("\nModel Performance Metrics Table:")
    print(metrics_df.to_string(index=False))
    
    return metrics_df

# Main Training Function
def train_fake_news_models(fake_path, real_path):
    """
    Main function to train all three models
    """
    print("Loading and preprocessing data...")
    # Load data
    df = load_data(fake_path, real_path)
    
    # Check for missing values
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # Drop duplicates
    df.drop_duplicates(subset=['text'], keep='first', inplace=True)
    print(f"Dataset shape after removing duplicates: {df.shape}")
    
    # Visualize class distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='label', data=df, palette=['#1f77b4', '#ff7f0e'])
    plt.title('Class Distribution in Dataset')
    plt.xlabel('Class (0: Real, 1: Fake)')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Real News', 'Fake News'])
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.savefig('visualizations/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Prepare features
    X, tokenizer = prepare_features(df)
    y = df['label'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    # Get vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary size: {vocab_size}")
    
    # Train Model 1: Baseline LSTM
    print("\nTraining Baseline LSTM Model...")
    model1 = build_baseline_lstm(vocab_size)
    model1.summary()
    
    history1 = model1.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    print("\nEvaluating Baseline LSTM Model...")
    metrics1 = evaluate_model(model1, X_test, y_test, "Baseline LSTM")
    
    # Train Model 2: Enhanced LSTM with Regularization
    print("\nTraining Enhanced LSTM Model with Regularization...")
    model2 = build_enhanced_lstm(vocab_size)
    model2.summary()
    
    history2 = model2.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    print("\nEvaluating Enhanced LSTM Model...")
    metrics2 = evaluate_model(model2, X_test, y_test, "Enhanced LSTM")
    
    # Train Model 3: Optimized Deep Learning Model
    print("\nTraining Optimized Deep Learning Model...")
    model3 = build_optimized_model(vocab_size)
    model3.summary()
    
    history3 = model3.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    print("\nEvaluating Optimized Deep Learning Model...")
    metrics3 = evaluate_model(model3, X_test, y_test, "Optimized Model")
    
    # Compare model performance
    model_names = ['Baseline LSTM', 'Enhanced LSTM', 'Optimized Model']
    metrics_list = [metrics1, metrics2, metrics3]
    histories = [history1, history2, history3]
    
    # Create metrics table
    metrics_df = create_metrics_table(metrics_list, model_names)
    
    # Plot training history
    plot_training_history(histories, model_names)
    
    # Plot metrics comparison
    plot_metrics_comparison(metrics_list, model_names)
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    
    # Get predictions for each model
    y_pred_prob1 = model1.predict(X_test).flatten()
    y_pred_prob2 = model2.predict(X_test).flatten()
    y_pred_prob3 = model3.predict(X_test).flatten()
    
    # Calculate false positive rate and true positive rate
    from sklearn.metrics import roc_curve, auc
    fpr1, tpr1, _ = roc_curve(y_test, y_pred_prob1)
    fpr2, tpr2, _ = roc_curve(y_test, y_pred_prob2)
    fpr3, tpr3, _ = roc_curve(y_test, y_pred_prob3)
    
    # Calculate AUC
    roc_auc1 = auc(fpr1, tpr1)
    roc_auc2 = auc(fpr2, tpr2)
    roc_auc3 = auc(fpr3, tpr3)
    
    # Plot ROC curves
    plt.plot(fpr1, tpr1, label=f'Baseline LSTM (AUC = {roc_auc1:.4f})')
    plt.plot(fpr2, tpr2, label=f'Enhanced LSTM (AUC = {roc_auc2:.4f})')
    plt.plot(fpr3, tpr3, label=f'Optimized Model (AUC = {roc_auc3:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig('visualizations/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save the best model (Optimized Model)
    model3.save('fake_news_detector_optimized.h5')
    
    # Save tokenizer
    import pickle
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Training complete. Best model saved as 'fake_news_detector_optimized.h5'")
    print("All visualizations saved in the 'visualizations' directory")
    
    return {
        'baseline_model': model1,
        'enhanced_model': model2,
        'optimized_model': model3,
        'tokenizer': tokenizer,
        'metrics': {
            'baseline': metrics1,
            'enhanced': metrics2,
            'optimized': metrics3
        },
        'histories': {
            'baseline': history1,
            'enhanced': history2,
            'optimized': history3
        }
    }

if __name__ == "__main__":
    # Set file paths
    fake_news_path = "Fake.csv"
    real_news_path = "True.csv"
    
    # Run the full training pipeline
    results = train_fake_news_models(fake_news_path, real_news_path)