# A Regularized LSTM Method for Detecting Fake News Articles

This project implements a Fake News Detection system using a Regularized LSTM deep learning model. 
It preprocesses text data, trains an optimized LSTM-based model, and provides predictions on whether an article is real or fake.

## 🚀 Features
- Text preprocessing (tokenization, lemmatization, padding)
- Regularized LSTM model for classification
- Accuracy: ~97–99%
- Flask/Django integration for deployment

## 📂 Project Structure
- **train_model.py** → Training scripts
- **fake_news_detector_optimized.h5** → Saved model
- **tokenizer.pickle** → Tokenizer
- **templates/** → HTML templates for web app
- **static/** → CSS, JS files
- **visualizations/** → Graphs and performance results

## ⚙️ How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/Mrkruthik1/A-Regularized-LSTM-Method-for-Detecting-Fake-News-Articles.git
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   python main.py
   ```

## 📊 Results
- LSTM baseline: 92%
- Enhanced LSTM: 97%
- Optimized model: 99%

## 👨‍💻 Author
Kruthik – Machine Learning Enthusiast

