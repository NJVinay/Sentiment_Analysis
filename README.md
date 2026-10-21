# Sentiment Analysis App

**Live Demo**: https://sentiment-analysis-nlp-app.streamlit.app/  

A machine learning-powered sentiment analysis application that categorizes text into positive, negative, or neutral sentiments with 90% precision. Built with Streamlit for an interactive web interface.

## Features

- **Real-time Sentiment Analysis**: Analyze text sentiment instantly
- **Three-class Classification**: Categorizes text as Positive, Negative, or Neutral
- **Confidence Scores**: Each prediction comes with a confidence indicator
- **Batch Analysis**: Analyze many texts at once — paste one per line or upload a CSV/TXT file
- **Export Results**: Download single, batch, or history results as CSV or JSON
- **History Tracking**: All predictions in a session are collected in a History tab
- **Advanced Text Processing**: Includes cleaning, tokenization, stopword removal, and lemmatization
- **Interactive UI**: Tabbed Streamlit interface with cached model loading for fast reruns
- **High Accuracy**: 90% precision using SVM model with TF-IDF vectorization

## Tech Stack

- **Machine Learning**: scikit-learn (SVM model)
- **NLP Processing**: NLTK
- **Web Framework**: Streamlit
- **Text Vectorization**: TF-IDF Vectorizer
- **Python**: 3.10 or 3.11 (scikit-learn 1.2.2 does not support 3.12+)

## Prerequisites

- Python 3.10 or 3.11
- pip package manager

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd Sentiment_Analysis
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. NLTK data (stopwords, punkt, wordnet, omw-1.4) is downloaded automatically on first launch.

## Required Files

Ensure the following files are in the project directory:

- `svm_model.pkl` - Pre-trained SVM model
- `tfidf_vectorizer.pkl` - Fitted TF-IDF vectorizer
- `Sentiment-Analysis1.jpg.webp` - Header image

## Usage

1. Run the Streamlit app:

```bash
streamlit run deployment.py
```

2. Open your browser and navigate to the local URL (typically `http://localhost:8501`)

3. Use one of the three tabs:
   - **Single Text**: Enter text and click "Predict" to get the sentiment with a confidence score
   - **Batch Analysis**: Paste multiple texts (one per line) or upload a CSV/TXT file, then click "Analyze Batch" to get a summary and a results table, downloadable as CSV or JSON
   - **History**: Review and export every prediction made in the current session

## Text Processing Pipeline

The application processes text through the following steps:

1. **Text Cleaning**: Converts to lowercase and removes extra whitespace
2. **Punctuation Removal**: Strips all punctuation marks
3. **Tokenization**: Splits text into individual tokens
4. **Stopword Removal**: Removes common English stopwords
5. **Lemmatization**: Reduces words to their base form using NLTK

## Model Details

- **Algorithm**: Support Vector Machine (SVM)
- **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Output Classes**:
  - `-1`: Negative sentiment
  - `0`: Neutral sentiment
  - `1`: Positive sentiment

## Project Structure

```
Sentiment_Analysis/
├── deployment.py              # Main Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                 # Project documentation
├── svm_model.pkl            # Pre-trained SVM model
├── tfidf_vectorizer.pkl     # Fitted vectorizer
└── Sentiment-Analysis1.jpg.webp  # UI image
```

## Author

**Naram Jyotir Vinay**

## Future Enhancements

- Support for multiple languages
- Model retraining interface
- Persistent (cross-session) history storage

## License

This project is open source and available for educational purposes.
