# Sentiment Analysis App

A machine learning-powered sentiment analysis application that categorizes text into positive, negative, or neutral sentiments with 90% precision. Built with Streamlit for an interactive web interface.

## Features

- **Real-time Sentiment Analysis**: Analyze text sentiment instantly
- **Three-class Classification**: Categorizes text as Positive, Negative, or Neutral
- **Advanced Text Processing**: Includes cleaning, tokenization, stopword removal, and lemmatization
- **Interactive UI**: User-friendly Streamlit interface
- **High Accuracy**: 90% precision using SVM model with TF-IDF vectorization

## Tech Stack

- **Machine Learning**: scikit-learn (SVM model)
- **NLP Processing**: NLTK, spaCy
- **Web Framework**: Streamlit
- **Text Vectorization**: TF-IDF Vectorizer
- **Python**: 3.x

## Prerequisites

- Python 3.7 or higher
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

3. Download required NLTK data:

```python
import nltk
nltk.download('stopwords')
```

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

3. Enter your text in the text area

4. Click the "Predict" button to get sentiment analysis results

## Text Processing Pipeline

The application processes text through the following steps:

1. **Text Cleaning**: Converts to lowercase and removes extra whitespace
2. **Punctuation Removal**: Strips all punctuation marks
3. **Tokenization**: Splits text into individual tokens
4. **Stopword Removal**: Removes common English stopwords
5. **Lemmatization**: Reduces words to their base form using spaCy

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
- Batch text processing
- Export results to CSV/JSON
- Model retraining interface
- Confidence score display
- Historical analysis tracking

## License

This project is open source and available for educational purposes.
