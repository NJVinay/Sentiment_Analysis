import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

with open("svm_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

def clean_text(text):
    text = text.lower()
    return text.strip()

def remove_punctuation(text):
    punctuation_free = "".join([i for i in text if i not in string.punctuation])
    return punctuation_free

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords and apply lemmatization
    processed = [lemmatizer.lemmatize(word) for word in tokens if word.lower() not in stop_words]
    return " ".join(processed)

st.title("Sentiment Analysis App")
st.markdown("By Naram Jyotir Vinay")
image = Image.open("Sentiment-Analysis1.jpg.webp")
st.image(image, width=700)  # Updated to use the `width` parameter

st.subheader("Enter your text here:")
user_input = st.text_area("Input Text", label_visibility="collapsed")  # Updated label

if user_input:
    user_input = clean_text(user_input)
    user_input = remove_punctuation(user_input)
    user_input = preprocess_text(user_input)

if st.button("Predict"):
    if user_input:
        text_vectorized = vectorizer.transform([user_input])
        prediction = model.predict(text_vectorized)[0]
        st.header("Prediction:")
        if prediction == -1:
            st.subheader("The sentiment of the given text is: Negative")
        elif prediction == 0:
            st.subheader("The sentiment of the given text is: Neutral")
        elif prediction == 1:
            st.subheader("The sentiment of the given text is: Positive")
    else:
        st.subheader("Please enter a text for prediction.")
