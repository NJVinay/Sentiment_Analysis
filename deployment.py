import io
import json
import string

import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from PIL import Image

st.set_page_config(page_title="Sentiment Analysis App", page_icon="💬", layout="centered")

LABELS = {-1: "Negative", 0: "Neutral", 1: "Positive"}
LABEL_STYLES = {
    "Negative": ("🔴", "error"),
    "Neutral": ("🟡", "info"),
    "Positive": ("🟢", "success"),
}


@st.cache_resource(show_spinner="Downloading NLTK data...")
def setup_nltk():
    for resource in ("stopwords", "punkt", "punkt_tab", "wordnet", "omw-1.4"):
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass  # punkt_tab only exists on newer NLTK versions
    return set(stopwords.words("english")), WordNetLemmatizer()


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    import pickle

    with open("svm_model.pkl", "rb") as file:
        model = pickle.load(file)
    with open("tfidf_vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)
    return model, vectorizer


stop_words, lemmatizer = setup_nltk()
model, vectorizer = load_model()


def preprocess_text(text):
    text = text.lower().strip()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    tokens = word_tokenize(text)
    processed = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(processed)


def predict_with_confidence(texts):
    """Predict sentiment for a list of raw texts. Returns list of (label, confidence)."""
    processed = [preprocess_text(t) for t in texts]
    vectors = vectorizer.transform(processed)
    predictions = model.predict(vectors)

    confidences = [None] * len(texts)
    if hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(vectors)
            confidences = probas.max(axis=1).tolist()
        except Exception:
            pass
    if confidences[0] is None and hasattr(model, "decision_function"):
        try:
            import numpy as np

            scores = model.decision_function(vectors)
            if scores.ndim == 1:
                scores = np.column_stack([-scores, scores])
            # Softmax over class scores as a pseudo-probability
            exp = np.exp(scores - scores.max(axis=1, keepdims=True))
            confidences = (exp.max(axis=1) / exp.sum(axis=1)).tolist()
        except Exception:
            pass

    return [(LABELS.get(p, str(p)), c) for p, c in zip(predictions, confidences)]


def show_result(label, confidence):
    emoji, style = LABEL_STYLES[label]
    message = f"{emoji} The sentiment of the given text is: **{label}**"
    getattr(st, style)(message)
    if confidence is not None:
        st.progress(confidence, text=f"Confidence: {confidence:.1%}")


def results_dataframe(texts, results):
    return pd.DataFrame(
        {
            "Text": texts,
            "Sentiment": [label for label, _ in results],
            "Confidence": [
                round(conf, 4) if conf is not None else None for _, conf in results
            ],
        }
    )


def download_buttons(df, key_prefix):
    col_csv, col_json = st.columns(2)
    with col_csv:
        st.download_button(
            "⬇️ Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="sentiment_results.csv",
            mime="text/csv",
            key=f"{key_prefix}_csv",
        )
    with col_json:
        st.download_button(
            "⬇️ Download JSON",
            json.dumps(df.to_dict(orient="records"), indent=2).encode("utf-8"),
            file_name="sentiment_results.json",
            mime="application/json",
            key=f"{key_prefix}_json",
        )


if "history" not in st.session_state:
    st.session_state.history = []

st.title("Sentiment Analysis App")
st.markdown("By Naram Jyotir Vinay")
try:
    image = Image.open("Sentiment-Analysis1.jpg.webp")
    st.image(image, width=700)
except FileNotFoundError:
    pass

tab_single, tab_batch, tab_history = st.tabs(["✏️ Single Text", "📄 Batch Analysis", "🕘 History"])

with tab_single:
    st.subheader("Enter your text here:")
    user_input = st.text_area("Input Text", label_visibility="collapsed")

    if st.button("Predict", type="primary"):
        if user_input.strip():
            with st.spinner("Analyzing..."):
                label, confidence = predict_with_confidence([user_input])[0]
            show_result(label, confidence)
            st.session_state.history.append(
                {
                    "Text": user_input,
                    "Sentiment": label,
                    "Confidence": round(confidence, 4) if confidence is not None else None,
                }
            )
        else:
            st.warning("Please enter a text for prediction.")

with tab_batch:
    st.subheader("Analyze multiple texts at once")
    st.caption(
        "Paste one text per line, or upload a CSV/TXT file. "
        "For CSV files, the first column is used."
    )

    batch_texts = []
    uploaded = st.file_uploader("Upload a file", type=["csv", "txt"])
    pasted = st.text_area("Or paste texts (one per line)", height=150)

    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df_in = pd.read_csv(uploaded)
                batch_texts = df_in.iloc[:, 0].dropna().astype(str).tolist()
            else:
                content = io.StringIO(uploaded.getvalue().decode("utf-8", errors="replace"))
                batch_texts = [line.strip() for line in content if line.strip()]
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
    elif pasted.strip():
        batch_texts = [line.strip() for line in pasted.splitlines() if line.strip()]

    if st.button("Analyze Batch", type="primary"):
        if batch_texts:
            with st.spinner(f"Analyzing {len(batch_texts)} texts..."):
                results = predict_with_confidence(batch_texts)
            df = results_dataframe(batch_texts, results)

            counts = df["Sentiment"].value_counts()
            col1, col2, col3 = st.columns(3)
            col1.metric("🟢 Positive", int(counts.get("Positive", 0)))
            col2.metric("🟡 Neutral", int(counts.get("Neutral", 0)))
            col3.metric("🔴 Negative", int(counts.get("Negative", 0)))

            st.dataframe(df, use_container_width=True)
            download_buttons(df, "batch")

            st.session_state.history.extend(df.to_dict(orient="records"))
        else:
            st.warning("Please upload a file or paste some texts first.")

with tab_history:
    st.subheader("Analysis history")
    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist, use_container_width=True)
        download_buttons(df_hist, "history")
        if st.button("Clear history"):
            st.session_state.history = []
            rerun = getattr(st, "rerun", None) or st.experimental_rerun
            rerun()
    else:
        st.caption("No predictions yet. Results from both tabs will appear here.")
