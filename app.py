import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("sentiment_reviewer_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# Streamlit UI
st.set_page_config(page_title="IMDB Sentiment Reviewer", page_icon="🎬", layout="centered")
st.title("🎬 IMDB Sentiment Reviewer")
st.write("Enter a movie review below and see if the model thinks it's **Positive** or **Negative**.")

review_text = st.text_area("Enter your review:")

if st.button("Predict Sentiment"):
    if review_text.strip() != "":
        review_vectorized = vectorizer.transform([review_text])

        prediction = model.predict(review_vectorized)[0]

        sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
        st.success(f"Prediction: {sentiment}")
    else:
        st.warning("Please enter a review before predicting.")
