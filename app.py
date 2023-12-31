import streamlit as st
import numpy as np
import pickle
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
import re

model = pickle.load(open('logistic_regression.pkl', 'rb'))
lb = pickle.load(open('label_encoder.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

stopwords = set(nltk.corpus.stopwords.words('english'))
def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(w) for w in text if w not in stopwords]
    return  " ".join(text)


def prediction(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf_vectorizer.transform([cleaned_text])
    predicted_label = model.predict(input_vectorized)[0]
    predicted_emotion = lb.inverse_transform([predicted_label])[0]
    label = np.max(model.predict(input_vectorized)[0])
    return predicted_emotion, label

st.title("Emotions Detection App")
st.write(['Joy', 'Fear', 'Love', 'Anger', 'Sadness', 'Suprise'])
input_text = st.text_input("enter your text here")

if st.button("predict"):
    predicted_emotion, label = prediction(input_text)
    st.write("Predicted labe:", predicted_emotion)
    st.write("predicted label:", label)

