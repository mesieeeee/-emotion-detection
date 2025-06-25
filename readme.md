# 😄 Emotion Detection from Text

A hybrid **Machine Learning** and **Deep Learning** solution for detecting emotions from user comments in real time. Trained on a labeled dataset of 16,000+ text samples across 5+ emotional categories, this system is built using classic ML classifiers and an **LSTM-based neural network**, and deployed with **Streamlit** for real-time inference and personalization.

---

## 🧠 Key Features

- ✨ **Text Preprocessing**: Cleaning, tokenization, stopword removal, lemmatization
- 📊 **Feature Engineering**: TF-IDF vectorization for ML models
- 🤖 **ML Models**: Logistic Regression, SVM, Random Forest, Naive Bayes
- 🔍 **DL Model**: LSTM (RNN) for sequential learning on tokenized text
- 🎯 **Performance**: Achieved up to **84% accuracy** using optimized hyperparameters
- 🌐 **Web Deployment**: Interactive Streamlit app for live predictions
- 🎁 **Bonus**: Personalized product recommendation suggestions based on detected emotion

---

## 🧰 Tech Stack

| Category           | Tools Used                               |
|--------------------|-------------------------------------------|
| Language           | Python                                    |
| ML Models          | Scikit-learn (LogReg, SVM, RF, NB)        |
| Deep Learning      | Keras, TensorFlow (LSTM)                  |
| Feature Extraction | TF-IDF Vectorizer                         |
| Deployment         | Streamlit                                 |

---

## 🔄 Workflow
**Preprocessing**:
Cleaned raw comments using regex, tokenization, stopword removal, and lemmatization.

**Feature Representation**:
Used TF-IDF to convert text into numerical feature vectors for ML models.

**Model Training**:
Trained 4 traditional ML classifiers and tuned hyperparameters using cross-validation.

**Deep Learning**:
Built an LSTM model to learn long-term dependencies from sequential word embeddings.

**Evaluation**:
Achieved up to 84% accuracy and consistent performance across metrics.

**Deployment**:
Created a user-friendly Streamlit interface for real-time emotion classification.

