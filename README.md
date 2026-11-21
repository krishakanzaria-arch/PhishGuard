# PhishGuard – Machine Learning-Based Email Detection System

PhishGuard is a machine learning-based web application designed to identify phishing and spam emails. It analyzes email content and predicts whether an email is likely to be legitimate or spam.

The project uses Natural Language Processing (NLP) and machine learning classification algorithms to detect suspicious email messages.

## Features

- Email classification using machine learning.
- Interactive web interface built with Streamlit.
- TF-IDF text vectorization.
- Multiple machine learning algorithms for comparison.
- Email prediction with clear results.
- Sample emails for testing.

## Technologies Used

- **Programming Language:** Python
- **Frontend:** Streamlit
- **Machine Learning:** Scikit-learn
- **NLP:** TF-IDF Vectorization
- **Libraries:** Pandas, NumPy, Joblib

## Machine Learning Algorithms

The following algorithms were implemented and evaluated:

- Logistic Regression
- Multinomial Naive Bayes
- Linear Support Vector Machine (SVM)

Linear SVM achieved an accuracy of approximately **99.72%** on the project's test dataset.

## Project Structure

```text
PhishGuard/
│
├── app.py
├── phishguard_model.pkl
├── phishguard_vectorizer.pkl
└── README.md
```

## How to Run

1. Clone this repository.
2. Install the required dependencies.
3. Run the Streamlit application.

```bash
pip install streamlit pandas numpy scikit-learn joblib
streamlit run app.py
```

## Project Purpose

This project was developed as part of an academic Machine Learning course to explore email classification, natural language processing, and practical applications of machine learning.

## Disclaimer

PhishGuard is an educational project. Its predictions are not guaranteed to identify every phishing or spam email. Users should always verify suspicious messages independently.

---

**Developed by Krisha Kanzaria**
