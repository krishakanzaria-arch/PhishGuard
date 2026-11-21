import math
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from database import (
    initialize_database,
    save_analysis,
    get_history,
    clear_history,
    get_statistics,
)

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="PhishGuard | Email Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "CEAS_08.csv" / "CEAS_08.csv"
initialize_database()

# ---------------- SIDEBAR NAVIGATION ----------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 12px 4px 20px 4px;">
            <div style="font-size: 28px; font-weight: 800; color: #5eead4;">
                🛡️ PhishGuard
            </div>
            <div style="color: #94a3b8; font-size: 13px; margin-top: 5px;">
                EMAIL SECURITY PLATFORM
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        "<p style='color:#94a3b8; font-size:12px; "
        "font-weight:700; letter-spacing:2px;'>WORKSPACE</p>",
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigation",
        [
            "📊  Dashboard",
            "🧠  ML Analytics",
            "🛡️  Email Analyzer",
            "🕘  Analysis History",
        ],
        label_visibility="collapsed",
        key="navigation_page",
    )
    st.divider()
    st.markdown(
        """
        <div style="background:#111c2e; border:1px solid #263449;
                    border-radius:12px; padding:14px;">
            <div style="color:#5eead4; font-weight:700;">
                ● System Ready
            </div>
            <div style="color:#94a3b8; font-size:12px; margin-top:6px;">
                Machine-learning model loaded
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------- CUSTOM DESIGN ----------------
st.markdown(
    """
    <style>
        .stApp { background-color: #0b1120; color: #e5e7eb; }
        [data-testid="stHeader"] { background-color: #0b1120; }
        .block-container { max-width: 1200px; padding-top: 2rem; padding-bottom: 3rem; }
        .hero {
            background: linear-gradient(135deg, #111c32, #102a35);
            border: 1px solid #24374b; border-radius: 20px;
            padding: 30px; margin-bottom: 25px;
        }
        .brand { color: #5eead4; font-size: 15px; font-weight: 700;
                 letter-spacing: 3px; margin-bottom: 12px; }
        .hero h1 { color: #f8fafc; font-size: 42px; margin: 0; }
        .hero p, .muted { color: #a8b6c9; font-size: 15px; }
        .section-title { color: #f8fafc; font-size: 22px; font-weight: 700;
                         margin-top: 10px; margin-bottom: 6px; }
        div[data-testid="stTextInput"] label,
        div[data-testid="stTextArea"] label { color: #dbeafe !important; font-weight: 600; }
        div[data-testid="stTextInput"] input,
        div[data-testid="stTextArea"] textarea {
            background-color: #111c2e !important; color: #f8fafc !important;
            border: 1px solid #33465e !important; border-radius: 10px !important;
        }
        div.stButton > button {
            background: linear-gradient(90deg, #0d9488, #14b8a6); color: white;
            border: none; border-radius: 10px; min-height: 48px;
            font-size: 16px; font-weight: 700; width: 100%;
        }
        div.stButton > button:hover { background: #0f766e; color: white; border: 1px solid #5eead4; }
        .result-card { background-color: #111c2e; border: 1px solid #33465e;
                       border-radius: 16px; padding: 22px; margin-top: 16px; }
        .safe-result { border-left: 5px solid #34d399; }
        .spam-result { border-left: 5px solid #fb7185; }
        .footer { border-top: 1px solid #263449; margin-top: 40px; padding-top: 18px;
                  color: #8293a8; font-size: 12px; text-align: center; }
        #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- LOAD PRODUCTION MODEL ----------------
@st.cache_resource
def load_model():
    model = joblib.load(BASE_DIR / "phishguard_model.pkl")
    vectorizer = joblib.load(BASE_DIR / "phishguard_vectorizer.pkl")
    return model, vectorizer


try:
    model, vectorizer = load_model()
except Exception as error:
    st.error(
        "Could not load the model files. Make sure both "
        "phishguard_model.pkl and phishguard_vectorizer.pkl "
        "are in the same folder as app.py."
    )
    st.stop()

# ---------------- DATASET + MODEL EVALUATION FOR ANALYTICS ----------------
@st.cache_data(show_spinner="Preparing dataset and evaluating ML models for the analytics page...")
def load_dataset():
    if not DATASET_PATH.exists():
        return None
    dataset = pd.read_csv(DATASET_PATH)
    return dataset


@st.cache_data(show_spinner="Evaluating three classifiers and tuning Linear SVM...")
def evaluate_models():
    """Reproduce the notebook split/TF-IDF setup and tune Linear SVM on training data only."""
    raw = pd.read_csv(DATASET_PATH)
    data = raw.copy()
    data["subject"] = data["subject"].fillna("").astype(str)
    data["body"] = data["body"].fillna("").astype(str)
    data["email_text"] = (
        data["subject"].str.strip() + " " + data["body"].str.strip()
    ).str.strip()
    data = data[data["email_text"].str.len() > 0].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        data["email_text"],
        data["label"],
        test_size=0.20,
        random_state=42,
        stratify=data["label"],
    )
    tfidf = TfidfVectorizer(
        lowercase=True,
        max_features=20000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Basic hyperparameter tuning: test three C values with 3-fold CV on a
    # reproducible subset of training rows, never on the held-out test set.
    tune_count = min(6000, X_train_tfidf.shape[0])
    rng = np.random.RandomState(42)
    tune_indices = rng.choice(X_train_tfidf.shape[0], size=tune_count, replace=False)
    X_tune = X_train_tfidf[tune_indices]
    y_tune = y_train.iloc[tune_indices]
    parameter_grid = {"C": [0.25, 1.0, 4.0]}
    grid_search = GridSearchCV(
        estimator=LinearSVC(random_state=42),
        param_grid=parameter_grid,
        cv=3,
        scoring="f1_weighted",
        n_jobs=1,
        return_train_score=False,
    )
    grid_search.fit(X_tune, y_tune)
    best_c = float(grid_search.best_params_["C"])
    tuning_results = pd.DataFrame(grid_search.cv_results_)[
        ["param_C", "mean_test_score", "std_test_score"]
    ].rename(columns={
        "param_C": "C parameter",
        "mean_test_score": "Mean CV weighted F1",
        "std_test_score": "CV F1 standard deviation",
    })
    tuning_results["Mean CV weighted F1"] *= 100
    tuning_results["CV F1 standard deviation"] *= 100

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear SVM (tuned)": LinearSVC(C=best_c, random_state=42),
    }
    comparison_rows = []
    details = {}
    for name, candidate in models.items():
        candidate.fit(X_train_tfidf, y_train)

        predictions = candidate.predict(X_test_tfidf)

        # Get prediction scores for ROC-AUC
        if hasattr(candidate, "predict_proba"):
            y_scores = candidate.predict_proba(X_test_tfidf)[:, 1]
        else:
            y_scores = candidate.decision_function(X_test_tfidf)

        fpr, tpr, thresholds = roc_curve(y_test, y_scores)
        auc_score = roc_auc_score(y_test, y_scores)

        report = classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        )

        cm = confusion_matrix(y_test, predictions, labels=[0, 1])

        comparison_rows.append({
            "Algorithm": name,
            "Accuracy (%)": accuracy_score(y_test, predictions) * 100,
            "Precision (%)": report["weighted avg"]["precision"] * 100,
            "Recall (%)": report["weighted avg"]["recall"] * 100,
            "F1-score (%)": report["weighted avg"]["f1-score"] * 100,
            "ROC-AUC Score": auc_score,
        })

        details[name] = {
            "confusion_matrix": cm,
            "report": report,
            "predictions": predictions,
            "fpr": fpr,
            "tpr": tpr,
            "roc_auc": auc_score,
        }

    comparison = pd.DataFrame(comparison_rows).sort_values(
        "Accuracy (%)", ascending=False
    ).reset_index(drop=True)
    return {
        "comparison": comparison,
        "details": details,
        "tuning_results": tuning_results,
        "best_c": best_c,
        "best_cv_f1": float(grid_search.best_score_) * 100,
        "tuning_sample_count": tune_count,
        "y_test": y_test.to_numpy(),
        "test_size": len(y_test),
        "train_size": len(y_train),
        "feature_count": X_train_tfidf.shape[1],
        "cleaned_rows": len(data),
        "raw_rows": len(raw),
    }


@st.cache_data(show_spinner="Running K-Means clustering on a sample of email text...")
def run_kmeans_clustering():
    """Discover text clusters without using labels to fit the clusters."""
    raw = pd.read_csv(DATASET_PATH)
    data = raw.copy()
    data["subject"] = data["subject"].fillna("").astype(str)
    data["body"] = data["body"].fillna("").astype(str)
    data["email_text"] = (data["subject"] + " " + data["body"]).str.strip()
    data = data[data["email_text"].str.len() > 0].copy()
    sample = data.sample(n=min(2500, len(data)), random_state=42).reset_index(drop=True)

    cluster_vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=3000,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        stop_words="english",
    )
    features = cluster_vectorizer.fit_transform(sample["email_text"])
    cluster_model = KMeans(n_clusters=2, random_state=42, n_init=5, max_iter=100)
    clusters = cluster_model.fit_predict(features)

    reducer = TruncatedSVD(n_components=2, random_state=42, n_iter=7)
    coordinates = reducer.fit_transform(features)
    plot_df = pd.DataFrame({
        "Component 1": coordinates[:, 0],
        "Component 2": coordinates[:, 1],
        "Cluster": [f"Cluster {value}" for value in clusters],
        "Dataset label": sample["label"].astype(str).to_numpy(),
        "Subject": sample["subject"].str.slice(0, 120).to_numpy(),
    })

    terms = cluster_vectorizer.get_feature_names_out()
    top_terms = {}
    for cluster_id, center in enumerate(cluster_model.cluster_centers_):
        top_indices = center.argsort()[-10:][::-1]
        top_terms[f"Cluster {cluster_id}"] = [str(terms[index]) for index in top_indices]

    label_names = sample["label"].map({
        0: "Class 0 — classified as legitimate",
        1: "Class 1 — spam / unsolicited",
    }).fillna(sample["label"].astype(str))
    crosstab = pd.crosstab(
        pd.Series([f"Cluster {value}" for value in clusters], name="K-Means cluster"),
        pd.Series(label_names, name="Original dataset label"),
    ).reset_index()
    cluster_counts = pd.Series(clusters).value_counts().sort_index().to_dict()
    return {
        "plot_df": plot_df,
        "top_terms": top_terms,
        "crosstab": crosstab,
        "sample_count": len(sample),
        "feature_count": features.shape[1],
        "inertia": float(cluster_model.inertia_),
        "cluster_counts": {f"Cluster {key}": int(value) for key, value in cluster_counts.items()},
        "explained_variance": float(reducer.explained_variance_ratio_.sum() * 100),
    }


# ---------------- SHARED HEADER ----------------
st.markdown(
    """
    <div class="hero">
        <div class="brand">PHISHGUARD / EMAIL SECURITY</div>
        <h1>Think before you click.</h1>
        <p>Analyze email content using machine learning. Identify messages classified as spam or legitimate.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- DASHBOARD PAGE ----------------
if page == "📊  Dashboard":
    st.markdown('<div class="section-title">Security overview</div>', unsafe_allow_html=True)
    stats = get_statistics()

    metric1, metric2, metric3, metric4 = st.columns(4)
    with metric1:
        st.metric("Total analyses", stats["total_analyses"])
    with metric2:
        st.metric("Spam / unsolicited", stats["spam_count"])
    with metric3:
        st.metric("Classified as legitimate", stats["legitimate_count"])
    with metric4:
        st.metric("Avg. decision strength", f'{stats["average_strength"]:.1f}%')

    st.caption(
        "These metrics describe emails analyzed in this app. Decision strength is not a calibrated probability."
    )

    
    # ---------------- CLASSIFICATION BREAKDOWN ----------------
    st.markdown("### Classification breakdown")

    chart_data = pd.DataFrame({
        "Classification": [
            "Spam / unsolicited",
            "Classified as legitimate"
        ],
        "Count": [
            stats["spam_count"],
            stats["legitimate_count"]
        ]
    })

    if stats["total_analyses"] > 0:
        chart = alt.Chart(chart_data).mark_arc(
            innerRadius=75,
            outerRadius=125
        ).encode(
            theta=alt.Theta(
                field="Count",
                type="quantitative"
            ),
            color=alt.Color(
                field="Classification",
                type="nominal",
                title="Classification"
            ),
            tooltip=[
                alt.Tooltip(
                    "Classification:N",
                    title="Category"
                ),
                alt.Tooltip(
                    "Count:Q",
                    title="Number of emails"
                )
            ]
        ).properties(
            height=320,
            title="Email Classification Distribution"
        )

        st.altair_chart(
            chart,
            use_container_width=True
        )
    else:
        st.info(
            "Analyze an email to generate the classification chart."
        )


    st.markdown("### Your analysis history")
    history = get_history(limit=100)
    if history:
        history_df = pd.DataFrame([
            {
                "Date and time": record["analyzed_at"].replace("T", " "),
                "Subject": record["subject"],
                "Classification": (
                    "Spam / unsolicited" if record["prediction"] == 1
                    else "Classified as legitimate"
                ),
            }
            for record in history[:5]
        ])
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    else:
        st.info("No analyses saved yet. Open Email Analyzer to analyze a sample or your own email.")

    st.markdown("### Explore the machine-learning project")
    st.write(
        "Open **ML Analytics** in the sidebar to explore the CEAS-08 dataset, "
        "exploratory data analysis, feature-length patterns, model comparison, "
        "classification metrics, and confusion matrices from the ML workflow."
    )

# ---------------- ML ANALYTICS PAGE ----------------
elif page == "🧠  ML Analytics":
    st.markdown('<div class="section-title">Machine-learning analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="muted">Dataset exploration, feature analysis, and evaluation of the three algorithms used in the project notebook.</p>',
        unsafe_allow_html=True,
    )

    dataset = load_dataset()
    if dataset is None:
        st.error(
            "The dataset was not found. Expected: dataset/CEAS_08.csv/CEAS_08.csv "
            "inside the PhishGuard project folder."
        )
        st.stop()

    with st.spinner("Loading ML evaluation results if they have not been calculated yet..."):
        evaluation = evaluate_models()

    # Dataset overview
    st.markdown("### 1. Dataset overview")
    total_rows, total_cols = dataset.shape
    class_counts = dataset["label"].value_counts().sort_index()
    class0 = int(class_counts.get(0, 0))
    class1 = int(class_counts.get(1, 0))
    missing_total = int(dataset.isnull().sum().sum())
    duplicate_total = int(dataset.duplicated().sum())

    a, b, c, d = st.columns(4)
    a.metric("Total emails", f"{total_rows:,}")
    b.metric("Dataset columns", total_cols)
    c.metric("Missing cells", f"{missing_total:,}")
    d.metric("Duplicate rows", f"{duplicate_total:,}")

    st.caption(
        "Source: CEAS-08 email spam dataset included in this project. "
        "Class 0 and Class 1 are the dataset's labels; they should not be interpreted as a validated phishing verdict."
    )

    with st.expander("View dataset columns, data types, and missing values"):
        quality_df = pd.DataFrame({
            "Data type": dataset.dtypes.astype(str),
            "Missing values": dataset.isnull().sum(),
        })
        st.dataframe(quality_df, use_container_width=True)
        st.write("Preview of the dataset:")
        st.dataframe(dataset.head(10), use_container_width=True, hide_index=True)

    # Target distribution
    st.markdown("### 2. Target class distribution")
    class_df = pd.DataFrame({
        "Class": ["Class 0 — classified as legitimate", "Class 1 — spam / unsolicited"],
        "Email count": [class0, class1],
    })
    class_chart = alt.Chart(class_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X("Class:N", sort=None, title=None, axis=alt.Axis(labelAngle=0, labelLimit=260)),
        y=alt.Y("Email count:Q", title="Number of emails"),
        color=alt.Color("Class:N", legend=None),
        tooltip=["Class:N", alt.Tooltip("Email count:Q", format=",")],
    ).properties(height=330)
    st.altair_chart(class_chart, use_container_width=True)
    st.dataframe(
        pd.DataFrame({
            "Dataset label": [0, 1],
            "Interpretation used in this app": ["Classified as legitimate", "Spam / unsolicited"],
            "Email count": [class0, class1],
            "Share (%)": [
                round(class0 / total_rows * 100, 2),
                round(class1 / total_rows * 100, 2),
            ],
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Prepare feature lengths for EDA
    eda = dataset.copy()
    eda["subject"] = eda["subject"].fillna("").astype(str)
    eda["body"] = eda["body"].fillna("").astype(str)
    eda["subject_length"] = eda["subject"].str.len()
    eda["email_length"] = eda["body"].str.len()
    eda["Class"] = eda["label"].map({
        0: "Class 0 — classified as legitimate",
        1: "Class 1 — spam / unsolicited",
    }).fillna(eda["label"].astype(str))

    # Length analysis
    st.markdown("### 3. Email body length by class")
    body_stats = eda.groupby("Class")["email_length"].agg(
        ["count", "mean", "median", "min", "max"]
    ).round(2).reset_index()
    body_box = alt.Chart(eda).mark_boxplot(extent="min-max").encode(
        x=alt.X("Class:N", title=None, axis=alt.Axis(labelAngle=0, labelLimit=260)),
        y=alt.Y("email_length:Q", title="Body length (characters)", scale=alt.Scale(type="sqrt")),
        color=alt.Color("Class:N", legend=None),
        tooltip=["Class:N", alt.Tooltip("email_length:Q", title="Body length")],
    ).properties(height=350)
    st.altair_chart(body_box, use_container_width=True)
    st.dataframe(body_stats, use_container_width=True, hide_index=True)
    st.caption("The vertical axis uses a square-root scale so unusually long emails do not hide the typical range.")

    st.markdown("### 4. Email subject length by class")
    subject_stats = eda.groupby("Class")["subject_length"].agg(
        ["count", "mean", "median", "min", "max"]
    ).round(2).reset_index()
    subject_box = alt.Chart(eda).mark_boxplot(extent="min-max").encode(
        x=alt.X("Class:N", title=None, axis=alt.Axis(labelAngle=0, labelLimit=260)),
        y=alt.Y("subject_length:Q", title="Subject length (characters)"),
        color=alt.Color("Class:N", legend=None),
        tooltip=["Class:N", alt.Tooltip("subject_length:Q", title="Subject length")],
    ).properties(height=350)
    st.altair_chart(subject_box, use_container_width=True)
    st.dataframe(subject_stats, use_container_width=True, hide_index=True)

    # Bivariate analysis
    st.markdown("### 5. Subject length vs. email body length")
    st.caption("A sample of up to 5,000 rows is plotted to keep the chart responsive.")
    scatter_data = eda.sample(n=min(5000, len(eda)), random_state=42)
    scatter = alt.Chart(scatter_data).mark_circle(size=35, opacity=0.45).encode(
        x=alt.X("subject_length:Q", title="Subject length (characters)"),
        y=alt.Y("email_length:Q", title="Body length (characters)", scale=alt.Scale(type="sqrt")),
        color=alt.Color("Class:N", title="Dataset class"),
        tooltip=[
            "Class:N",
            alt.Tooltip("subject_length:Q", title="Subject length"),
            alt.Tooltip("email_length:Q", title="Body length"),
        ],
    ).properties(height=400)
    st.altair_chart(scatter, use_container_width=True)

    # Statistics and correlation
    st.markdown("### 6. Numeric statistical summary")
    st.dataframe(
        eda[["label", "email_length", "subject_length"]].describe().round(2),
        use_container_width=True,
    )
    st.markdown("### 7. Correlation analysis")
    corr = (
        eda[["label", "email_length", "subject_length"]]
        .corr()
        .rename_axis("Row feature")
        .reset_index()
        .melt(
            id_vars="Row feature",
            var_name="Column feature",
            value_name="Correlation",
        )
    )
    heatmap = alt.Chart(corr).mark_rect(cornerRadius=4).encode(
        x=alt.X("Column feature:N", title=None),
        y=alt.Y("Row feature:N", title=None),
        color=alt.Color(
            "Correlation:Q",
            scale=alt.Scale(scheme="redblue", domain=[-1, 1]),
            legend=alt.Legend(format=".2f"),
        ),
        tooltip=[
            "Row feature:N",
            "Column feature:N",
            alt.Tooltip("Correlation:Q", format=".3f"),
        ],
    ).properties(height=300)
    heatmap_text = alt.Chart(corr).mark_text(fontSize=14).encode(
        x="Column feature:N",
        y="Row feature:N",
        text=alt.Text("Correlation:Q", format=".2f"),
        color=alt.value("#f8fafc"),
    )
    st.altair_chart(heatmap + heatmap_text, use_container_width=True)
    st.caption("Correlation shows linear association among these numeric columns; it does not establish causation.")

    # Model comparison
    st.markdown("### 9. Model evaluation and performance metrics")
    st.write(
        "The app recreates the notebook's stratified 80:20 train/test split and TF-IDF settings "
        "(up to 20,000 features, unigrams and bigrams, English stop words) to compare the algorithms."
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Training emails", f'{evaluation["train_size"]:,}')
    m2.metric("Testing emails", f'{evaluation["test_size"]:,}')
    m3.metric("TF-IDF features", f'{evaluation["feature_count"]:,}')

    comparison = evaluation["comparison"]
    accuracy_chart = alt.Chart(comparison).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X("Algorithm:N", sort="-y", title=None, axis=alt.Axis(labelAngle=-15)),
        y=alt.Y("Accuracy (%):Q", title="Test accuracy (%)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("Algorithm:N", legend=None),
        tooltip=[
            "Algorithm:N",
            alt.Tooltip("Accuracy (%):Q", format=".2f"),
            alt.Tooltip("Precision (%):Q", format=".2f"),
            alt.Tooltip("Recall (%):Q", format=".2f"),
            alt.Tooltip("F1-score (%):Q", format=".2f"),
        ],
    ).properties(height=350)
    st.altair_chart(accuracy_chart, use_container_width=True)
    st.dataframe(comparison.round(2), use_container_width=True, hide_index=True)

    # ROC-AUC Analysis
    st.markdown("### ROC-AUC Analysis")

    roc_rows = []

    for algorithm, model_details in evaluation["details"].items():
        for fpr_value, tpr_value in zip(
            model_details["fpr"], model_details["tpr"]
        ):
            roc_rows.append({
                "False Positive Rate": float(fpr_value),
                "True Positive Rate": float(tpr_value),
                "Algorithm": algorithm,
                "ROC-AUC": float(model_details["roc_auc"]),
            })

    roc_df = pd.DataFrame(roc_rows)

    roc_lines = alt.Chart(roc_df).mark_line(
        strokeWidth=2
    ).encode(
        x=alt.X(
            "False Positive Rate:Q",
            title="False Positive Rate",
            scale=alt.Scale(domain=[0, 1])
        ),
        y=alt.Y(
            "True Positive Rate:Q",
            title="True Positive Rate",
            scale=alt.Scale(domain=[0, 1])
        ),
        color=alt.Color("Algorithm:N", title="Model"),
        tooltip=[
            "Algorithm:N",
            alt.Tooltip("False Positive Rate:Q", format=".3f"),
            alt.Tooltip("True Positive Rate:Q", format=".3f"),
            alt.Tooltip("ROC-AUC:Q", format=".4f"),
        ],
    ).properties(
        height=400,
        title="ROC Curves — Model Comparison"
    )

    baseline_df = pd.DataFrame({
        "False Positive Rate": [0, 1],
        "True Positive Rate": [0, 1]
    })

    baseline = alt.Chart(baseline_df).mark_line(
        strokeDash=[6, 4],
        color="#94a3b8"
    ).encode(
        x="False Positive Rate:Q",
        y="True Positive Rate:Q"
    )

    st.altair_chart(
        roc_lines + baseline,
        use_container_width=True
    )

    # ROC-AUC score comparison
    roc_comparison = comparison[
        ["Algorithm", "ROC-AUC Score"]
    ].sort_values(
        "ROC-AUC Score",
        ascending=False
    )

    st.markdown("#### ROC-AUC Score Comparison")

    st.dataframe(
        roc_comparison.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "ROC-AUC measures how well a model ranks positive examples "
        "above negative examples across different thresholds. "
        "Scores are calculated using the held-out test dataset. "
        "The CEAS-08 dataset contains spam labels and does not "
        "validate real-world phishing detection."
    )

    # Model-specific detailed metrics
    selected_model = st.selectbox(
        "Inspect detailed evaluation for",
        comparison["Algorithm"].tolist(),
        index=comparison["Algorithm"].tolist().index("Linear SVM (tuned)"),
    )
    selected_details = evaluation["details"][selected_model]
    report = selected_details["report"]
    cm = selected_details["confusion_matrix"]

    st.markdown(f"#### Classification report — {selected_model}")
    report_df = pd.DataFrame(report).transpose()
    report_df.index.name = "Class / metric"
    st.dataframe(report_df.round(3), use_container_width=True)

    st.markdown(f"#### Confusion matrix — {selected_model}")
    cm_df = pd.DataFrame(
        [
            {"Actual class": "Class 0", "Predicted class": "Class 0", "Emails": int(cm[0, 0])},
            {"Actual class": "Class 0", "Predicted class": "Class 1", "Emails": int(cm[0, 1])},
            {"Actual class": "Class 1", "Predicted class": "Class 0", "Emails": int(cm[1, 0])},
            {"Actual class": "Class 1", "Predicted class": "Class 1", "Emails": int(cm[1, 1])},
        ]
    )
    cm_chart = alt.Chart(cm_df).mark_rect().encode(
        x=alt.X("Predicted class:N", title="Predicted label"),
        y=alt.Y("Actual class:N", title="Actual label"),
        color=alt.Color("Emails:Q", scale=alt.Scale(scheme="tealblues")),
        tooltip=["Actual class:N", "Predicted class:N", alt.Tooltip("Emails:Q", format=",")],
    )
    cm_text = alt.Chart(cm_df).mark_text(fontSize=20, fontWeight="bold", color="white").encode(
        x="Predicted class:N", y="Actual class:N", text=alt.Text("Emails:Q", format=",")
    )
    st.altair_chart(cm_chart + cm_text, use_container_width=True)
    st.caption(
        "These are offline evaluation metrics from the dataset split, not live app-user statistics. "
        "Very high scores should be interpreted cautiously and may depend on dataset properties."
    )

    # Model building and basic hyperparameter tuning
    st.divider()
    st.markdown("### 10. Model building and hyperparameter tuning")
    st.write(
        "Three supervised classifiers are trained on the same stratified 80:20 split: "
        "Logistic Regression, Multinomial Naive Bayes, and Linear SVM. "
        "TF-IDF converts email text into numeric features."
    )
    build_a, build_b, build_c = st.columns(3)
    build_a.metric("Training set", f'{evaluation["train_size"]:,} emails')
    build_b.metric("Test set", f'{evaluation["test_size"]:,} emails')
    build_c.metric("TF-IDF features", f'{evaluation["feature_count"]:,}')

    st.markdown("#### Linear SVM hyperparameter tuning")
    st.write(
        "The C parameter controls the regularization strength. We compare C = 0.25, 1.0, and 4.0 "
        "using 3-fold cross-validation on a reproducible sample of the training set. "
        "The held-out test set is not used to choose C."
    )
    tuning_results = evaluation["tuning_results"]
    tuning_chart = alt.Chart(tuning_results).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X("C parameter:Q", scale=alt.Scale(type="log"), title="Linear SVM C parameter"),
        y=alt.Y("Mean CV weighted F1:Q", title="Mean cross-validation weighted F1 (%)"),
        tooltip=[
            alt.Tooltip("C parameter:Q", format=".2f"),
            alt.Tooltip("Mean CV weighted F1:Q", format=".3f"),
            alt.Tooltip("CV F1 standard deviation:Q", format=".3f"),
        ],
    ).properties(height=300)
    st.altair_chart(tuning_chart, use_container_width=True)
    st.dataframe(tuning_results.round(3), use_container_width=True, hide_index=True)

    best_row = comparison.iloc[0]
    best_col, best_info = st.columns([1, 2])
    best_col.metric("Best model by test accuracy", best_row["Algorithm"])
    best_col.metric("Test accuracy", f'{best_row["Accuracy (%)"]:.2f}%')
    best_info.markdown(
        f"**Selected Linear SVM C:** `{evaluation['best_c']}`  \n"
        f"**Best CV weighted F1 during tuning:** `{evaluation['best_cv_f1']:.2f}%`  \n"
        f"**Tuning sample:** `{evaluation['tuning_sample_count']:,}` training emails"
    )
    st.caption(
        "The best model label is based on the held-out test accuracy in this educational comparison. "
        "For a production decision, use a separate validation set and evaluate on an independent dataset."
    )

    # Advanced technique: unsupervised K-Means pattern discovery
    st.divider()
    st.markdown("### 11. Advanced techniques and pattern discovery — K-Means")
    st.write(
        "K-Means groups email messages by similarity in their TF-IDF text features. "
        "The clustering algorithm does not use the dataset labels while creating clusters. "
        "For performance, this visualization uses a reproducible sample of up to 2,500 emails."
    )
    cluster_results = run_kmeans_clustering()
    cluster_metrics = st.columns(3)
    cluster_metrics[0].metric("Emails clustered", f'{cluster_results["sample_count"]:,}')
    cluster_metrics[1].metric("Text features", f'{cluster_results["feature_count"]:,}')
    cluster_metrics[2].metric("2D variance shown", f'{cluster_results["explained_variance"]:.1f}%')

    cluster_chart = alt.Chart(cluster_results["plot_df"]).mark_circle(size=45, opacity=0.65).encode(
        x=alt.X("Component 1:Q", title="SVD component 1"),
        y=alt.Y("Component 2:Q", title="SVD component 2"),
        color=alt.Color("Cluster:N", title="K-Means cluster"),
        tooltip=[
            "Cluster:N",
            "Dataset label:N",
            "Subject:N",
            alt.Tooltip("Component 1:Q", format=".3f"),
            alt.Tooltip("Component 2:Q", format=".3f"),
        ],
    ).properties(height=470)
    st.altair_chart(cluster_chart, use_container_width=True)
    st.caption(
        "The two plotted axes are a 2D Truncated SVD projection of the TF-IDF vectors. "
        "Points that appear close in this projection may be more similar in text space, but the projection loses information."
    )

    st.markdown("#### Most representative terms in each cluster")
    term_cols = st.columns(2)
    for index, (cluster_name, words) in enumerate(cluster_results["top_terms"].items()):
        with term_cols[index]:
            st.markdown(f"**{cluster_name}**")
            st.write(", ".join(words))

    st.markdown("#### Compare discovered clusters with original dataset labels")
    st.dataframe(cluster_results["crosstab"], use_container_width=True, hide_index=True)
    st.markdown(
        "**How to interpret this:** use the representative terms to describe the themes of each cluster, "
        "then inspect the label counts to see whether a cluster contains mostly one dataset class or a mixture. "
        "K-Means clusters are not automatically 'spam' and 'legitimate'; their meaning must be interpreted from the terms and label distribution."
    )
    st.caption(
        "K-Means is an unsupervised exploratory technique, not the classifier used by the Email Analyzer. "
        "The source dataset contains spam labels, so results should not be presented as validated phishing detection."
    )


# ---------------- EMAIL ANALYZER PAGE ----------------
elif page == "🛡️  Email Analyzer":

    
    samples = {
        "Online shopping — delivery update": {
            "subject": "Update on Your Order #IN-4827316: Delivery Scheduled for Tomorrow",
            "body": """Dear Customer,

Thank you for shopping with us. We are writing to confirm that your recent order #IN-4827316 has been processed and is currently on its way to your delivery address.

Your package is expected to arrive tomorrow between 10:00 AM and 6:00 PM. Please ensure that someone is available to receive the delivery. You can review your order details and check the latest shipping status by signing in to your account through our official website.

If you have already received your package, please disregard this notification. For any questions regarding your order, contact our customer support team using the contact details listed on our official website.

Thank you for choosing us.

Kind regards,
Customer Support Team""",
        },

        "Suspicious — prize notification": {
            "subject": "Congratulations! Your Email Address Has Been Selected for a Special Reward",
            "body": """Dear Winner,

We are pleased to inform you that your email address has been selected as a recipient of our exclusive promotional reward. You have been chosen from a large number of participants to receive a complimentary gift as part of our limited-time customer appreciation campaign.

To complete your reward registration, please click the link provided in this message and submit the requested information. Your reward must be claimed within the next 24 hours. Failure to complete the verification process before the deadline may result in cancellation of your claim.

Please note that this offer is available for a limited time only. We encourage you to act immediately to avoid missing this opportunity.

If you have questions, reply to this email for further assistance.

Congratulations once again,
Promotional Rewards Department

Note: This is a fictional sample for testing email classification.""",
        },

        "Suspicious — account verification": {
            "subject": "Urgent Action Required: Your Account May Be Temporarily Suspended",
            "body": """Dear Valued Customer,

We recently detected an issue with the verification status of your account. According to our records, some account information requires confirmation before you can continue using all available services.

To prevent a temporary interruption, please review your account information as soon as possible. Use the verification link included in this example message to complete the requested process.

If verification is not completed within 24 hours, access to certain account features may be restricted until the review is finished.

We understand that account security is important to you. If you believe you have received this message in error, please contact the relevant service provider using its independently verified official website or customer support number.

Thank you for your cooperation.

Account Security Team

Note: This fictional sample demonstrates urgency-based wording and is not a real account notice.""",
        },

        "Workplace — meeting follow-up": {
            "subject": "Follow-Up: Project Review Meeting and Action Items for This Week",
            "body": """Hi Team,

Thank you for joining today's project review meeting. We discussed the current development progress, the remaining tasks for this sprint, and a few issues that need to be resolved before the next internal demonstration.

As agreed, the development team will complete the pending interface updates by Wednesday afternoon. The testing team will review the latest build and document any issues that need attention. Please add your progress updates to the shared project tracker before the end of the day so that everyone has a clear picture of the remaining work.

Our next review meeting is scheduled for Friday at 11:00 AM. If you have any questions or anticipate a delay with your assigned tasks, please let me know in advance.

Thanks again for your cooperation.

Best regards,
Project Coordination Team""",
        },

        "Banking — transaction notification": {
            "subject": "Transaction Alert: A Recent Transaction Has Been Recorded",
            "body": """Dear Customer,

This is an automated notification regarding recent activity associated with your account. A transaction has been recorded in accordance with the details submitted through your banking service.

If you recognize this activity, no further action is required. You may review your transaction history by opening your bank's official mobile application or by entering its verified website address directly into your browser.

If you do not recognize the transaction, contact your bank immediately using the customer support number printed on your card or published on its official website. Never share your PIN, password, or one-time password with anyone claiming to represent your bank.

This notification is provided for your information.

Regards,
Customer Notifications Team""",
        },

        "Suspicious — unexpected prize and urgent action": {
            "subject": "Final Reminder: Claim Your Exclusive Gift Before Your Registration Expires",
            "body": """Dear Customer,

We are contacting you regarding an exclusive reward that has reportedly been reserved for your email address. Our promotional records indicate that your registration has not yet been completed.

To proceed, you are requested to confirm your details using the claim process described in this message. Please complete the process before the stated deadline, as unclaimed rewards may be removed from the promotional list.

This opportunity is being presented as time-sensitive, and immediate action is encouraged. If you did not enter a competition or expect a reward from the sender, independently verify the message before responding or following any links.

Do not provide banking credentials, passwords, or verification codes to claim an unexpected prize.

Promotional Notifications Team

Note: This is a fictional sample designed for testing. It does not represent a real promotion.""",
        },
    }

    if "reset_email_form" not in st.session_state:
       st.session_state.reset_email_form = False

    # Initialize form state
    if "email_subject" not in st.session_state:
        st.session_state.email_subject = ""

    if "email_body" not in st.session_state:
        st.session_state.email_body = ""

    if "previous_sample" not in st.session_state:
        st.session_state.previous_sample = "Select a sample..."

    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None

    st.markdown(
        '<div class="section-title">Email Analyzer</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="muted">Analyze an email using PhishGuard’s trained text-classification model.</p>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.7, 1], gap="large")

    with left:
        st.markdown("### Enter email details")

        sample_options = [
    "Select a sample...",
    *samples.keys(),
]

        sample_choice = st.selectbox(
            "Try a sample email",
            sample_options,
            key="sample_choice",
        )

        if sample_choice != st.session_state.previous_sample:
            if sample_choice in samples:
                st.session_state.email_subject = samples[sample_choice]["subject"]
                st.session_state.email_body = samples[sample_choice]["body"]

            st.session_state.previous_sample = sample_choice
            st.session_state.analysis_result = None

        subject = st.text_input(
            "Email subject",
            placeholder="e.g. Your account needs verification",
            key="email_subject",
            max_chars=500,
        )

        body = st.text_area(
            "Email body",
            height=230,
            placeholder="Paste the email message here...",
            key="email_body",
            max_chars=20000,
        )

        
        button_col1, button_col2 = st.columns(2)

        with button_col1:
            analyze = st.button(
                "🛡️ Analyze email",
                type="primary",
                use_container_width=True,
            )

        
        def clear_email_form():
            st.session_state.email_subject = ""
            st.session_state.email_body = ""
            st.session_state.previous_sample = "Select a sample..."
            st.session_state.analysis_result = None
            st.session_state.sample_choice = "Select a sample..."

        with button_col2:
            st.button(
                "✕ Clear form",
                use_container_width=True,
                on_click=clear_email_form,
                key="clear_email_button",
            )


       
    with right:
        st.markdown("### About PhishGuard")

        st.markdown(
            """
            <div class="result-card">
                <h3>🧠 Machine-learning analysis</h3>
                <p class="muted">
                    Uses TF-IDF text features and a trained Linear Support Vector Machine (SVM).
                </p>
            </div>
            <div class="result-card">
                <h3>🔎 What it analyzes</h3>
                <p class="muted">
                    The words and phrases in the subject and body of the supplied email.
                </p>
            </div>
            <div class="result-card">
                <h3>⚠️ Important limitation</h3>
                <p class="muted">
                    The model was trained on an email spam dataset. It is not a validated
                    phishing detector. Neither result guarantees that a message is safe
                    or malicious.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.info(
            "For privacy, avoid pasting passwords, verification codes, "
            "financial details, or other sensitive information."
        )

    # ---------------- RUN ANALYSIS ----------------
    if analyze:
        clean_subject = subject.strip()
        clean_body = body.strip()

        if not clean_subject and not clean_body:
            st.warning("Please enter an email subject or body.")
            st.session_state.analysis_result = None

        else:
            email_text = f"{clean_subject} {clean_body}".strip()

            try:
                features = vectorizer.transform([email_text])
                prediction = int(model.predict(features)[0])
                decision_score = float(model.decision_function(features)[0])

                # A bounded relative indicator, not a probability.
                absolute_score = abs(decision_score)
                confidence = (
                    100.0 * absolute_score / (1.0 + absolute_score)
                )

                # Simple phrase indicators for user awareness only.
                # They do not determine the machine-learning prediction.
                indicator_phrases = [
                    "click here",
                    "verify your account",
                    "account suspension",
                    "claim your prize",
                    "act now",
                    "limited time",
                    "urgent",
                    "password",
                    "bank account",
                    "gift card",
                    "one-time password",
                    "otp",
                    "http://",
                    "https://",
                ]

                lower_text = email_text.lower()
                detected_indicators = [
                    phrase
                    for phrase in indicator_phrases
                    if phrase in lower_text
                ]

                save_analysis(
                    subject=clean_subject or "(No subject)",
                    prediction=prediction,
                    decision_strength=confidence,
                )

                st.session_state.analysis_result = {
                    "subject": clean_subject or "(No subject)",
                    "body": clean_body,
                    "text": email_text,
                    "prediction": prediction,
                    "decision_score": decision_score,
                    "strength": confidence,
                    "indicators": detected_indicators,
                }

            except Exception as error:
                st.error(
                    "PhishGuard could not analyze this email. "
                    "Check that your model and vectorizer are loaded correctly."
                )
                st.caption(f"Technical details: {error}")

    # ---------------- DISPLAY RESULTS ----------------
    result = st.session_state.analysis_result

    if result is not None:
        st.divider()
        st.markdown("## Analysis results")

        result_col, metric_col = st.columns([1.5, 1])

        with result_col:
            if result["prediction"] == 1:
                st.markdown(
                    """
                    <div class="result-card spam-result">
                        <h3>⚠️ Classified as spam / unsolicited</h3>
                        <p>
                            The model assigned this email to Class 1.
                            Review unexpected requests, links, and offers carefully.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="result-card safe-result">
                        <h3>✓ Classified as likely legitimate</h3>
                        <p>
                            The model assigned this email to Class 0.
                            This result does not establish that the email is safe.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with metric_col:
            st.markdown("### Model decision strength")
            st.progress(
                max(0, min(100, int(result["strength"])))
            )
            st.metric(
                "Relative strength",
                f'{result["strength"]:.1f}%',
            )
            st.caption(
                "This is a transformed SVM decision score, not a calibrated "
                "probability or a measure of real-world safety."
            )

        st.caption(
            f'Model prediction: Class {result["prediction"]}'
        )

        st.markdown("### Text indicators")

        if result["indicators"]:
            st.warning(
                "The following predefined phrases or URL patterns were found. "
                "Their presence alone does not prove that an email is malicious."
            )

            st.write(" · ".join(result["indicators"]))
        else:
            st.success(
                "No phrases from the basic indicator list were found. "
                "This does not mean that the email is safe."
            )

        with st.expander("View analyzed email"):
            st.markdown("**Subject**")
            st.write(result["subject"])
            st.markdown("**Body**")
            st.write(result["body"])


# ---------------- EMAIL HISTORY PAGE ----------------
elif page == "🕘  Analysis History":
    st.markdown('<div class="section-title">Email analysis history</div>', unsafe_allow_html=True)
    st.markdown('<p class="muted">Your most recent saved email analyses.</p>', unsafe_allow_html=True)
    history = get_history(limit=100)

    if history:
        history_df = pd.DataFrame([
            {
                "Date and time": record["analyzed_at"].replace("T", " "),
                "Email subject": record["subject"],
                "Classification": (
                    "Spam / unsolicited" if record["prediction"] == 1
                    else "Classified as legitimate"
                ),
                "Relative decision strength": f'{record["decision_strength"]:.1f}%',
            }
            for record in history
        ])
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        if st.button("Clear analysis history"):
            clear_history()
            st.success("Analysis history cleared.")
            st.rerun()
    else:
        st.info("No saved analyses yet. Analyze an email to see it here.")

# ---------------- FOOTER ----------------
st.markdown(
    """
    <div class="footer">
        PHISHGUARD · MACHINE-LEARNING EMAIL CLASSIFICATION<br><br>
        Educational project · Predictions are not security guarantees
    </div>
    """,
    unsafe_allow_html=True,
)
