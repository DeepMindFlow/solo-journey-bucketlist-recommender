# streamlit/app.py
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
import plotly.express as px
from pathlib import Path

# Project-level imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data_loader import load_data_from_postgresql
from src.feature_engineering import prepare_numpy_data

# Optional: Set page config
st.set_page_config(
    page_title="🌟 DreamSeeker Solo-Activity Bucket List Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
alt.theme.enable("dark")



# Create DataFrame for easier filtering
raw_columns = ['user_id', 'activity_id', 'activity_name', 'category',
               'activity_type', 'user_mood', 'user_interest_score']


@st.cache_data(show_spinner=True)
def load_and_process():
    # Load raw data from PostgreSQL
    X_raw, y_raw = load_data_from_postgresql()
    X_raw = np.array(X_raw)
    y_raw = np.array(y_raw).astype(int)

    # Feature engineering step
    X_cleaned, y_cleaned, feature_names = prepare_numpy_data(X_raw, y_raw)

    # Create DataFrame for UI filtering and visualization
    df = pd.DataFrame(X_raw, columns=raw_columns)
    df['user_interest_score'] = df['user_interest_score'].astype(float)
    df['label'] = y_raw

    return df, X_cleaned, y_cleaned, feature_names

# Load & prepare data (cached)
df, X_cleaned, y_cleaned, feature_names = load_and_process()

# Sidebar Filters
with st.sidebar:
    st.title("🌟 DreamSeeker Filters")

    # Category filter
    categories = df['category'].unique().tolist()
    selected_category = st.selectbox("Select Category", ["All"] + categories)

    # Mood filter
    moods = df['user_mood'].unique().tolist()
    selected_mood = st.selectbox("Select User Mood", ["All"] + moods)

    # Score range filter
    score_range = st.slider("User Interest Score Range", 0.0, 1.0, (0.0, 1.0))

# Filtered DataFrame
filtered_df = df.copy()
if selected_category != "All":
    filtered_df = filtered_df[filtered_df['category'] == selected_category]
if selected_mood != "All":
    filtered_df = filtered_df[filtered_df['user_mood'] == selected_mood]
filtered_df = filtered_df[(filtered_df['user_interest_score'] >= score_range[0]) &
                          (filtered_df['user_interest_score'] <= score_range[1])]

# Show warning if no data after filtering
if filtered_df.empty:
    st.warning("⚠️ No data matches the selected filters. Please adjust your filters.")
else:
    # Layout: KPIs & Donut Charts
    col1, col2, col3 = st.columns((1.5, 1.5, 2))
    with col1:
        st.metric("Total Activities", len(filtered_df))
    with col2:
        st.metric("Avg Interest Score", f"{filtered_df['user_interest_score'].mean():.2f}")
    with col3:
        st.metric("Positive Labels", int(filtered_df['label'].sum()))

    # Donut Chart: Category Breakdown
    st.subheader("🎯 Category Breakdown")
    category_counts = filtered_df['category'].value_counts().reset_index()
    category_counts.columns = ['category', 'count']
    fig_cat = px.pie(
        category_counts,
        names='category',
        values='count',
        hole=0.5,
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    st.plotly_chart(fig_cat, use_container_width=True)

    # Label distribution bar
    st.subheader("🧮 Label Distribution")
    st.bar_chart(filtered_df['label'].value_counts())

    # Interest Score Histogram
    st.subheader("📈 Interest Score Histogram")
    fig, ax = plt.subplots()
    ax.hist(filtered_df['user_interest_score'], bins=10, color='skyblue', edgecolor='black')
    ax.set_xlabel("Interest Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Interest Scores")
    st.pyplot(fig)


    # Match Number of preview rows
    num_preview_rows = 20
    # Raw Data Preview
    st.subheader("📋 Preview Filtered Data")
    st.dataframe(filtered_df.head(num_preview_rows))

    # Cleaned Feature Preview
    st.subheader("🧼 Cleaned Feature Preview (NumPy Only)")
    # Combine cleaned features with label
    preview_array = np.hstack([X_cleaned[:num_preview_rows], y_cleaned[:num_preview_rows].reshape(-1, 1)])
    # Create column names
    preview_headers = feature_names + ['label']
    # Ensure shape matches
    if preview_array.shape[1] == len(preview_headers):
        preview_cleaned_df = pd.DataFrame(preview_array, columns=preview_headers)
        st.dataframe(preview_cleaned_df.head(num_preview_rows))
    else:
        st.error(
            f"Shape mismatch: preview_array has {preview_array.shape[1]} columns, but headers list has {len(preview_headers)}.")




    # Evaluation Curves
    st.subheader("📉 Model Evaluation Curves")
    colA, colB, colC = st.columns(3)

    # Get image directory relative to this script
    image_dir = Path(__file__).parents[1] / "images"

    with colA:
        roc_path = image_dir / "ROC.png"
        if roc_path.exists():
            st.image(str(roc_path), caption="ROC Curve", use_container_width=True)
        else:
            st.warning("ROC image not found.")

    with colB:
        pr_path = image_dir / "precision-recall_curve.png"
        if pr_path.exists():
            st.image(str(pr_path), caption="Precision-Recall Curve", use_container_width=True)
        else:
            st.warning("Precision-Recall image not found.")

    with colC:
        thresh_path = image_dir / "Threshold_vs_Metrics.png"
        if thresh_path.exists():
            st.image(str(thresh_path), caption="Threshold vs Metrics", use_container_width=True)
        else:
            st.warning("Threshold vs Metrics image not found.")
