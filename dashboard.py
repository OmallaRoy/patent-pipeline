import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from sklearn.linear_model import LinearRegression
import json
import os

st.set_page_config(page_title="Patent Intelligence Dashboard", layout="wide")

REPORTS_DIR = "reports/"
VISUALS_DIR = "visuals/"

st.title("Global Patent Intelligence Dashboard")
st.markdown("Data source: PatentsView - USPTO Granted Patents")

with open(REPORTS_DIR + "report.json") as f:
    report = json.load(f)

summary = report["summary"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Patents",   f"{summary['total_patents']:,}")
col2.metric("Total Inventors", f"{summary['total_inventors']:,}")
col3.metric("Total Companies", f"{summary['total_companies']:,}")
col4.metric("Total Countries", f"{summary['total_countries']:,}")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Patent Analytics", "ML Productivity Predictor", "Data Tables"])

with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Top 10 Inventors")
        st.image(VISUALS_DIR + "top_inventors.png")
    with col_b:
        st.subheader("Top 10 Companies")
        st.image(VISUALS_DIR + "top_companies.png")

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Top 10 Countries")
        st.image(VISUALS_DIR + "top_countries.png")
    with col_d:
        st.subheader("Country Patent Share")
        st.image(VISUALS_DIR + "country_share.png")

    st.subheader("Patents Per Year")
    st.image(VISUALS_DIR + "patents_per_year.png")

with tab2:
    st.subheader("Inventor Productivity Predictor")
    st.markdown("Enter a patent count to predict an inventor's productivity level using Linear Regression.")

    st.markdown("---")

    col_ml1, col_ml2 = st.columns(2)

    with col_ml1:
        st.subheader("Make a Prediction")

        patent_input = st.slider(
            "Select number of patents filed by inventor:",
            min_value=1,
            max_value=50,
            value=5,
            step=1
        )

        df_ml = pd.read_csv(REPORTS_DIR + "ml_productivity.csv")

        X = df_ml[['patent_count']]
        y = df_ml['productivity_label'].map({'Low': 0, 'Medium': 1, 'High': 2})

        model = LinearRegression()
        model.fit(X, y)

        pred = model.predict(pd.DataFrame({'patent_count': [patent_input]}))[0]
        pred_rounded = int(np.round(pred).clip(0, 2))
        label = {0: 'Low', 1: 'Medium', 2: 'High'}[pred_rounded]
        color = {'Low': 'red', 'Medium': 'orange', 'High': 'green'}[label]

        st.markdown("---")
        st.markdown(f"### Prediction Result")
        st.markdown(f"An inventor with **{patent_input} patents** is predicted to be:")
        st.markdown(f"<h1 style='color:{color}'>{label} Productivity</h1>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Productivity Scale:**")
        st.markdown("- Low → 1 to 2 patents")
        st.markdown("- Medium → 3 to 9 patents")
        st.markdown("- High → 10 or more patents")

    with col_ml2:
        st.subheader("Model Performance")
        st.image(VISUALS_DIR + "ml_productivity.png")

        st.markdown("**Model Details:**")
        st.markdown("- Algorithm: Linear Regression")
        st.markdown("- Feature: Patent count per inventor")
        st.markdown("- Target: Productivity category")
        st.markdown("- Training samples: 73,953")
        st.markdown("- Testing samples: 18,489")
        st.markdown("- R2 Score: 0.5237")
        st.markdown("- Mean Absolute Error: 0.0202")

    st.markdown("---")
    st.subheader("Full Inventor Productivity Classifications")
    st.dataframe(df_ml.head(50), use_container_width=True)

with tab3:
    st.subheader("Top 10 Inventors Table")
    df_inv = pd.DataFrame(report["top_inventors"])
    st.dataframe(df_inv, use_container_width=True)

    st.subheader("Top 10 Companies Table")
    df_comp = pd.DataFrame(report["top_companies"])
    st.dataframe(df_comp, use_container_width=True)

    st.subheader("Top 10 Countries Table")
    df_countries = pd.DataFrame(report["top_countries"])
    st.dataframe(df_countries, use_container_width=True)