
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Top 10 Inventors")
    st.image(VISUALS_DIR + "top_inventors.png")

with col_b:
    st.subheader("Top 10 Companies")
    st.image(VISUALS_DIR + "top_companies.png")

st.markdown("---")

col_c, col_d = st.columns(2)

with col_c:
    st.subheader("Top 10 Countries")
    st.image(VISUALS_DIR + "top_countries.png")

with col_d:
    st.subheader("Country Share")
    st.image(VISUALS_DIR + "country_share.png")

st.markdown("---")
st.subheader("Patents Per Year")
st.image(VISUALS_DIR + "patents_per_year.png")

st.markdown("---")
st.subheader("Top 10 Inventors Table")
df_inv = pd.DataFrame(report["top_inventors"])
st.dataframe(df_inv, use_container_width=True)

st.subheader("Top 10 Companies Table")
df_comp = pd.DataFrame(report["top_companies"])
st.dataframe(df_comp, use_container_width=True)

st.subheader("Top 10 Countries Table")
df_countries = pd.DataFrame(report["top_countries"])
st.dataframe(df_countries, use_container_width=True)
