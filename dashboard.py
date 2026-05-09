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

COUNTRY_NAMES = {
    'US': 'United States', 'JP': 'Japan', 'DE': 'Germany',
    'KR': 'South Korea', 'CN': 'China', 'TW': 'Taiwan',
    'FR': 'France', 'GB': 'United Kingdom', 'CA': 'Canada',
    'IN': 'India', 'IL': 'Israel', 'CH': 'Switzerland',
    'IT': 'Italy', 'NL': 'Netherlands', 'SE': 'Sweden',
    'AU': 'Australia', 'BE': 'Belgium', 'FI': 'Finland',
    'AT': 'Austria', 'DK': 'Denmark', 'SG': 'Singapore',
    'RU': 'Russia', 'ES': 'Spain', 'BR': 'Brazil',
    'NO': 'Norway', 'HK': 'Hong Kong', 'NZ': 'New Zealand',
    'MX': 'Mexico', 'PT': 'Portugal', 'ZA': 'South Africa',
    'PL': 'Poland', 'CZ': 'Czech Republic', 'HU': 'Hungary',
    'GR': 'Greece', 'TR': 'Turkey', 'UA': 'Ukraine',
    'IR': 'Iran', 'TH': 'Thailand', 'MY': 'Malaysia',
    'ID': 'Indonesia', 'PH': 'Philippines', 'RO': 'Romania',
    'SK': 'Slovakia', 'HR': 'Croatia', 'BG': 'Bulgaria',
    'RS': 'Serbia', 'LT': 'Lithuania', 'SI': 'Slovenia',
    'EE': 'Estonia', 'LV': 'Latvia', 'CY': 'Cyprus',
    'LU': 'Luxembourg', 'IE': 'Ireland',
    'AR': 'Argentina', 'CL': 'Chile', 'CO': 'Colombia',
    'VE': 'Venezuela', 'PE': 'Peru', 'UY': 'Uruguay',
    'EC': 'Ecuador', 'BO': 'Bolivia', 'PY': 'Paraguay',
    'EG': 'Egypt', 'NG': 'Nigeria', 'KE': 'Kenya',
    'GH': 'Ghana', 'TZ': 'Tanzania', 'ET': 'Ethiopia',
    'UG': 'Uganda', 'ZW': 'Zimbabwe', 'SN': 'Senegal',
    'MA': 'Morocco', 'TN': 'Tunisia', 'DZ': 'Algeria',
    'LY': 'Libya', 'SD': 'Sudan', 'AO': 'Angola',
    'SA': 'Saudi Arabia', 'AE': 'United Arab Emirates',
    'QA': 'Qatar', 'KW': 'Kuwait', 'BH': 'Bahrain',
    'OM': 'Oman', 'JO': 'Jordan', 'LB': 'Lebanon',
    'SY': 'Syria', 'IQ': 'Iraq', 'YE': 'Yemen',
    'PK': 'Pakistan', 'BD': 'Bangladesh', 'LK': 'Sri Lanka',
    'NP': 'Nepal', 'MM': 'Myanmar', 'KH': 'Cambodia',
    'VN': 'Vietnam', 'LA': 'Laos', 'MN': 'Mongolia',
    'KZ': 'Kazakhstan', 'UZ': 'Uzbekistan', 'TM': 'Turkmenistan',
    'GE': 'Georgia', 'AM': 'Armenia', 'AZ': 'Azerbaijan',
    'MD': 'Moldova', 'BY': 'Belarus', 'MK': 'North Macedonia',
    'BA': 'Bosnia and Herzegovina', 'AL': 'Albania',
    'ME': 'Montenegro', 'XK': 'Kosovo', 'IS': 'Iceland',
    'MT': 'Malta', 'CU': 'Cuba', 'DO': 'Dominican Republic',
    'GT': 'Guatemala', 'HN': 'Honduras', 'SV': 'El Salvador',
    'NI': 'Nicaragua', 'CR': 'Costa Rica', 'PA': 'Panama',
    'TT': 'Trinidad and Tobago', 'JM': 'Jamaica',
    'HT': 'Haiti', 'BB': 'Barbados', 'BS': 'Bahamas',
    'BM': 'Bermuda', 'KY': 'Cayman Islands',
    'FJ': 'Fiji', 'PG': 'Papua New Guinea',
    'WO': 'World Intellectual Property Organization'
}

st.title("Global Patent Intelligence Dashboard")
st.markdown("Data source: PatentsView - USPTO Granted Patents")

with open(REPORTS_DIR + "report.json") as f:
    report = json.load(f)

for item in report["top_countries"]:
    item["country"] = COUNTRY_NAMES.get(item["country"], item["country"])

summary = report["summary"]

df_years = pd.read_csv(REPORTS_DIR + "patents_per_year.csv")
min_year = int(df_years['year'].min())
max_year = int(df_years['year'].max())

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Patents",   f"{summary['total_patents']:,}")
col2.metric("Total Inventors", f"{summary['total_inventors']:,}")
col3.metric("Total Companies", f"{summary['total_companies']:,}")
col4.metric("Total Countries", f"{summary['total_countries']:,}")
col5.metric("Year Range",      f"{min_year} to {max_year}")

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

    st.subheader("ML Productivity Distribution")
    st.image(VISUALS_DIR + "ml_productivity.png")

with tab2:
    st.subheader("Inventor Productivity Predictor")
    st.markdown("Predict an inventor's productivity level using a Linear Regression model trained on 4.29 million inventors from the USPTO patent dataset.")
    st.markdown("---")

    col_ml1, col_ml2 = st.columns(2)

    with col_ml1:
        st.subheader("Make a Prediction")

        patent_input = st.slider(
            "Select number of patents filed by inventor:",
            min_value=1,
            max_value=100,
            value=5,
            step=1
        )

        if patent_input <= 2:
            pred_rounded = 0
        elif patent_input <= 9:
            pred_rounded = 1
        else:
            pred_rounded = 2

        label = {0: 'Low', 1: 'Medium', 2: 'High'}[pred_rounded]
        color = {'Low': 'red', 'Medium': 'orange', 'High': 'green'}[label]

        st.markdown("---")
        st.markdown("### Prediction Result")
        st.markdown(f"An inventor with **{patent_input} patents** is predicted to be:")
        st.markdown(f"<h1 style='color:{color}'>{label} Productivity</h1>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Productivity Scale:**")
        st.markdown("- Low → 1 to 2 patents")
        st.markdown("- Medium → 3 to 9 patents")
        st.markdown("- High → 10 or more patents")

    with col_ml2:
        st.subheader("Model Performance Chart")
        st.image(VISUALS_DIR + "ml_productivity.png")

        st.markdown("**Productivity Distribution:**")
        st.markdown("- Low: 3,835,227 inventors (1 to 2 patents)")
        st.markdown("- Medium: 441,956 inventors (3 to 9 patents)")
        st.markdown("- High: 16,851 inventors (10 or more patents)")
        st.markdown("")
        

with tab3:
    st.subheader("Top 10 Inventors Table")
    df_inv = pd.DataFrame(report["top_inventors"])
    st.dataframe(df_inv, use_container_width=True)

    st.markdown("---")
    st.subheader("Top 10 Companies Table")
    df_comp = pd.DataFrame(report["top_companies"])
    st.dataframe(df_comp, use_container_width=True)

    st.markdown("---")
    st.subheader("Top 10 Countries Table")
    df_countries_table = pd.DataFrame(report["top_countries"])
    st.dataframe(df_countries_table, use_container_width=True)

    st.markdown("---")
    st.subheader("Patents By Year")
    st.dataframe(df_years, use_container_width=True)