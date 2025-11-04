# newest.py
# Final integrated Streamlit dashboard — NO MORE WHITE BARS ABOVE CHARTS
# - EDA sections wrapped in soft cards directly
# - Cluster columns fixed
# - Layout unchanged
# - Fully responsive

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
from math import pi
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import base64

# -------------------------
# CONFIG & THEME
# -------------------------
CONFIG = {
    "features": [
        'Age', 'Income', 'Has_Children', 'TotalMnt', 'TotalPurchases',
        'NumWebVisitsMonth', 'TotalAcceptedCmp', 'Days_Customer'
    ],
    "cluster_descriptions": {
        0: "Responsive High Spenders: Higher income, actively engages with promotions, younger, spends more per transaction",
        1: "Budget-Conscious Parent: Family-oriented, frequent low-spend shopper, loyal, older, browses often but ignores flashy campaigns"
    },
    "cluster_marketing": {
        0: (
            "• Personalized dynamic content: “For you”, “Recommended based on your style”\n"
            "• Offer flash sales, early access, VIP tiers — they respond to exclusivity\n"
            "• Use retargeting + social proof: reviews, bestsellers, influencer tags\n"
            "• Upsell/cross-sell at checkout — they’re willing to spend more"
        ),
        1: (
            "• Avoid blasting campaigns — use contextual nudges: “Based on your recent views...”, “Top picks for families like yours”\n"
            "• Improve product filtering & categorization — help them find what they need fast\n"
            "• Integrate loyalty program: show points earned on every visit/purchase\n"
            "• Trigger helpful emails: abandoned cart, restock alerts, “you viewed this last week”\n"
            "• Avoid flashy banners or heavy discounts — focus on utility, trust, and convenience"
        )
    },
    "colors": {
        "background": "#FFF3E0",
        "text_primary": "#333333",
        "accent": "#FF9800",
        "accent_hover": "#E65100",
        "card_bg_eda": "#fffaf5",
        "border_eda": "#e6dccb",
        "header_beige": "#FFFFFF",
        "card_bg_soft": "#faf9f6",
        "card_bg_strong": "#FFFFFF",
    }
}

FEATURES = CONFIG["features"]
COLORS = CONFIG["colors"]

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Customer Segmentation System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Eliminate top white space
st.markdown("<style>div.block-container{padding-top:1rem !important;}</style>", unsafe_allow_html=True)

# -------------------------
# GLOBAL STYLES (CSS)
# -------------------------
st.markdown(f"""
<style>
/* Page background */
.stApp {{
    background-color: {COLORS['background']} !important;
}}

/* Body text */
.stApp p, .stApp div, .stApp span, .stApp label {{
    color: {COLORS['text_primary']} !important;
}}

/* Headers black */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
    color: #000000 !important;
}}

/* Banner text white */
.header-banner h1, .header-banner p {{
    color: {COLORS['header_beige']} !important;
}}

/* EDA soft card */
.soft-card {{
    background-color: {COLORS['card_bg_eda']} !important;
    border: 1px solid {COLORS['border_eda']} !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin-bottom: 16px !important;
    box-shadow: 0 3px 8px rgba(0,0,0,0.05) !important;
}}

/* Outer soft wrapper */
.card-soft {{
    background-color: {COLORS['card_bg_soft']} !important;
    border: 1px solid {COLORS['border_eda']} !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04) !important;
}}

/* Expander styling */
.stExpander summary {{
    font-weight: 800 !important;
    color: #000000 !important;
    font-size: 1rem !important;
}}
.stExpander div[data-testid="stExpanderContent"] {{
    background-color: {COLORS['card_bg_strong']} !important;
    border-radius: 10px !important;
    padding: 14px !important;
    margin-top: 8px !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06) !important;
    border: 1px solid #eee;
}}

.expander-heading {{
    color: #000 !important;
    font-weight: 800;
    margin-bottom: 6px;
    font-size: 1.05rem;
}}

/* KPI cards */
.kpi-card {{
    background-color: {COLORS['card_bg_strong']};
    padding: 12px;
    border-radius: 8px;
    border-left: 5px solid {COLORS['accent']};
    box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    color: {COLORS['text_primary']};
}}

/* Plots */
.stPlotlyChart {{
    margin-top: 0 !important;
    padding-top: 0 !important;
}}

/* Responsive */
@media (max-width: 768px) {{
    .soft-card, .card-soft {{
        padding: 12px !important;
        margin-bottom: 12px !important;
    }}
    h1, h2, h3 {{
        font-size: 1.5rem !important;
    }}
    .stColumn {{
        width: 100% !important;
    }}
}}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Helper wrappers (now unused in EDA)
# -------------------------
def start_soft_card():
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)

def end_soft_card():
    st.markdown('</div>', unsafe_allow_html=True)

def start_card_soft():
    st.markdown('<div class="card-soft">', unsafe_allow_html=True)

def end_card_soft():
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# User Authentication
# =============================================================================
USERS = {
    "admin": "segment2024",
    "analyst1": "analyze2024",
    "manager": "lead2024"
}

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

# =============================================================================
# Load Raw Data
# =============================================================================
@st.cache_data
def load_raw_data():
    for path in ['marketing_campaign.csv', 'data/marketing_campaign.csv']:
        if os.path.exists(path):
            try:
                try:
                    df = pd.read_csv(path, sep='\t')
                    return df
                except:
                    df = pd.read_csv(path)
                    return df
            except Exception as e:
                st.warning(f"Error loading {path}: {e}")
    return None

# =============================================================================
# Load Models & Metrics
# =============================================================================
@st.cache_resource
def load_models():
    def safe_load(path):
        try:
            return joblib.load(path) if os.path.exists(path) else None
        except Exception as e:
            st.warning(f"Failed to load {path}: {e}")
            return None

    scaler = safe_load('models/scaler.pkl')
    pca = safe_load('models/pca.pkl')
    clusterer = safe_load('models/clusterer.pkl')
    supervised_model = safe_load('models/segment_classifier.pkl')

    profiles = None
    if os.path.exists('models/cluster_profiles.csv'):
        try:
            profiles = pd.read_csv('models/cluster_profiles.csv', index_col=0)
        except Exception as e:
            st.warning(f"Failed to load cluster profiles: {e}")

    return scaler, pca, clusterer, profiles, supervised_model

@st.cache_data
def load_model_metrics():
    try:
        with open('models/model_metrics.json', 'r') as f:
            return json.load(f)
    except:
        return None

# =============================================================================
# LOGIN PAGE
# =============================================================================
if not st.session_state.logged_in:
    st.title("🔐 Targeted Marketing System for Grocery Retail")
    st.markdown("Enter credentials to access confidential grocery retail dashboard")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        if submit:
            if username in USERS and USERS[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
    st.stop()

# =============================================================================
# LOGOUT BUTTON
# =============================================================================
col_spacer, col_logout = st.columns([6, 1])
with col_logout:
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

# =============================================================================
# Preprocessing Function
# =============================================================================
def preprocess_for_eda(df_raw):
    df = df_raw.copy()
    df = df.dropna(subset=['Income'])
    if "Year_Birth" in df.columns:
        df["Age"] = 2024 - df["Year_Birth"]
        df = df[df["Age"] <= 100].copy()
    else:
        if "Age" not in df.columns:
            df["Age"] = np.nan
    if "Marital_Status" in df.columns:
        df["Marital_Status"] = df["Marital_Status"].replace(
            {"Absurd": "Others", "YOLO": "Others", "Alone": "Others"}
        )
    if "Education" in df.columns:
        df["Education"] = df["Education"].replace({
            "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
            "Graduation": "Graduate", "Master": "Postgraduate", "PhD": "Postgraduate"
        })
    if all(c in df.columns for c in ["Kidhome", "Teenhome"]):
        df["Has_Children"] = ((df["Kidhome"] + df["Teenhome"]) > 0).astype(int)
    spending_cols = ['MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds']
    present_spend_cols = [c for c in spending_cols if c in df.columns]
    if present_spend_cols:
        df["TotalMnt"] = df[present_spend_cols].sum(axis=1)
    purchase_cols = ['NumWebPurchases','NumCatalogPurchases','NumStorePurchases']
    present_purchase_cols = [c for c in purchase_cols if c in df.columns]
    if present_purchase_cols:
        df["TotalPurchases"] = df[present_purchase_cols].sum(axis=1)
    cmp_cols = ['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','Response']
    present_cmp_cols = [c for c in cmp_cols if c in df.columns]
    if present_cmp_cols:
        df["TotalAcceptedCmp"] = df[present_cmp_cols].sum(axis=1)
    if "Dt_Customer" in df.columns:
        df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format='%d-%m-%Y', errors='coerce')
        df = df.dropna(subset=['Dt_Customer'])
        df["Days_Customer"] = (pd.to_datetime("2024-01-01") - df["Dt_Customer"]).dt.days
    return df

# =============================================================================
# NAVIGATION
# =============================================================================
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Customer Dashboard", "Predict Customer Segment"])

# =============================================================================
# HOME PAGE
# =============================================================================
if page == "Home":
    header_image_path = "header.jpeg"
    is_image = False
    try:
        with open(header_image_path, "rb") as f:
            img_data = f.read()
        img_base64 = base64.b64encode(img_data).decode()
        bg_image = f"data:image/jpeg;base64,{img_base64}"
        is_image = True
    except FileNotFoundError:
        is_image = False

    bg_style = f"url('{bg_image}')" if is_image else COLORS['accent']

    st.markdown(f"""
    <style>
        .header-banner {{
            background-image: {bg_style};
            background-size: cover;
            background-position: center;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 18px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        }}
        .header-banner h1 {{
            font-size: 2.3rem;
            margin: 0;
            font-weight: 700;
        }}
        .header-banner p {{
            font-size: 1.05rem;
            margin-top: 8px;
            opacity: 0.95;
        }}
    </style>

    <div class="header-banner">
        <h1>Discover What Drives Your Shoppers</h1>
        <p>Unlock actionable insights with AI-powered customer segmentation</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ## 🎯 Introduction
    This system segments customers into **two distinct groups** using machine learning:
    - **Budget Segment**: Value-conscious, price-sensitive shoppers
    - **Premium Segment**: High-income, brand-loyal customers

    Insights help tailor marketing, improve retention, and boost ROI.
    """)

    st.markdown("---")
    st.subheader("📊 Dataset Overview")
    df_raw = load_raw_data()
    if df_raw is not None:
        df_home = preprocess_for_eda(df_raw)
        st.write(f"Records after cleaning: **{len(df_home)}**")
        show_cols = [c for c in FEATURES if c in df_home.columns]
        st.dataframe(df_home[show_cols].head(3) if show_cols else df_home.head(3))

        avg_spent = df_home['TotalMnt'].mean() if 'TotalMnt' in df_home.columns else 0
        avg_age = df_home['Age'].mean() if 'Age' in df_home.columns else 0
        avg_income = df_home['Income'].mean() if 'Income' in df_home.columns else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="kpi-card">
                <h4 style="margin:0;">💰 Avg. Spending</h4>
                <p style="font-size:20px; font-weight:bold; margin:6px 0 0 0;">${avg_spent:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="kpi-card">
                <h4 style="margin:0;">👤 Avg. Age</h4>
                <p style="font-size:20px; font-weight:bold; margin:6px 0 0 0;">{avg_age:.1f}</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="kpi-card">
                <h4 style="margin:0;">💳 Avg. Income</h4>
                <p style="font-size:20px; font-weight:bold; margin:6px 0 0 0;">${avg_income:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error("Dataset not found. Place `marketing_campaign.csv` in the project root.")

    st.markdown("---")
    st.subheader("💡 Why Two Segments?")
    st.markdown("""
    - Simplifies decision-making
    - Aligns with natural customer split (spending median)
    - Enables clear, actionable marketing strategies
    """)

# =============================================================================
# CUSTOMER DASHBOARD (EDA)
# =============================================================================
elif page == "Customer Dashboard":
    st.title("📈 Customer Dashboard")

    df_raw = load_raw_data()
    if df_raw is None:
        st.error("Dataset not found.")
        st.stop()

    df_full = preprocess_for_eda(df_raw)

    # Filters
    st.sidebar.header("🔍 Filters")
    age_bins = [18, 30, 45, 60, 100]
    age_labels = ["18-29", "30-44", "45-59", "60+"]
    if 'Age' in df_full.columns:
        df_full['AgeGroup'] = pd.cut(df_full['Age'], bins=age_bins, labels=age_labels, right=False)
    else:
        df_full['AgeGroup'] = "Unknown"

    selected_age_group = st.sidebar.selectbox("Age Group", ["All"] + list(age_labels))

    min_income = int(df_full['Income'].min()) if 'Income' in df_full.columns else 0
    max_income = int(df_full['Income'].max()) if 'Income' in df_full.columns else 100000
    income_range = st.sidebar.slider(
        "Income Range ($)",
        min_value=min_income,
        max_value=max_income,
        value=(min_income, max_income),
        step=5000
    )

    min_spent = int(df_full['TotalMnt'].min()) if 'TotalMnt' in df_full.columns else 0
    max_spent = int(df_full['TotalMnt'].max()) if 'TotalMnt' in df_full.columns else 10000
    spending_range = st.sidebar.slider(
        "Total Spending Range ($)",
        min_value=min_spent,
        max_value=max_spent,
        value=(min_spent, max_spent),
        step=100
    )

    education_options = ["All"] + sorted(df_full['Education'].unique().tolist()) if 'Education' in df_full.columns else ["All"]
    selected_education = st.sidebar.selectbox("Education Level", education_options)
    marital_options = ["All"] + sorted(df_full['Marital_Status'].unique().tolist()) if 'Marital_Status' in df_full.columns else ["All"]
    selected_marital = st.sidebar.selectbox("Marital Status", marital_options)
    children_options = ["All", "Yes", "No"]
    selected_children = st.sidebar.selectbox("Has Children", children_options)

    # Apply filters
    filtered_df = df_full.copy()
    if selected_age_group != "All":
        filtered_df = filtered_df[filtered_df['AgeGroup'] == selected_age_group]
    filtered_df = filtered_df[
        (filtered_df['Income'] >= income_range[0]) &
        (filtered_df['Income'] <= income_range[1])
    ] if 'Income' in filtered_df.columns else filtered_df
    filtered_df = filtered_df[
        (filtered_df['TotalMnt'] >= spending_range[0]) &
        (filtered_df['TotalMnt'] <= spending_range[1])
    ] if 'TotalMnt' in filtered_df.columns else filtered_df
    if selected_education != "All":
        filtered_df = filtered_df[filtered_df['Education'] == selected_education]
    if selected_marital != "All":
        filtered_df = filtered_df[filtered_df['Marital_Status'] == selected_marital]
    if selected_children == "Yes":
        filtered_df = filtered_df[filtered_df['Has_Children'] == 1]
    elif selected_children == "No":
        filtered_df = filtered_df[filtered_df['Has_Children'] == 0]

    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
        st.stop()

    # KPIs
    avg_spent = filtered_df['TotalMnt'].mean() if 'TotalMnt' in filtered_df.columns else 0
    avg_age = filtered_df['Age'].mean() if 'Age' in filtered_df.columns else 0
    avg_income = filtered_df['Income'].mean() if 'Income' in filtered_df.columns else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <h4 style="margin:0;">💰 Avg. Spending</h4>
            <p style="font-size:20px; font-weight:bold; margin:6px 0 0 0;">${avg_spent:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <h4 style="margin:0;">👤 Avg. Age</h4>
            <p style="font-size:20px; font-weight:bold; margin:6px 0 0 0;">{avg_age:.1f}</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <h4 style="margin:0;">💳 Avg. Income</h4>
            <p style="font-size:20px; font-weight:bold; margin:6px 0 0 0;">${avg_income:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Demographics",
        "Purchase Behavior",
        "Campaign Response",
        "Clustering & Marketing"
    ])

    # ---------------------------
    # Tab 1: Demographics
    # ---------------------------
    with tab1:
        st.markdown("## 👥 Demographics")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            hist_data = filtered_df['Age'] if 'Age' in filtered_df.columns else pd.Series([])
            if not hist_data.empty:
                kde = gaussian_kde(hist_data)
                x_vals = np.linspace(hist_data.min(), hist_data.max(), 200)
                y_kde = kde(x_vals) * len(hist_data) * (hist_data.max() - hist_data.min()) / 20
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=hist_data,
                    nbinsx=20,
                    marker_color=COLORS["accent"],
                    opacity=0.7,
                    hovertemplate="<b>Age</b>: %{x}<br><b>Count</b>: %{y}<extra></extra>",
                    name="Age"
                ))
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_kde,
                    mode='lines',
                    line=dict(color=COLORS["accent_hover"], width=3),
                    name="Trend (KDE)"
                ))
                fig.update_layout(
                    title='Age Distribution',
                    xaxis_title='Age',
                    yaxis_title='Count',
                    plot_bgcolor=COLORS["card_bg_eda"],
                    paper_bgcolor=COLORS["card_bg_eda"],
                    font_color=COLORS["text_primary"]
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Age data not available.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            hist_data = filtered_df['Income'] if 'Income' in filtered_df.columns else pd.Series([])
            if not hist_data.empty:
                kde = gaussian_kde(hist_data)
                x_vals = np.linspace(hist_data.min(), hist_data.max(), 200)
                y_kde = kde(x_vals) * len(hist_data) * (hist_data.max() - hist_data.min()) / 30
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=hist_data,
                    nbinsx=30,
                    marker_color=COLORS["accent"],
                    opacity=0.7,
                    hovertemplate="<b>Income</b>: $%{x:,.0f}<br><b>Count</b>: %{y}<extra></extra>",
                    name="Income"
                ))
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_kde,
                    mode='lines',
                    line=dict(color=COLORS["accent_hover"], width=3),
                    name="Trend (KDE)"
                ))
                fig.update_layout(
                    title='Income Distribution',
                    xaxis_title='Income ($)',
                    yaxis_title='Count',
                    plot_bgcolor=COLORS["card_bg_eda"],
                    paper_bgcolor=COLORS["card_bg_eda"],
                    font_color=COLORS["text_primary"]
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Income data not available.")
            st.markdown('</div>', unsafe_allow_html=True)

        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            if 'Education' in filtered_df.columns:
                edu_counts = filtered_df['Education'].value_counts().reset_index()
                edu_counts.columns = ['Education', 'Count']
                fig = px.bar(
                    edu_counts,
                    x='Education',
                    y='Count',
                    title='Education Level',
                    color='Count',
                    color_continuous_scale='Oranges',
                    labels={'Education': 'Education Level', 'Count': 'Customer Count'}
                )
                fig.update_traces(
                    hovertemplate="<b>Education</b>: %{x}<br><b>Count</b>: %{y}<extra></extra>"
                )
                fig.update_layout(
                    plot_bgcolor=COLORS["card_bg_eda"],
                    paper_bgcolor=COLORS["card_bg_eda"],
                    font_color=COLORS["text_primary"],
                    xaxis_title="Education Level",
                    yaxis_title="Customer Count"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Education data not available.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_d:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            if 'Marital_Status' in filtered_df.columns:
                mar_counts = filtered_df['Marital_Status'].value_counts().reset_index()
                mar_counts.columns = ['Marital Status', 'Count']
                fig = px.bar(
                    mar_counts,
                    x='Marital Status',
                    y='Count',
                    title='Marital Status',
                    color='Count',
                    color_continuous_scale='Greens',
                    labels={'Marital Status': 'Marital Status', 'Count': 'Customer Count'}
                )
                fig.update_traces(
                    hovertemplate="<b>Marital Status</b>: %{x}<br><b>Count</b>: %{y}<extra></extra>"
                )
                fig.update_layout(
                    plot_bgcolor=COLORS["card_bg_eda"],
                    paper_bgcolor=COLORS["card_bg_eda"],
                    font_color=COLORS["text_primary"],
                    xaxis_title="Marital Status",
                    yaxis_title="Customer Count"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Marital status data not available.")
            st.markdown('</div>', unsafe_allow_html=True)

        start_card_soft()
        with st.expander("📌 Key Takeaways (Demographics)", expanded=False):
            st.markdown(f"<div class='expander-heading'>Summary</div>", unsafe_allow_html=True)
            st.markdown("Summarize main demographic findings here.")
        with st.expander("🚀 Actionable Insights (Demographics)", expanded=False):
            st.markdown(f"<div class='expander-heading'>Next Steps</div>", unsafe_allow_html=True)
            st.markdown("List recommended actions.")
        end_card_soft()

    # ---------------------------
    # Tab 2: Purchase Behavior
    # ---------------------------
    with tab2:
        st.markdown("## 🛒 Purchase Behavior")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            spending_cols = ['MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds']
            present_spending = [c for c in spending_cols if c in filtered_df.columns]
            if present_spending:
                spending_means = filtered_df[present_spending].mean().reset_index()
                spending_means.columns = ['Category', 'Avg Spending']
                fig = px.bar(
                    spending_means,
                    x='Category',
                    y='Avg Spending',
                    title='Spending by Category',
                    color='Avg Spending',
                    color_continuous_scale='Blues',
                    labels={'Category': 'Product Category', 'Avg Spending': 'Average Spending ($)'}
                )
                fig.update_traces(
                    hovertemplate="<b>Category</b>: %{x}<br><b>Avg Spending</b>: $%{y:,.0f}<extra></extra>"
                )
                fig.update_layout(
                    plot_bgcolor=COLORS["card_bg_eda"],
                    paper_bgcolor=COLORS["card_bg_eda"],
                    font_color=COLORS["text_primary"],
                    xaxis_title="Product Category",
                    yaxis_title="Average Spending ($)"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Spending category columns not available.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            purchase_types = ['NumWebPurchases','NumCatalogPurchases','NumStorePurchases']
            present_purchase_types = [c for c in purchase_types if c in filtered_df.columns]
            if present_purchase_types:
                purchase_means = filtered_df[present_purchase_types].mean().reset_index()
                purchase_means.columns = ['Channel', 'Avg Purchases']
                fig = px.bar(
                    purchase_means,
                    x='Channel',
                    y='Avg Purchases',
                    title='Purchase Channel Preference',
                    color='Avg Purchases',
                    color_continuous_scale='Purples',
                    labels={'Channel': 'Purchase Channel', 'Avg Purchases': 'Average Purchases'}
                )
                fig.update_traces(
                    hovertemplate="<b>Channel</b>: %{x}<br><b>Avg Purchases</b>: %{y:.1f}<extra></extra>"
                )
                fig.update_layout(
                    plot_bgcolor=COLORS["card_bg_eda"],
                    paper_bgcolor=COLORS["card_bg_eda"],
                    font_color=COLORS["text_primary"],
                    xaxis_title="Purchase Channel",
                    yaxis_title="Average Purchases"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Purchase channel columns not available.")
            st.markdown('</div>', unsafe_allow_html=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            if 'Income' in filtered_df.columns and 'TotalMnt' in filtered_df.columns:
                fig = px.scatter(
                    filtered_df,
                    x='Income',
                    y='TotalMnt',
                    color='Has_Children' if 'Has_Children' in filtered_df.columns else None,
                    hover_data=[c for c in ['Age', 'Education', 'TotalPurchases'] if c in filtered_df.columns],
                    title='Income vs Total Spending',
                    labels={'Income': 'Income ($)', 'TotalMnt': 'Total Spending ($)'}
                )
                fig.update_layout(
                    plot_bgcolor=COLORS["card_bg_eda"],
                    paper_bgcolor=COLORS["card_bg_eda"],
                    font_color=COLORS["text_primary"],
                    xaxis_title="Income ($)",
                    yaxis_title="Total Spending ($)"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Income or Total Spending not available for scatter plot.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            if 'TotalPurchases' in filtered_df.columns:
                hist_data = filtered_df['TotalPurchases']
                kde = gaussian_kde(hist_data)
                x_vals = np.linspace(max(0, hist_data.min()), hist_data.max(), 200)
                y_kde = kde(x_vals) * len(hist_data) * (hist_data.max() - hist_data.min()) / 20
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=hist_data,
                    nbinsx=20,
                    marker_color=COLORS["accent"],
                    opacity=0.7,
                    hovertemplate="<b>Total Purchases</b>: %{x}<br><b>Count</b>: %{y}<extra></extra>",
                    name="Total Purchases"
                ))
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_kde,
                    mode='lines',
                    line=dict(color=COLORS["accent_hover"], width=3),
                    name="Trend (KDE)"
                ))
                fig.update_layout(
                    title='Total Purchases Distribution',
                    xaxis_title='Total Purchases',
                    yaxis_title='Count',
                    plot_bgcolor=COLORS["card_bg_eda"],
                    paper_bgcolor=COLORS["card_bg_eda"],
                    font_color=COLORS["text_primary"]
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("TotalPurchases column not available.")
            st.markdown('</div>', unsafe_allow_html=True)

        start_card_soft()
        with st.expander("📌 Key Takeaways (Purchase Behavior)", expanded=False):
            st.markdown(f"<div class='expander-heading'>Summary</div>", unsafe_allow_html=True)
            st.markdown("Summarize main purchase behavior findings here.")
        with st.expander("🚀 Actionable Insights (Purchase Behavior)", expanded=False):
            st.markdown(f"<div class='expander-heading'>Next Steps</div>", unsafe_allow_html=True)
            st.markdown("List recommended actions.")
        end_card_soft()

    # ---------------------------
    # Tab 3: Campaign Response
    # ---------------------------
    with tab3:
        st.markdown("## 📣 Campaign Response")

        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        campaign_cols = ['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','Response']
        present_campaign_cols = [c for c in campaign_cols if c in filtered_df.columns]
        if present_campaign_cols:
            acceptance_rates = (filtered_df[present_campaign_cols].mean() * 100).reset_index()
            acceptance_rates.columns = ['Campaign', 'Acceptance Rate (%)']
            fig = px.bar(
                acceptance_rates,
                x='Campaign',
                y='Acceptance Rate (%)',
                title='Campaign Acceptance Rates (%)',
                color='Acceptance Rate (%)',
                color_continuous_scale='Reds',
                labels={'Campaign': 'Campaign', 'Acceptance Rate (%)': 'Acceptance Rate (%)'}
            )
            fig.update_traces(
                hovertemplate="<b>Campaign</b>: %{x}<br><b>Acceptance Rate</b>: %{y:.1f}%<extra></extra>"
            )
            fig.update_layout(
                plot_bgcolor=COLORS["card_bg_eda"],
                paper_bgcolor=COLORS["card_bg_eda"],
                font_color=COLORS["text_primary"],
                xaxis_title="Campaign",
                yaxis_title="Acceptance Rate (%)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Campaign columns not available.")
        st.markdown('</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            if 'TotalAcceptedCmp' in filtered_df.columns and 'TotalMnt' in filtered_df.columns:
                fig = px.scatter(
                    filtered_df,
                    x='TotalAcceptedCmp',
                    y='TotalMnt',
                    color='TotalAcceptedCmp' if 'TotalAcceptedCmp' in filtered_df.columns else None,
                    hover_data=[c for c in ['Age', 'Income', 'Has_Children'] if c in filtered_df.columns],
                    title='Campaigns Accepted vs Spending',
                    labels={'TotalAcceptedCmp': 'Campaigns Accepted', 'TotalMnt': 'Total Spending ($)'},
                    color_continuous_scale='YlOrRd'
                )
                fig.update_layout(
                    plot_bgcolor=COLORS["card_bg_eda"],
                    paper_bgcolor=COLORS["card_bg_eda"],
                    font_color=COLORS["text_primary"],
                    xaxis_title="Campaigns Accepted",
                    yaxis_title="Total Spending ($)"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Required columns not available for this scatter.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            if 'NumWebVisitsMonth' in filtered_df.columns and 'TotalPurchases' in filtered_df.columns:
                fig = px.scatter(
                    filtered_df,
                    x='NumWebVisitsMonth',
                    y='TotalPurchases',
                    color='NumWebVisitsMonth',
                    hover_data=[c for c in ['Age', 'Income', 'TotalMnt'] if c in filtered_df.columns],
                    title='Web Visits vs Total Purchases',
                    labels={'NumWebVisitsMonth': 'Monthly Web Visits', 'TotalPurchases': 'Total Purchases'},
                    color_continuous_scale='Oranges'
                )
                fig.update_layout(
                    plot_bgcolor=COLORS["card_bg_eda"],
                    paper_bgcolor=COLORS["card_bg_eda"],
                    font_color=COLORS["text_primary"],
                    xaxis_title="Monthly Web Visits",
                    yaxis_title="Total Purchases"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Required columns not available for this scatter.")
            st.markdown('</div>', unsafe_allow_html=True)

        start_card_soft()
        with st.expander("📌 Key Takeaways (Campaigns)", expanded=False):
            st.markdown(f"<div class='expander-heading'>Summary</div>", unsafe_allow_html=True)
            st.markdown("Summarize campaign response highlights here.")
        with st.expander("🚀 Actionable Insights (Campaigns)", expanded=False):
            st.markdown(f"<div class='expander-heading'>Next Steps</div>", unsafe_allow_html=True)
            st.markdown("List recommended campaign steps.")
        end_card_soft()

    # ---------------------------
    # Tab 4: Clustering & Marketing — ✅ FIXED WITH INLINE HTML
    # ---------------------------
    with tab4:
        st.markdown("## 🧩 Customer Personas (2 Strategic Clusters)")

        scaler, pca, clusterer, cluster_profiles, supervised_model = load_models()

        if cluster_profiles is not None and len(cluster_profiles) >= 2:
            try:
                cluster_profiles.index = [int(i) for i in cluster_profiles.index]
            except:
                pass

            cols = st.columns(2)
            cluster_ids = cluster_profiles.index.tolist()[:2]

            for i, cluster_id in enumerate(cluster_ids):
                with cols[i]:
                    start_card_soft()
                    st.markdown(f"### Cluster {cluster_id}: {CONFIG['cluster_descriptions'].get(cluster_id, f'Cluster {cluster_id}')}")
                    
                    # ✅ Wrap in inline HTML for guaranteed styling
                    st.markdown("""
                    <div style='
                        background-color: #ffffff;
                        border: 1px solid #e9e3d8;
                        border-radius: 12px;
                        box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
                        padding: 20px;
                        margin-bottom: 16px;
                    '>
                    """, unsafe_allow_html=True)

                    st.markdown("#### 👤 Behavioral Profile")
                    try:
                        top_features = cluster_profiles.loc[cluster_id].sort_values(ascending=False).head(6)
                        for feat, val in top_features.items():
                            st.markdown(f"- **{feat}**: {val:.2f}")
                    except Exception:
                        st.markdown("- Profile summary not available.")

                    with st.expander("💡 Why This Matters", expanded=False):
                        st.markdown(f"<div class='expander-heading'>Why this cluster matters</div>", unsafe_allow_html=True)
                        st.markdown("Summarize the business importance and conversion opportunities for this cluster here.")

                    categories = list(cluster_profiles.columns)
                    try:
                        values = cluster_profiles.loc[cluster_id].values.flatten().tolist()
                        values_plot = values + values[:1]
                        theta = categories + [categories[0]]
                        fig = go.Figure(
                            data=go.Scatterpolar(
                                r=values_plot,
                                theta=theta,
                                fill='toself',
                                line_color=COLORS["accent"],
                                hoverinfo='text',
                                text=[f"{cat}: {val:.2f}" for cat, val in zip(categories, cluster_profiles.loc[cluster_id])]
                            )
                        )
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(visible=True, range=[0, 1]),
                                angularaxis=dict(direction="clockwise")
                            ),
                            showlegend=False,
                            title=f"Behavioral Profile",
                            plot_bgcolor="#ffffff",
                            paper_bgcolor="#ffffff",
                            font_color=COLORS["text_primary"]
                        )
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    except Exception:
                        st.info("Radar chart could not be generated for this cluster.")

                    with st.expander("💼 Recommended Marketing Strategy", expanded=False):
                        st.markdown(f"<div class='expander-heading'>Actionable Tactics</div>", unsafe_allow_html=True)
                        marketing_text = CONFIG.get("cluster_marketing", {}).get(cluster_id, "")
                        if marketing_text:
                            for line in marketing_text.split("\n"):
                                if line.strip():
                                    st.markdown(f"- {line.strip()}")
                        else:
                            st.markdown("- Add recommended marketing tactics here.")

                    st.markdown("</div>", unsafe_allow_html=True)  # Close cluster card
                    end_card_soft()

            # Heatmap — preserved
            st.markdown("### 🌡️ Cluster Comparison 🔍")
            start_card_soft()
            df_heatmap = cluster_profiles.reset_index()
            df_heatmap.columns.name = None
            df_melted = df_heatmap.melt(id_vars=df_heatmap.columns[0], var_name='Feature', value_name='Value')
            try:
                fig = px.density_heatmap(
                    df_melted,
                    x='Feature',
                    y=df_heatmap.columns[0],
                    z='Value',
                    title='Cluster Feature Comparison (Normalized)',
                    color_continuous_scale='YlGnBu',
                    labels={'Feature': 'Feature', df_heatmap.columns[0]: 'Cluster', 'Value': 'Normalized Value'}
                )
                fig.update_traces(
                    hovertemplate="<b>Cluster</b>: %{y}<br><b>Feature</b>: %{x}<br><b>Value</b>: %{z:.2f}<extra></extra>"
                )
                fig.update_layout(
                    plot_bgcolor=COLORS["card_bg_soft"],
                    paper_bgcolor=COLORS["card_bg_soft"],
                    font_color=COLORS["text_primary"]
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.info("Heatmap could not be generated. Check cluster_profiles format.")
            end_card_soft()

            # Marketing Strategies Overview
            st.markdown("### 💡 Marketing Strategies Overview")
            start_card_soft()
            mcol1, mcol2 = st.columns(2)
            for idx, cid in enumerate(cluster_ids):
                with (mcol1 if idx == 0 else mcol2):
                    st.markdown(f"#### Cluster {cid}")
                    marketing_notes = CONFIG.get("cluster_marketing", {}).get(cid, "")
                    if marketing_notes:
                        for line in marketing_notes.split("\n"):
                            if line.strip():
                                st.markdown(f"- {line.strip()}")
                    else:
                        st.markdown("- No marketing notes available.")
            end_card_soft()

            # Final expanders
            start_card_soft()
            with st.expander("📌 Key Takeaways (Clustering)", expanded=False):
                st.markdown(f"<div class='expander-heading'>Summary</div>", unsafe_allow_html=True)
                st.markdown("Summarize key findings from clustering.")
            with st.expander("🚀 Actionable Insights (Clustering)", expanded=False):
                st.markdown(f"<div class='expander-heading'>Recommendations</div>", unsafe_allow_html=True)
                st.markdown("List concrete tactical actions.")
            end_card_soft()

            # Model Performance
            st.markdown("## 📈 Model Performance")
            metrics = load_model_metrics()
            if metrics:
                colp1, colp2 = st.columns(2)
                with colp1:
                    silhouette = metrics.get('silhouette_score', None)
                    if silhouette is not None:
                        st.metric("Silhouette Score (Clustering)", f"{silhouette:.3f}")
                        st.caption("Measures cluster separation (higher = better)")
                if 'logistic_accuracy' in metrics:
                    with colp2:
                        st.metric("Logistic Regression Accuracy", f"{metrics['logistic_accuracy']:.2%}")
                        st.caption("Classification performance on test set")
                    class_metrics = {
                        "Precision": metrics.get("logistic_precision", 0),
                        "Recall": metrics.get("logistic_recall", 0),
                        "F1-Score": metrics.get("logistic_f1", 0)
                    }
                    st.dataframe(pd.DataFrame([class_metrics]).T.rename(columns={0: "Score"}))
                    if "confusion_matrix" in metrics:
                        cm = np.array(metrics["confusion_matrix"])
                        fig_cm = px.imshow(
                            cm,
                            text_auto=True,
                            labels=dict(x="Predicted", y="Actual"),
                            x=["Budget-Conscious Parent", "Responsive High Spenders"],
                            y=["Budget-Conscious Parent", "Responsive High Spenders"],
                            color_continuous_scale="Blues",
                            title="Confusion Matrix"
                        )
                        fig_cm.update_layout(
                            plot_bgcolor=COLORS["background"],
                            paper_bgcolor=COLORS["background"],
                            font_color=COLORS["text_primary"]
                        )
                        st.plotly_chart(fig_cm, use_container_width=True)
            else:
                st.info("Model metrics not found. Place `model_metrics.json` in the `models/` folder.")
        else:
            st.error("Clustering model not found or cluster_profiles missing/invalid.")

# =============================================================================
# PREDICT CUSTOMER SEGMENT
# =============================================================================
elif page == "Predict Customer Segment":
    st.title("🔮 Predict Customer Segment")
    scaler, pca, clusterer, cluster_profiles, supervised_model = load_models()
    if scaler is None or cluster_profiles is None:
        st.error("Models or cluster profiles not loaded.")
        st.stop()

    cluster_insights = {
        0: {
            "name": "Responsive High Spenders",
            "profile": [
                "✅ Higher income and younger demographic",
                "✅ Actively engages with promotional campaigns",
                "✅ High total spending and responds to 2+ campaigns on average"
            ],
            "why_it_matters": [
                "• Campaigns work because they match their lifestyle — they see offers as value, not noise"
            ],
            "marketing": [
                "🎯 Personalize dynamically: 'Recommended for you'",
                "✨ Offer flash sales & VIP early access"
            ]
        },
        1: {
            "name": "Budget-Conscious Parent",
            "profile": [
                "👨‍👩‍👧‍👦 Family-oriented; frequent low-spend transactions"
            ],
            "why_it_matters": [
                "• They browse often and need utility-driven messaging"
            ],
            "marketing": [
                "🧭 Use contextual nudges and show loyalty points"
            ]
        }
    }

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
            has_children = st.selectbox("Has Children?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            total_mnt = st.number_input("Total Spending ($)", min_value=0, value=600, step=50)
        with col2:
            total_purchases = st.number_input("Total Purchases", min_value=0, value=20)
            web_visits = st.number_input("Monthly Web Visits", min_value=0, value=5)
            total_cmp = st.number_input(
                "Accepted Campaigns (0–6)",
                min_value=0,
                max_value=6,
                value=1,
                help="Total campaigns accepted (including last campaign 'Response')"
            )
            days_customer = st.number_input("Days as Customer", min_value=0, value=365)
        submitted = st.form_submit_button("🚀 Predict Segment")

    if submitted:
        input_data = np.array([[age, income, has_children, total_mnt, total_purchases,
                                web_visits, total_cmp, days_customer]])
        try:
            scaled = scaler.transform(input_data)
            if supervised_model is not None:
                pred_label = int(supervised_model.predict(scaled)[0])
            else:
                reduced = pca.transform(scaled)
                pred_label = int(clusterer.predict(reduced)[0])

            insight = cluster_insights.get(pred_label, cluster_insights[0])
            segment_name = insight["name"]

            confidence_msg = ""
            if supervised_model is not None and hasattr(supervised_model, "predict_proba"):
                proba = supervised_model.predict_proba(scaled)[0]
                confidence = max(proba) * 100
                confidence_msg = f" (Confidence: **{confidence:.1f}%**)"

            st.success(f"### 🎯 Predicted Segment: **{segment_name}**{confidence_msg}")

            st.markdown("### 👤 Customer Profile")
            for point in insight["profile"]:
                st.markdown(f"- {point}")

            with st.expander("💡 Why This Matters"):
                for point in insight["why_it_matters"]:
                    st.markdown(f"- {point}")

            st.markdown("### 📊 Behavioral Radar Chart")
            categories = list(cluster_profiles.columns)
            try:
                values = cluster_profiles.loc[pred_label].values.flatten().tolist()
                values_plot = values + values[:1]
                fig = go.Figure(
                    data=go.Scatterpolar(
                        r=values_plot,
                        theta=categories + [categories[0]],
                        fill='toself',
                        line_color=COLORS["accent"],
                        hoverinfo='text',
                        text=[f"{cat}: {val:.2f}" for cat, val in zip(categories, cluster_profiles.loc[pred_label])]
                    )
                )
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1]),
                        angularaxis=dict(direction="clockwise")
                    ),
                    showlegend=False,
                    title=f"Cluster {pred_label} Behavioral Profile",
                    plot_bgcolor=COLORS["background"],
                    paper_bgcolor=COLORS["background"],
                    font_color=COLORS["text_primary"]
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.info("Radar chart not available for predicted cluster.")

            with st.expander("💼 Recommended Marketing Strategy"):
                st.markdown("#### Actionable Tactics:")
                for tactic in insight["marketing"]:
                    st.markdown(f"- {tactic}")

        except Exception as e:
            st.error(f"Prediction error: {e}")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.caption("💡 Powered by binary customer segmentation using Logistic Regression and KMeans.")