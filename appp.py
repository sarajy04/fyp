# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from math import pi
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Customer Segmentation System",
    page_icon="🛒",
    layout="wide"
)

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
                df = pd.read_csv(path, sep='\t')
                return df
            except Exception as e:
                st.warning(f"Error loading {path}: {e}")
    return None

# =============================================================================
# Load Models
# =============================================================================
@st.cache_resource
def load_models():
    if not os.path.exists('models'):
        return None, None, None, None, None
    try:
        scaler = joblib.load('models/scaler.pkl')
        pca = joblib.load('models/pca.pkl')
        clusterer = joblib.load('models/clusterer.pkl')
        profiles = pd.read_csv('models/cluster_profiles.csv', index_col=0)
        try:
            supervised_model = joblib.load('models/segment_classifier.pkl')
        except:
            supervised_model = None
        return scaler, pca, clusterer, profiles, supervised_model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None, None, None, None

# =============================================================================
# Login Page
# =============================================================================
if not st.session_state.logged_in:
    st.title("🔐 Customer Segmentation System - Login")
    st.markdown("Enter credentials to access the dashboard")
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
# Logout
# =============================================================================
col1, col2 = st.columns([6, 1])
with col2:
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

# =============================================================================
# Global Constants (2 Groups Only)
# =============================================================================
FEATURES = [
    'Age', 'Income', 'Has_Children', 'TotalMnt', 'TotalPurchases',
    'NumWebVisitsMonth', 'TotalAcceptedCmp', 'Days_Customer'
]

# For unsupervised clusters (2 groups)
CLUSTER_DESCRIPTIONS = {
    0: "Value-Conscious Shoppers: Lower income, price-sensitive",
    1: "Premium Loyalists: Higher income, brand-loyal"
}

CLUSTER_MARKETING = {
    0: "• Offer budget bundles\n• Highlight discounts\n• Use value-based messaging",
    1: "• Recommend premium products\n• Offer loyalty rewards\n• Provide personalized service"
}

# For supervised segments (2 groups)
SEGMENT_DESCRIPTIONS = {
    0: "Budget Segment: Spending below median",
    1: "Premium Segment: Spending above median"
}

SEGMENT_MARKETING = {
    0: "• Promote entry-level products\n• Use price incentives\n• Focus on affordability",
    1: "• Upsell premium items\n• Offer exclusive access\n• Emphasize quality"
}

# =============================================================================
# Preprocessing Function
# =============================================================================
def preprocess_for_eda(df_raw):
    df = df_raw.copy()
    df = df.dropna(subset=['Income'])
    df["Age"] = 2024 - df["Year_Birth"]
    df = df[df["Age"] <= 100].copy()
    df["Marital_Status"] = df["Marital_Status"].replace(
        {"Absurd": "Others", "YOLO": "Others", "Alone": "Others"}
    )
    df["Education"] = df["Education"].replace({
        "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
        "Graduation": "Graduate", "Master": "Postgraduate", "PhD": "Postgraduate"
    })
    df["Has_Children"] = ((df["Kidhome"] + df["Teenhome"]) > 0).astype(int)
    df["TotalMnt"] = df[['MntWines','MntFruits','MntMeatProducts',
                        'MntFishProducts','MntSweetProducts','MntGoldProds']].sum(axis=1)
    df["TotalPurchases"] = df[['NumWebPurchases','NumCatalogPurchases','NumStorePurchases']].sum(axis=1)
    df["TotalAcceptedCmp"] = df[['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3',
                                'AcceptedCmp4','AcceptedCmp5','Response']].sum(axis=1)
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format='%d-%m-%Y', errors='coerce')
    df = df.dropna(subset=['Dt_Customer'])
    df["Days_Customer"] = (pd.to_datetime("2024-01-01") - df["Dt_Customer"]).dt.days
    return df

# =============================================================================
# Navigation
# =============================================================================
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Customer Dashboard", "Predict Customer Segment"])

# =============================================================================
# HOME PAGE
# =============================================================================
if page == "Home":
    st.title("🏠 Welcome to the Customer Segmentation System")
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
        st.dataframe(df_home[FEATURES].head(3))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Income", f"${df_home['Income'].mean():,.0f}")
        with col2:
            st.metric("Avg Age", f"{df_home['Age'].mean():.1f}")
        with col3:
            st.metric("Avg Spending", f"${df_home['TotalMnt'].mean():,.0f}")
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
# CUSTOMER DASHBOARD
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
    education_options = ["All"] + sorted(df_full['Education'].unique().tolist())
    selected_education = st.sidebar.selectbox("Education Level", education_options)
    marital_options = ["All"] + sorted(df_full['Marital_Status'].unique().tolist())
    selected_marital = st.sidebar.selectbox("Marital Status", marital_options)
    children_options = ["All", "Yes", "No"]
    selected_children = st.sidebar.selectbox("Has Children", children_options)
    
    # Apply filters
    filtered_df = df_full.copy()
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
    avg_spent = filtered_df['TotalMnt'].mean()
    avg_age = filtered_df['Age'].mean()
    avg_income = filtered_df['Income'].mean()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💰 Avg. Spending", f"${avg_spent:,.0f}")
    with col2:
        st.metric("👤 Avg. Age", f"{avg_age:.1f}")
    with col3:
        st.metric("💳 Avg. Income", f"${avg_income:,.0f}")
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["EDA", "Clustering Results", "Marketing Suggestions"])
    
    # Tab 1: EDA
    with tab1:
        st.markdown("## 📊 Exploratory Data Analysis")
        PRIMARY_COLOR = "#007BFF"
        BACKGROUND_COLOR = "#F0F2F6"
        
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(facecolor=BACKGROUND_COLOR)
            ax.hist(filtered_df['Age'], bins=20, color=PRIMARY_COLOR, edgecolor='white', alpha=0.8)
            ax.set_title('Age Distribution')
            ax.set_facecolor(BACKGROUND_COLOR)
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(facecolor=BACKGROUND_COLOR)
            ax.hist(filtered_df['Income'], bins=30, color=PRIMARY_COLOR, edgecolor='white', alpha=0.8)
            ax.set_title('Income Distribution')
            ax.set_facecolor(BACKGROUND_COLOR)
            st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        with col1:
            spending_cols = ['MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds']
            spending_means = filtered_df[spending_cols].mean()
            colors = plt.cm.Blues(np.linspace(0.4, 1.0, len(spending_means)))
            fig, ax = plt.subplots(facecolor=BACKGROUND_COLOR)
            ax.bar(spending_means.index, spending_means.values, color=colors, edgecolor='white')
            ax.set_title('Spending by Category')
            ax.set_facecolor(BACKGROUND_COLOR)
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
        with col2:
            campaign_cols = ['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','Response']
            acceptance_rates = filtered_df[campaign_cols].mean() * 100
            colors = plt.cm.Purples(np.linspace(0.4, 1.0, len(acceptance_rates)))
            fig, ax = plt.subplots(facecolor=BACKGROUND_COLOR)
            ax.bar(acceptance_rates.index, acceptance_rates.values, color=colors, edgecolor='white')
            ax.set_title('Campaign Acceptance')
            ax.set_facecolor(BACKGROUND_COLOR)
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
    
    # Tab 2: Clustering Results
    with tab2:
        scaler, pca, clusterer, cluster_profiles, supervised_model = load_models()
        if cluster_profiles is not None and len(cluster_profiles) == 2:
            st.markdown("## 🧩 Clustering Results (2 Segments)")
            
            cols = st.columns(2)
            for cluster_id in [0, 1]:
                with cols[cluster_id]:
                    st.markdown(f"### Cluster {cluster_id}")
                    st.info(CLUSTER_DESCRIPTIONS[cluster_id])
                    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
                    categories = list(cluster_profiles.columns)
                    N = len(categories)
                    angles = [n / float(N) * 2 * pi for n in range(N)]
                    angles += angles[:1]
                    values = cluster_profiles.loc[cluster_id].values.flatten().tolist()
                    values += values[:1]
                    ax.plot(angles, values, linewidth=2, linestyle='solid')
                    ax.fill(angles, values, alpha=0.25)
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(categories, size=8)
                    ax.set_ylim(0, 1)
                    st.pyplot(fig)
            
            st.markdown("### 🌡️ Feature Heatmap")
            fig, ax = plt.subplots(figsize=(10, 3))
            sns.heatmap(cluster_profiles, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
            st.pyplot(fig)
        else:
            st.error("Clustering model not found or not 2 clusters.")
    
    # Tab 3: Marketing Suggestions
    with tab3:
        st.markdown("## 💡 Marketing Strategies")
        for cid in [0, 1]:
            st.markdown(f"### {'Budget' if cid == 0 else 'Premium'} Segment")
            st.markdown(f"**Profile**: {CLUSTER_DESCRIPTIONS[cid]}")
            st.markdown(f"**Actions**:\n{CLUSTER_MARKETING[cid]}")
            st.markdown("---")

# =============================================================================
# PREDICT CUSTOMER SEGMENT
# =============================================================================
elif page == "Predict Customer Segment":
    st.title("🔮 Predict Customer Segment")
    scaler, pca, clusterer, cluster_profiles, supervised_model = load_models()
    if scaler is None:
        st.error("Models not loaded.")
        st.stop()
    
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
            total_cmp = st.number_input("Accepted Campaigns (0-6)", min_value=0, max_value=6, value=1)
            days_customer = st.number_input("Days as Customer", min_value=0, value=365)
        submitted = st.form_submit_button("🚀 Predict Segment")
    
    if submitted:
        input_data = np.array([[age, income, has_children, total_mnt, total_purchases,
                                web_visits, total_cmp, days_customer]])
        try:
            scaled = scaler.transform(input_data)
            
            if supervised_model is not None:
                # Use Logistic Regression (2 segments)
                pred_label = int(supervised_model.predict(scaled)[0])
                segment_name = "Budget" if pred_label == 0 else "Premium"
                st.success(f"### 🎯 Predicted Segment: **{segment_name}**")
                st.info(f"**Profile**: {SEGMENT_DESCRIPTIONS[pred_label]}")
                st.markdown("### 💼 Recommended Strategy")
                st.markdown(SEGMENT_MARKETING[pred_label])
            else:
                # Fallback to clustering
                reduced = pca.transform(scaled)
                cluster_id = int(clusterer.predict(reduced)[0])
                st.success(f"### 🎯 Predicted Cluster: **Cluster {cluster_id}**")
                st.info(f"**Profile**: {CLUSTER_DESCRIPTIONS[cluster_id]}")
                st.markdown("### 💼 Recommended Strategy")
                st.markdown(CLUSTER_MARKETING[cluster_id])
                
        except Exception as e:
            st.error(f"Prediction error: {e}")

# =============================================================================
# Footer
# =============================================================================
st.markdown("---")
st.caption("💡 Powered by binary customer segmentation using Logistic Regression and KMeans.")