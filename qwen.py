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

USERNAME = "admin"
PASSWORD = "segment2024"

def authenticate(username, password):
    return username == USERNAME and password == PASSWORD

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# =============================================================================
# Load Data for EDA (if available)
# =============================================================================
@st.cache_data
def load_eda_data():
    """Load raw dataset for EDA (if exists)"""
    try:
        # Try common locations
        for path in ['marketing_campaign.csv', 'data/marketing_campaign.csv']:
            if os.path.exists(path):
                df = pd.read_csv(path, sep='\t')
                return df
        return None
    except Exception as e:
        st.warning(f"Could not load raw data for EDA: {e}")
        return None

# =============================================================================
# Load Pre-trained Models & Assets
# =============================================================================
@st.cache_resource
def load_models():
    if not os.path.exists('models'):
        return None, None, None, None
    
    try:
        scaler = joblib.load('models/scaler.pkl')
        pca = joblib.load('models/pca.pkl')
        clusterer = joblib.load('models/clusterer.pkl')
        profiles = pd.read_csv('models/cluster_profiles.csv', index_col=0)
        return scaler, pca, clusterer, profiles
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

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
            if authenticate(username, password):
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
    st.stop()

# =============================================================================
# Logout Button (Top Right)
# =============================================================================
col1, col2 = st.columns([6, 1])
with col2:
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()

# =============================================================================
# Main App (Only visible when logged in)
# =============================================================================
scaler, pca, clusterer, cluster_profiles = load_models()
df_raw = load_eda_data()

FEATURES = [
    'Age', 'Income', 'Has_Children', 'TotalMnt', 'TotalPurchases',
    'NumWebVisitsMonth', 'TotalAcceptedCmp', 'Days_Customer'
]

CLUSTER_DESCRIPTIONS = {
    0: "Budget-Conscious Browsers: Low income, high web visits, low spending",
    1: "Loyal High-Spenders: High income, high spending, low discount usage",
    2: "Deal-Seeking Parents: Medium income, high campaign acceptance, with children",
    3: "At-Risk Inactives: Low engagement, declining purchases, high web visits"
}

st.title("🛒 Customer Segmentation System")
st.markdown("""
This system segments retail customers into behavioral groups using unsupervised machine learning.  
Enter customer details below to predict their segment and view insights.
""")

# =============================================================================
# Sidebar: Customer Input Form
# =============================================================================
st.sidebar.header("👤 Customer Details")
with st.sidebar.form("customer_input"):
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
    has_children = st.selectbox("Has Children?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    total_mnt = st.number_input("Total Spending ($)", min_value=0, value=600, step=50)
    total_purchases = st.number_input("Total Purchases", min_value=0, value=20)
    web_visits = st.number_input("Monthly Web Visits", min_value=0, value=5)
    total_cmp = st.number_input("Accepted Campaigns (0-6)", min_value=0, max_value=6, value=1)
    days_customer = st.number_input("Days as Customer", min_value=0, value=365)
    
    submitted = st.form_submit_button("🔍 Predict Segment")

# =============================================================================
# Prediction Logic
# =============================================================================
if submitted:
    if scaler is None:
        st.error("Models not loaded. Please run training pipeline first.")
        st.stop()
    
    input_data = np.array([[age, income, has_children, total_mnt, total_purchases,
                            web_visits, total_cmp, days_customer]])
    
    try:
        scaled = scaler.transform(input_data)
        reduced = pca.transform(scaled)
        cluster_id = clusterer.predict(reduced)[0]
        
        st.success(f"### 🎯 Predicted Segment: **Cluster {cluster_id}**")
        st.info(f"**Profile**: {CLUSTER_DESCRIPTIONS.get(cluster_id, 'Unknown behavior pattern')}")
        
        # Comparison table
        customer_norm = pd.DataFrame(scaled, columns=FEATURES)
        cluster_norm = cluster_profiles.loc[cluster_id:cluster_id]
        comparison_df = pd.DataFrame({
            'Customer': customer_norm.iloc[0].values,
            'Cluster Average': cluster_norm.iloc[0].values
        }, index=FEATURES)
        st.dataframe(comparison_df.style.format("{:.2f}"))
        
    except Exception as e:
        st.error(f"Prediction error: {e}")

# =============================================================================
# EDA Section (Only if raw data available)
# =============================================================================
if df_raw is not None:
    st.markdown("---")
    st.subheader("📊 Exploratory Data Analysis")
    
    # Clean data for EDA
    df_eda = df_raw.copy()
    df_eda = df_eda.dropna(subset=['Income'])
    df_eda['Age'] = 2024 - df_eda['Year_Birth']
    df_eda = df_eda[df_eda['Age'] <= 100]
    df_eda['TotalMnt'] = df_eda[['MntWines','MntFruits','MntMeatProducts',
                                'MntFishProducts','MntSweetProducts','MntGoldProds']].sum(axis=1)
    df_eda['TotalPurchases'] = df_eda[['NumWebPurchases','NumCatalogPurchases','NumStorePurchases']].sum(axis=1)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Spending", "Campaigns", "Demographics"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.hist(df_eda['Age'], bins=20, color='skyblue', edgecolor='black')
            ax.set_title('Age Distribution')
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            ax.hist(df_eda['Income'], bins=30, color='lightgreen', edgecolor='black')
            ax.set_title('Income Distribution')
            st.pyplot(fig)
    
    with tab2:
        st.markdown("#### Spending by Product Category")
        spending_cols = ['MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds']
        spending_means = df_eda[spending_cols].mean()
        fig, ax = plt.subplots()
        spending_means.plot(kind='bar', ax=ax, color='coral')
        ax.set_ylabel('Average Spending ($)')
        ax.set_title('Average Spending by Category')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with tab3:
        st.markdown("#### Campaign Acceptance Rates")
        campaign_cols = ['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','Response']
        acceptance_rates = df_eda[campaign_cols].mean() * 100
        fig, ax = plt.subplots()
        acceptance_rates.plot(kind='bar', ax=ax, color='purple')
        ax.set_ylabel('Acceptance Rate (%)')
        ax.set_title('Campaign Acceptance Rates')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            edu_counts = df_eda['Education'].value_counts()
            fig, ax = plt.subplots()
            ax.pie(edu_counts.values, labels=edu_counts.index, autopct='%1.1f%%')
            ax.set_title('Education Levels')
            st.pyplot(fig)
        with col2:
            marital_counts = df_eda['Marital_Status'].value_counts()
            fig, ax = plt.subplots()
            ax.bar(marital_counts.index, marital_counts.values, color='gold')
            ax.set_title('Marital Status')
            plt.xticks(rotation=45)
            st.pyplot(fig)

# =============================================================================
# Visualization Tabs (Cluster Insights)
# =============================================================================
st.markdown("---")
st.subheader("📈 Cluster Insights")

if cluster_profiles is not None:
    tab1, tab2, tab3 = st.tabs(["Radar Charts", "Feature Heatmap", "Cluster Summary"])
    
    with tab1:
        st.markdown("#### Behavioral Profiles by Cluster")
        cols = st.columns(len(cluster_profiles))
        for idx, cluster_id in enumerate(cluster_profiles.index):
            with cols[idx]:
                st.markdown(f"**Cluster {cluster_id}**")
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
    
    with tab2:
        st.markdown("#### Cluster Feature Heatmap (Normalized)")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(cluster_profiles, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
        plt.ylabel("Cluster")
        plt.xlabel("Features (Normalized)")
        st.pyplot(fig)
    
    with tab3:
        st.markdown("#### Cluster Characteristics")
        summary_data = []
        for cid in cluster_profiles.index:
            desc = CLUSTER_DESCRIPTIONS.get(cid, "N/A")
            summary_data.append({"Cluster": cid, "Profile": desc})
        summary_df = pd.DataFrame(summary_data)
        st.table(summary_df.set_index("Cluster"))
else:
    st.info("Cluster profiles will appear here after models are loaded.")

# =============================================================================
# Footer
# =============================================================================
st.markdown("---")
st.caption("💡 **Note**: This system uses unsupervised clustering trained on the Customer Personality Analysis dataset. "
           "Clusters represent behavioral patterns—not predefined business segments.")