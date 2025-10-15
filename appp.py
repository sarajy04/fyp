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
        # Try to load supervised classifier (Random Forest)
        try:
            rf_model = joblib.load('models/segment_classifier.pkl')
        except:
            rf_model = None
        return scaler, pca, clusterer, profiles, rf_model
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
# Global Constants
# =============================================================================
FEATURES = [
    'Age', 'Income', 'Has_Children', 'TotalMnt', 'TotalPurchases',
    'NumWebVisitsMonth', 'TotalAcceptedCmp', 'Days_Customer'
]

# For unsupervised clusters
CLUSTER_DESCRIPTIONS = {
    0: "Budget-Conscious Browsers: Low income, high web visits, low spending",
    1: "Loyal High-Spenders: High income, high spending, low discount usage",
    2: "Deal-Seeking Parents: Medium income, high campaign acceptance, with children",
    3: "At-Risk Inactives: Low engagement, declining purchases, high web visits"
}

CLUSTER_MARKETING = {
    0: "• Offer time-limited discounts\n• Promote via email/newsletter\n• Highlight value-for-money products",
    1: "• Upsell premium products\n• Personalize recommendations\n• Reward loyalty with exclusive perks",
    2: "• Bundle family-oriented products\n• Emphasize child-friendly features\n• Use social proof (parent testimonials)",
    3: "• Reactivation campaigns (special offers)\n• Survey to understand churn reasons\n• Re-engage with personalized content"
}

# For supervised segments
SEGMENT_DESCRIPTIONS = {
    'Low': "Budget-conscious shoppers with limited spending",
    'Medium': "Moderate spenders with balanced engagement",
    'High': "High-value customers with premium purchasing behavior"
}

SEGMENT_MARKETING = {
    'Low': "• Offer entry-level products\n• Use price-based promotions\n• Build trust through education",
    'Medium': "• Cross-sell complementary items\n• Introduce loyalty programs\n• Share user testimonials",
    'High': "• Provide VIP treatment\n• Offer early access to new products\n• Personalize high-touch service"
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
if page == "Home":
    st.title("🏠 Welcome to the Customer Segmentation System")
    st.markdown("""
    ## 🎯 Introduction
    Helping uncover hidden patterns of customer behavior in grocery retail using machine learning techniques. By analyzing relevant factors like income, spending habits, family size, and recency of purchases, the app groups customers into meaningful segments. These clusters make it easier to understand who your customers really are — from loyal big spenders to budget-conscious shoppers. The app transforms raw marketing data into clear insights. The goal? To help businesses make smarter, more personalized decisions and connect with their customers on a deeper level.
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
    st.subheader("💡 Why Segment Customers?")
    st.markdown("""
    - Improve **decision making**
    - Identify **high-value customers**
    - Deliver **personalized customer experiences**
    - Increase **marketing ROI**
    - Reduce **customer churn rate**
    - Guide **product development**
    """)

# =============================================================================
# CUSTOMER DASHBOARD
elif page == "Customer Dashboard":
    st.title("📈 Customer Dashboard")
    
    df_raw = load_raw_data()
    if df_raw is not None:
        df_eda = preprocess_for_eda(df_raw)
        
        avg_spent = df_eda['TotalMnt'].mean()
        avg_age = df_eda['Age'].mean()
        avg_income = df_eda['Income'].mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💰 Avg. Spending", f"${avg_spent:,.0f}")
        with col2:
            st.metric("👤 Avg. Age", f"{avg_age:.1f}")
        with col3:
            st.metric("💳 Avg. Income", f"${avg_income:,.0f}")
        
        st.markdown("---")
    else:
        st.error("Dataset not found. Place `marketing_campaign.csv` in the project root.")
        st.stop()
    
    tab1, tab2, tab3 = st.tabs(["EDA: Demographics & Behavior", "Clustering Results", "Marketing Suggestions"])
    
    # Tab 1: EDA
    with tab1:
        if df_raw is not None:
            st.markdown("## 📊 Exploratory Data Analysis")
            
            PRIMARY_COLOR = "#007BFF"
            BACKGROUND_COLOR = "#F0F2F6"
            
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(facecolor=BACKGROUND_COLOR)
                ax.hist(df_eda['Age'], bins=20, color=PRIMARY_COLOR, edgecolor='white', alpha=0.8)
                ax.set_title('Age Distribution', fontsize=14, fontweight='bold')
                ax.set_facecolor(BACKGROUND_COLOR)
                st.pyplot(fig)
            with col2:
                fig, ax = plt.subplots(facecolor=BACKGROUND_COLOR)
                ax.hist(df_eda['Income'], bins=30, color=PRIMARY_COLOR, edgecolor='white', alpha=0.8)
                ax.set_title('Income Distribution', fontsize=14, fontweight='bold')
                ax.set_facecolor(BACKGROUND_COLOR)
                st.pyplot(fig)
            
            col1, col2 = st.columns(2)
            with col1:
                spending_cols = ['MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds']
                spending_means = df_eda[spending_cols].mean()
                colors = plt.cm.Blues(np.linspace(0.4, 1.0, len(spending_means)))
                fig, ax = plt.subplots(facecolor=BACKGROUND_COLOR)
                ax.bar(spending_means.index, spending_means.values, color=colors, edgecolor='white')
                ax.set_ylabel('Average Spending ($)', fontweight='bold')
                ax.set_title('Spending by Category', fontsize=14, fontweight='bold')
                ax.set_facecolor(BACKGROUND_COLOR)
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)
            with col2:
                campaign_cols = ['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','Response']
                acceptance_rates = df_eda[campaign_cols].mean() * 100
                colors = plt.cm.Purples(np.linspace(0.4, 1.0, len(acceptance_rates)))
                fig, ax = plt.subplots(facecolor=BACKGROUND_COLOR)
                ax.bar(acceptance_rates.index, acceptance_rates.values, color=colors, edgecolor='white')
                ax.set_ylabel('Acceptance Rate (%)', fontweight='bold')
                ax.set_title('Campaign Acceptance', fontsize=14, fontweight='bold')
                ax.set_facecolor(BACKGROUND_COLOR)
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)
            
            col1, col2 = st.columns(2)
            with col1:
                edu_counts = df_eda['Education'].value_counts()
                colors = plt.cm.Blues(np.linspace(0.4, 1.0, len(edu_counts)))
                fig, ax = plt.subplots(facecolor=BACKGROUND_COLOR)
                ax.pie(edu_counts.values, labels=edu_counts.index, autopct='%1.1f%%', colors=colors)
                ax.set_title('Education Levels', fontsize=14, fontweight='bold')
                st.pyplot(fig)
            with col2:
                marital_counts = df_eda['Marital_Status'].value_counts()
                colors = plt.cm.Greens(np.linspace(0.4, 1.0, len(marital_counts)))
                fig, ax = plt.subplots(facecolor=BACKGROUND_COLOR)
                ax.bar(marital_counts.index, marital_counts.values, color=colors, edgecolor='white')
                ax.set_title('Marital Status', fontsize=14, fontweight='bold')
                ax.set_facecolor(BACKGROUND_COLOR)
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)
        else:
            st.error("Dataset not found.")
    
    # Tab 2: Clustering Results
    with tab2:
        scaler, pca, clusterer, cluster_profiles, rf_model = load_models()
        if cluster_profiles is not None:
            st.markdown(f"## 🧩 Clustering Results ({len(cluster_profiles)} Segments)")
            
            st.markdown("### 📡 Cluster Profiles")
            n_clusters = len(cluster_profiles)
            cols = st.columns(min(4, n_clusters))
            for idx, cluster_id in enumerate(cluster_profiles.index):
                with cols[idx % len(cols)]:
                    st.markdown(f"### Cluster {cluster_id}")
                    st.info(CLUSTER_DESCRIPTIONS.get(cluster_id, "N/A"))
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
            
            st.markdown("### 🌡️ Feature Heatmap (Normalized)")
            fig, ax = plt.subplots(figsize=(10, max(4, n_clusters)))
            sns.heatmap(cluster_profiles, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
            ax.set_ylabel("Cluster")
            ax.set_xlabel("Features (Normalized)")
            st.pyplot(fig)
        else:
            st.error("Models not found. Run training pipeline first.")
    
    # Tab 3: Marketing Suggestions
    with tab3:
        st.markdown("## 💡 Marketing Strategies by Segment")
        for cid in sorted(CLUSTER_DESCRIPTIONS.keys()):
            st.markdown(f"### Cluster {cid}")
            st.markdown(f"**Profile**: {CLUSTER_DESCRIPTIONS[cid]}")
            st.markdown(f"**Actions**:\n{CLUSTER_MARKETING[cid]}")
            st.markdown("---")

# =============================================================================
# PREDICT TAB (Now uses Random Forest if available)
# =============================================================================
elif page == "Predict Customer Segment":
    st.title("🔮 Predict Customer Segment")
    scaler, pca, clusterer, cluster_profiles, rf_model = load_models()
    if scaler is None:
        st.error("Models not loaded. Train and save models first.")
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
            
            if rf_model is not None:
                # ✅ Use Random Forest (supervised)
                pred_label = rf_model.predict(scaled)[0]
                label_map_rev = {0: 'Low', 1: 'Medium', 2: 'High'}
                segment_name = label_map_rev[pred_label]
                st.success(f"### 🎯 Predicted Segment: **{segment_name}**")
                st.info(f"**Profile**: {SEGMENT_DESCRIPTIONS.get(segment_name, 'N/A')}")
                st.markdown("### 💼 Recommended Strategy")
                st.markdown(SEGMENT_MARKETING.get(segment_name, "No strategy available"))
            else:
                # ❌ Fall back to KMeans (unsupervised)
                reduced = pca.transform(scaled)
                cluster_id = int(clusterer.predict(reduced)[0])
                st.success(f"### 🎯 Predicted Cluster: **Cluster {cluster_id}**")
                st.info(f"**Profile**: {CLUSTER_DESCRIPTIONS.get(cluster_id, 'Unknown')}")
                st.markdown("### 💼 Recommended Strategy")
                st.markdown(CLUSTER_MARKETING.get(cluster_id, "No strategy available"))
                
        except Exception as e:
            st.error(f"Prediction error: {e}")

# =============================================================================
# Footer
st.markdown("---")
st.caption("💡 Powered by machine learning on the Customer Personality Analysis dataset.")