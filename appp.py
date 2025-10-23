# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from math import pi
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# Configuration & Constants (Updated with Brand Colors)
# =============================================================================
CONFIG = {
    "features": [
        'Age', 'Income', 'Has_Children', 'TotalMnt', 'TotalPurchases',
        'NumWebVisitsMonth', 'TotalAcceptedCmp', 'Days_Customer'
    ],
    "cluster_descriptions": {
        0: "Value-Conscious Shoppers: Lower income, price-sensitive",
        1: "Premium Loyalists: Higher income, brand-loyal"
    },
    "cluster_marketing": {
        0: "• Offer budget bundles\n• Highlight discounts\n• Use value-based messaging",
        1: "• Recommend premium products\n• Offer loyalty rewards\n• Provide personalized service"
    },
    "segment_descriptions": {
        0: "Budget Segment: Spending below median",
        1: "Premium Segment: Spending above median"
    },
    "segment_marketing": {
        0: "• Promote entry-level products\n• Use price incentives\n• Focus on affordability",
        1: "• Upsell premium items\n• Offer exclusive access\n• Emphasize quality"
    },
    "colors": {
        "background": "#FAFAFA",       # Off-white
        "text_primary": "#333333",     # Dark Gray
        "accent": "#FF9800",           # Tangerine Orange
        "accent_hover": "#E65100",     # Burnt Orange
        "card_bg": "#FFF3E0",          # Light Cream
        "border": "#E0E0E0"            # Soft Gray
    }
}

FEATURES = CONFIG["features"]
COLORS = CONFIG["colors"]

# Set page config
st.set_page_config(
    page_title="Customer Segmentation System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
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
# Preprocessing Function
# =============================================================================
def preprocess_for_eda(df_raw):
    """
    Clean and engineer features from raw marketing campaign data.
    Returns a DataFrame ready for EDA or modeling.
    """
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
    
    # Age Group Filter
    age_bins = [18, 30, 45, 60, 100]
    age_labels = ["18-29", "30-44", "45-59", "60+"]
    df_full['AgeGroup'] = pd.cut(df_full['Age'], bins=age_bins, labels=age_labels, right=False)
    
    selected_age_group = st.sidebar.selectbox("Age Group", ["All"] + list(age_labels))
    
    # Income Range Filter
    min_income = int(df_full['Income'].min())
    max_income = int(df_full['Income'].max())
    income_range = st.sidebar.slider(
        "Income Range ($)",
        min_value=min_income,
        max_value=max_income,
        value=(min_income, max_income),
        step=5000
    )
    
    # Total Spending Filter
    min_spent = int(df_full['TotalMnt'].min())
    max_spent = int(df_full['TotalMnt'].max())
    spending_range = st.sidebar.slider(
        "Total Spending Range ($)",
        min_value=min_spent,
        max_value=max_spent,
        value=(min_spent, max_spent),
        step=100
    )
    
    # Other Filters
    education_options = ["All"] + sorted(df_full['Education'].unique().tolist())
    selected_education = st.sidebar.selectbox("Education Level", education_options)
    marital_options = ["All"] + sorted(df_full['Marital_Status'].unique().tolist())
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
    ]
    filtered_df = filtered_df[
        (filtered_df['TotalMnt'] >= spending_range[0]) &
        (filtered_df['TotalMnt'] <= spending_range[1])
    ]
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
    
    # KPIs with colored background
    avg_spent = filtered_df['TotalMnt'].mean()
    avg_age = filtered_df['Age'].mean()
    avg_income = filtered_df['Income'].mean()
    
    col1, col2, col3 = st.columns(3)
    kpi_style = f"background-color: {COLORS['card_bg']}; padding: 15px; border-radius: 8px; border-left: 5px solid {COLORS['accent']};"
    
    with col1:
        st.markdown(f"""
        <div style="{kpi_style}">
            <h4>💰 Avg. Spending</h4>
            <p style="font-size: 20px; font-weight: bold;">${avg_spent:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="{kpi_style}">
            <h4>👤 Avg. Age</h4>
            <p style="font-size: 20px; font-weight: bold;">{avg_age:.1f}</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div style="{kpi_style}">
            <h4>💳 Avg. Income</h4>
            <p style="font-size: 20px; font-weight: bold;">${avg_income:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Demographics",
        "Purchase Behavior",
        "Campaign Response",
        "Clustering & Marketing"
    ])
    
    # Tab 1: Demographics
    with tab1:
        st.markdown("## 👥 Demographics")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6,4), facecolor=COLORS["background"])
            ax.hist(filtered_df['Age'], bins=20, color=COLORS["accent"], edgecolor=COLORS["border"], alpha=0.8)
            ax.set_title('Age Distribution', color=COLORS["text_primary"])
            ax.set_facecolor(COLORS["background"])
            ax.grid(True, color=COLORS["border"])
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(6,4), facecolor=COLORS["background"])
            ax.hist(filtered_df['Income'], bins=30, color=COLORS["accent"], edgecolor=COLORS["border"], alpha=0.8)
            ax.set_title('Income Distribution', color=COLORS["text_primary"])
            ax.set_facecolor(COLORS["background"])
            ax.grid(True, color=COLORS["border"])
            st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        with col1:
            edu_counts = filtered_df['Education'].value_counts().sort_index()
            colors = plt.cm.Oranges(np.linspace(0.4, 1.0, len(edu_counts)))
            fig, ax = plt.subplots(figsize=(6,4), facecolor=COLORS["background"])
            ax.bar(edu_counts.index, edu_counts.values, color=colors, edgecolor=COLORS["border"])
            ax.set_title('Education Level', color=COLORS["text_primary"])
            ax.set_facecolor(COLORS["background"])
            ax.grid(True, color=COLORS["border"])
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
        with col2:
            mar_counts = filtered_df['Marital_Status'].value_counts().sort_index()
            colors = plt.cm.Greens(np.linspace(0.4, 1.0, len(mar_counts)))
            fig, ax = plt.subplots(figsize=(6,4), facecolor=COLORS["background"])
            ax.bar(mar_counts.index, mar_counts.values, color=colors, edgecolor=COLORS["border"])
            ax.set_title('Marital Status', color=COLORS["text_primary"])
            ax.set_facecolor(COLORS["background"])
            ax.grid(True, color=COLORS["border"])
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
    
    # Tab 2: Purchase Behavior
    with tab2:
        st.markdown("## 🛒 Purchase Behavior")
        col1, col2 = st.columns(2)
        with col1:
            spending_cols = ['MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds']
            spending_means = filtered_df[spending_cols].mean()
            colors = plt.cm.Blues(np.linspace(0.4, 1.0, len(spending_means)))
            fig, ax = plt.subplots(figsize=(6,4), facecolor=COLORS["background"])
            ax.bar(spending_means.index, spending_means.values, color=colors, edgecolor=COLORS["border"])
            ax.set_title('Spending by Category', color=COLORS["text_primary"])
            ax.set_facecolor(COLORS["background"])
            ax.grid(True, color=COLORS["border"])
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
        with col2:
            purchase_types = ['NumWebPurchases','NumCatalogPurchases','NumStorePurchases']
            purchase_means = filtered_df[purchase_types].mean()
            colors = plt.cm.Purples(np.linspace(0.4, 1.0, len(purchase_means)))
            fig, ax = plt.subplots(figsize=(6,4), facecolor=COLORS["background"])
            ax.bar(purchase_types, purchase_means.values, color=colors, edgecolor=COLORS["border"])
            ax.set_title('Purchase Channel Preference', color=COLORS["text_primary"])
            ax.set_facecolor(COLORS["background"])
            ax.grid(True, color=COLORS["border"])
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6,4), facecolor=COLORS["background"])
            ax.scatter(filtered_df['Income'], filtered_df['TotalMnt'], alpha=0.6, color=COLORS["accent"])
            ax.set_title('Income vs Total Spending', color=COLORS["text_primary"])
            ax.set_xlabel('Income ($)', color=COLORS["text_primary"])
            ax.set_ylabel('Total Spending ($)', color=COLORS["text_primary"])
            ax.set_facecolor(COLORS["background"])
            ax.grid(True, color=COLORS["border"])
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(6,4), facecolor=COLORS["background"])
            ax.hist(filtered_df['TotalPurchases'], bins=20, color=COLORS["accent"], edgecolor=COLORS["border"], alpha=0.8)
            ax.set_title('Total Purchases Distribution', color=COLORS["text_primary"])
            ax.set_facecolor(COLORS["background"])
            ax.grid(True, color=COLORS["border"])
            st.pyplot(fig)
    
    # Tab 3: Campaign Response
    with tab3:
        st.markdown("## 📣 Campaign Response")
        campaign_cols = ['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','Response']
        acceptance_rates = filtered_df[campaign_cols].mean() * 100
        
        fig, ax = plt.subplots(figsize=(8,4), facecolor=COLORS["background"])
        colors = plt.cm.Reds(np.linspace(0.4, 1.0, len(acceptance_rates)))
        bars = ax.bar(acceptance_rates.index, acceptance_rates.values, color=colors, edgecolor=COLORS["border"])
        ax.set_title('Campaign Acceptance Rates (%)', color=COLORS["text_primary"])
        ax.set_facecolor(COLORS["background"])
        ax.grid(True, color=COLORS["border"])
        plt.xticks(rotation=45, ha='right')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 1, f'{height:.1f}%', 
                    ha='center', va='bottom', fontsize=9, color=COLORS["text_primary"])
        st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6,4), facecolor=COLORS["background"])
            ax.scatter(filtered_df['TotalAcceptedCmp'], filtered_df['TotalMnt'], alpha=0.6, color=COLORS["accent"])
            ax.set_title('Campaigns Accepted vs Spending', color=COLORS["text_primary"])
            ax.set_xlabel('Campaigns Accepted', color=COLORS["text_primary"])
            ax.set_ylabel('Total Spending ($)', color=COLORS["text_primary"])
            ax.set_facecolor(COLORS["background"])
            ax.grid(True, color=COLORS["border"])
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(6,4), facecolor=COLORS["background"])
            web_visits = filtered_df['NumWebVisitsMonth']
            purchases = filtered_df['TotalPurchases']
            scatter = ax.scatter(web_visits, purchases, c=web_visits, cmap='Oranges', alpha=0.7)
            ax.set_title('Web Visits vs Total Purchases', color=COLORS["text_primary"])
            ax.set_xlabel('Monthly Web Visits', color=COLORS["text_primary"])
            ax.set_ylabel('Total Purchases', color=COLORS["text_primary"])
            ax.set_facecolor(COLORS["background"])
            ax.grid(True, color=COLORS["border"])
            plt.colorbar(scatter, ax=ax, label='Web Visits')
            st.pyplot(fig)
    
    # Tab 4: Clustering & Marketing
    with tab4:
        scaler, pca, clusterer, cluster_profiles, supervised_model = load_models()
        if cluster_profiles is not None and len(cluster_profiles) == 2:
            st.markdown("## 🧩 Clustering Results (2 Segments)")
            
            cols = st.columns(2)
            for cluster_id in [0, 1]:
                with cols[cluster_id]:
                    st.markdown(f"### Cluster {cluster_id}")
                    st.info(CONFIG["cluster_descriptions"][cluster_id])
                    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
                    categories = list(cluster_profiles.columns)
                    N = len(categories)
                    angles = [n / float(N) * 2 * pi for n in range(N)]
                    angles += angles[:1]
                    values = cluster_profiles.loc[cluster_id].values.flatten().tolist()
                    values += values[:1]
                    ax.plot(angles, values, linewidth=2, linestyle='solid', color=COLORS["accent"])
                    ax.fill(angles, values, alpha=0.25, color=COLORS["accent"])
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(categories, size=8, color=COLORS["text_primary"])
                    ax.set_ylim(0, 1)
                    ax.set_facecolor(COLORS["background"])
                    st.pyplot(fig)
            
            st.markdown("### 🌡️ Feature Heatmap")
            fig, ax = plt.subplots(figsize=(10, 3), facecolor=COLORS["background"])
            sns.heatmap(cluster_profiles, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax, cbar_kws={'shrink': 0.8})
            ax.set_facecolor(COLORS["background"])
            st.pyplot(fig)
            
            st.markdown("## 💡 Marketing Strategies")
            for cid in [0, 1]:
                st.markdown(f"### {'Budget' if cid == 0 else 'Premium'} Segment")
                st.markdown(f"**Profile**: {CONFIG['cluster_descriptions'][cid]}")
                st.markdown(f"**Actions**:\n{CONFIG['cluster_marketing'][cid]}")
                st.markdown("---")
        else:
            st.error("Clustering model not found or not 2 clusters.")

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
                segment_name = "Budget" if pred_label == 0 else "Premium"
                
                # Show confidence if available
                confidence_msg = ""
                if hasattr(supervised_model, "predict_proba"):
                    proba = supervised_model.predict_proba(scaled)[0]
                    confidence = max(proba) * 100
                    confidence_msg = f" (Confidence: {confidence:.1f}%)"
                
                st.success(f"### 🎯 Predicted Segment: **{segment_name}**{confidence_msg}")
                st.info(f"**Profile**: {CONFIG['segment_descriptions'][pred_label]}")
                st.markdown("### 💼 Recommended Strategy")
                st.markdown(CONFIG['segment_marketing'][pred_label])
            else:
                reduced = pca.transform(scaled)
                cluster_id = int(clusterer.predict(reduced)[0])
                st.success(f"### 🎯 Predicted Cluster: **Cluster {cluster_id}**")
                st.info(f"**Profile**: {CONFIG['cluster_descriptions'][cluster_id]}")
                st.markdown("### 💼 Recommended Strategy")
                st.markdown(CONFIG['cluster_marketing'][cluster_id])
                
        except Exception as e:
            st.error(f"Prediction error: {e}")

# =============================================================================
# Footer
# =============================================================================
st.markdown("---")
st.caption("💡 Powered by binary customer segmentation using Logistic Regression and KMeans.")