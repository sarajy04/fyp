# app.py
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

# =============================================================================
# Configuration & Constants
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
        "background": "#FFF3E0",      
        "text_primary": "#333333",
        "accent": "#4260C1",
        "accent_hover": "#272278",
        "card_bg": "#F6DEBA",          
        "border": "#E0E0E0"
    }
}

FEATURES = CONFIG["features"]
COLORS = CONFIG["colors"]

# Set page config with new background

st.set_page_config(
    page_title="Customer Segmentation System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(f"""
<style>
    .stApp {{
        background-color: {COLORS['background']} !important;
    }}
    h1, h2, h3, h4, h5, h6, p, div {{
        color: {COLORS['text_primary']} !important;
    }}
</style>
""", unsafe_allow_html=True)

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

@st.cache_data
def load_model_metrics():
    try:
        with open('models/model_metrics.json', 'r') as f:
            return json.load(f)
    except:
        return None

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
    # Banner
    st.markdown(f"""
    <div style="background-color: {COLORS['accent']}; padding: 30px; border-radius: 10px; text-align: center; margin-bottom: 30px;">
        <h1 style="color: white; font-size: 2.5rem;">Discover What Drives Your Shoppers</h1>
        <p style="color: white; font-size: 1.2rem; margin-top: 10px;">
            Unlock actionable insights with AI-powered customer segmentation
        </p>
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
        st.dataframe(df_home[FEATURES].head(3))
        
        # KPI Cards with colored background (same as dashboard)
        avg_spent = df_home['TotalMnt'].mean()
        avg_age = df_home['Age'].mean()
        avg_income = df_home['Income'].mean()
        
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
    
    age_bins = [18, 30, 45, 60, 100]
    age_labels = ["18-29", "30-44", "45-59", "60+"]
    df_full['AgeGroup'] = pd.cut(df_full['Age'], bins=age_bins, labels=age_labels, right=False)
    
    selected_age_group = st.sidebar.selectbox("Age Group", ["All"] + list(age_labels))
    
    min_income = int(df_full['Income'].min())
    max_income = int(df_full['Income'].max())
    income_range = st.sidebar.slider(
        "Income Range ($)",
        min_value=min_income,
        max_value=max_income,
        value=(min_income, max_income),
        step=5000
    )
    
    min_spent = int(df_full['TotalMnt'].min())
    max_spent = int(df_full['TotalMnt'].max())
    spending_range = st.sidebar.slider(
        "Total Spending Range ($)",
        min_value=min_spent,
        max_value=max_spent,
        value=(min_spent, max_spent),
        step=100
    )
    
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
        # Histogram
            hist_data = filtered_df['Age']
            kde = gaussian_kde(hist_data)
            x_vals = np.linspace(hist_data.min(), hist_data.max(), 200)
            y_kde = kde(x_vals) * len(hist_data) * (hist_data.max() - hist_data.min()) / 20  # Scale to histogram

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
                plot_bgcolor=COLORS["background"],
                paper_bgcolor=COLORS["background"],
                font_color=COLORS["text_primary"]
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            hist_data = filtered_df['Income']
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
                plot_bgcolor=COLORS["background"],
                paper_bgcolor=COLORS["background"],
                font_color=COLORS["text_primary"]
            )
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        
        with col1:
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
                plot_bgcolor=COLORS["background"],
                paper_bgcolor=COLORS["background"],
                font_color=COLORS["text_primary"],
                xaxis_title="Education Level",
                yaxis_title="Customer Count"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
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
                plot_bgcolor=COLORS["background"],
                paper_bgcolor=COLORS["background"],
                font_color=COLORS["text_primary"],
                xaxis_title="Marital Status",
                yaxis_title="Customer Count"
            )
            st.plotly_chart(fig, use_container_width=True)

    # Tab 2: Purchase Behavior
    with tab2:
        st.markdown("## 🛒 Purchase Behavior")

        col1, col2 = st.columns(2)
        
        with col1:
            spending_cols = ['MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds']
            spending_means = filtered_df[spending_cols].mean().reset_index()
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
                plot_bgcolor=COLORS["background"],
                paper_bgcolor=COLORS["background"],
                font_color=COLORS["text_primary"],
                xaxis_title="Product Category",
                yaxis_title="Average Spending ($)"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            purchase_types = ['NumWebPurchases','NumCatalogPurchases','NumStorePurchases']
            purchase_means = filtered_df[purchase_types].mean().reset_index()
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
                plot_bgcolor=COLORS["background"],
                paper_bgcolor=COLORS["background"],
                font_color=COLORS["text_primary"],
                xaxis_title="Purchase Channel",
                yaxis_title="Average Purchases"
            )
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(
                filtered_df,
                x='Income',
                y='TotalMnt',
                color='Has_Children',
                hover_data=['Age', 'Education', 'TotalPurchases'],
                title='Income vs Total Spending',
                labels={'Income': 'Income ($)', 'TotalMnt': 'Total Spending ($)'},
                color_discrete_map={0: '#FF9800', 1: '#E65100'}
            )
            fig.update_traces(
                hovertemplate=(
                    "<b>Income</b>: $%{x:,.0f}<br>"
                    "<b>Spending</b>: $%{y:,.0f}<br>"
                    "<b>Age</b>: %{customdata[0]}<br>"
                    "<b>Education</b>: %{customdata[1]}<br>"
                    "<b>Purchases</b>: %{customdata[2]}<extra></extra>"
                )
            )
            fig.update_layout(
                plot_bgcolor=COLORS["background"],
                paper_bgcolor=COLORS["background"],
                font_color=COLORS["text_primary"],
                xaxis_title="Income ($)",
                yaxis_title="Total Spending ($)"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:  # inside Tab 2, second column of second row
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
                plot_bgcolor=COLORS["background"],
                paper_bgcolor=COLORS["background"],
                font_color=COLORS["text_primary"]
            )
            st.plotly_chart(fig, use_container_width=True)    

    # Tab 3: Campaign Response
    with tab3:
        st.markdown("## 📣 Campaign Response")

        campaign_cols = ['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','Response']
        acceptance_rates = (filtered_df[campaign_cols].mean() * 100).reset_index()
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
            plot_bgcolor=COLORS["background"],
            paper_bgcolor=COLORS["background"],
            font_color=COLORS["text_primary"],
            xaxis_title="Campaign",
            yaxis_title="Acceptance Rate (%)"
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(
                filtered_df,
                x='TotalAcceptedCmp',
                y='TotalMnt',
                color='TotalAcceptedCmp',
                hover_data=['Age', 'Income', 'Has_Children'],
                title='Campaigns Accepted vs Spending',
                labels={'TotalAcceptedCmp': 'Campaigns Accepted', 'TotalMnt': 'Total Spending ($)'},
                color_continuous_scale='YlOrRd'
            )
            fig.update_traces(
                hovertemplate=(
                    "<b>Campaigns Accepted</b>: %{x}<br>"
                    "<b>Spending</b>: $%{y:,.0f}<br>"
                    "<b>Age</b>: %{customdata[0]}<br>"
                    "<b>Income</b>: $%{customdata[1]:,.0f}<br>"
                    "<b>Has Children</b>: %{customdata[2]}<extra></extra>"
                )
            )
            fig.update_layout(
                plot_bgcolor=COLORS["background"],
                paper_bgcolor=COLORS["background"],
                font_color=COLORS["text_primary"],
                xaxis_title="Campaigns Accepted",
                yaxis_title="Total Spending ($)"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(
                filtered_df,
                x='NumWebVisitsMonth',
                y='TotalPurchases',
                color='NumWebVisitsMonth',
                hover_data=['Age', 'Income', 'TotalMnt'],
                title='Web Visits vs Total Purchases',
                labels={'NumWebVisitsMonth': 'Monthly Web Visits', 'TotalPurchases': 'Total Purchases'},
                color_continuous_scale='Oranges'
            )
            fig.update_traces(
                hovertemplate=(
                    "<b>Web Visits</b>: %{x}<br>"
                    "<b>Purchases</b>: %{y}<br>"
                    "<b>Age</b>: %{customdata[0]}<br>"
                    "<b>Income</b>: $%{customdata[1]:,.0f}<br>"
                    "<b>Spending</b>: $%{customdata[2]:,.0f}<extra></extra>"
                )
            )
            fig.update_layout(
                plot_bgcolor=COLORS["background"],
                paper_bgcolor=COLORS["background"],
                font_color=COLORS["text_primary"],
                xaxis_title="Monthly Web Visits",
                yaxis_title="Total Purchases"
            )
            st.plotly_chart(fig, use_container_width=True)

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
                    
                    categories = list(cluster_profiles.columns)
                    N = len(categories)
                    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
                    values = cluster_profiles.loc[cluster_id].values.flatten().tolist()
                    values += values[:1]
                    angles += angles[:1]
                    
                    fig = go.Figure(
                        data=go.Scatterpolar(
                            r=values,
                            theta=categories + [categories[0]],
                            fill='toself',
                            line_color=COLORS["accent"],
                            hoverinfo='text',
                            text=[f"{cat}: {val:.2f}" for cat, val in zip(categories, cluster_profiles.loc[cluster_id])]
                        )
                    )
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 1]),
                            angularaxis=dict(direction="clockwise", period=N)
                        ),
                        showlegend=False,
                        title=f"Cluster {cluster_id} Profile",
                        plot_bgcolor=COLORS["background"],
                        paper_bgcolor=COLORS["background"],
                        font_color=COLORS["text_primary"]
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 🌡️ Feature Heatmap")
            df_heatmap = cluster_profiles.reset_index()
            df_heatmap.columns.name = None  # Remove name if any
            df_melted = df_heatmap.melt(id_vars=df_heatmap.columns[0], var_name='Feature', value_name='Value')
            fig = px.density_heatmap(
                df_melted,
                x='Feature',
                y=df_heatmap.columns[0],  # e.g., 'Cluster'
                z='Value',
                title='Feature Heatmap',
                color_continuous_scale='YlGnBu',
                labels={'Feature': 'Feature', df_heatmap.columns[0]: 'Cluster', 'Value': 'Normalized Value'}
            )
            fig.update_traces(
                hovertemplate="<b>Cluster</b>: %{y}<br><b>Feature</b>: %{x}<br><b>Value</b>: %{z:.2f}<extra></extra>"
            )
            fig.update_layout(
                plot_bgcolor=COLORS["background"],
                paper_bgcolor=COLORS["background"],
                font_color=COLORS["text_primary"]
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("## 💡 Marketing Strategies")
            for cid in [0, 1]:
                st.markdown(f"### {'Budget' if cid == 0 else 'Premium'} Segment")
                st.markdown(f"**Profile**: {CONFIG['cluster_descriptions'][cid]}")
                st.markdown(f"**Actions**:\n{CONFIG['cluster_marketing'][cid]}")
                st.markdown("---")
                
            # Model Performance
            st.markdown("## 📈 Model Performance")
            metrics = load_model_metrics()
            if metrics:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Silhouette Score (Clustering)", f"{metrics.get('silhouette_score', 'N/A'):.3f}")
                    st.caption("Measures cluster separation (higher = better)")
                if 'logistic_accuracy' in metrics:
                    with col2:
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
                        fig = px.imshow(
                            cm,
                            text_auto=True,
                            labels=dict(x="Predicted", y="Actual"),
                            x=["Budget", "Premium"],
                            y=["Budget", "Premium"],
                            color_continuous_scale="Blues",
                            title="Confusion Matrix"
                        )
                        fig.update_layout(
                            plot_bgcolor=COLORS["background"],
                            paper_bgcolor=COLORS["background"],
                            font_color=COLORS["text_primary"]
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Model metrics not found. Place `model_metrics.json` in the `models/` folder.")
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