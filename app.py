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
# pylance: reportUndefinedVariable=false

CONFIG = {
    "features": [
        'Age', 'Income', 'Has_Children', 'TotalMnt', 'TotalPurchases',
        'NumWebVisitsMonth', 'TotalAcceptedCmp', 'Days_Customer'
    ],
    "cluster_descriptions": {
        0: "Budget-Conscious Parent: Family-oriented, frequent low-spend shopper, loyal, older, browses often but ignores flashy campaigns",
        1: "Responsive High Spenders: Higher income, actively engages with promotions, younger, spends more per transaction"
    },
    "cluster_marketing": {
        0: (
            "• Avoid blasting campaigns — use contextual nudges: “Based on your recent views...”, “Top picks for families like yours”\n"
            "• Improve product filtering & categorization — help them find what they need fast\n"
            "• Integrate loyalty program: show points earned on every visit/purchase\n"
            "• Trigger helpful emails: abandoned cart, restock alerts, “you viewed this last week”\n"
            "• Avoid flashy banners or heavy discounts — focus on utility, trust, and convenience"
        ),
        1: (
            "• Personalized dynamic content: “For you”, “Recommended based on your style”\n"
            "• Offer flash sales, early access, VIP tiers — they respond to exclusivity\n"
            "• Use retargeting + social proof: reviews, bestsellers, influencer tags\n"
            "• Upsell/cross-sell at checkout — they’re willing to spend more"
        )
    },
    "segment_descriptions": {
        0: "Budget Segment: Value-focused family shoppers (Cluster 1)",
        1: "Premium Segment: High spenders, responsive to offers (Cluster 0)"
    },
    "segment_marketing": {
        0: (
            "• Contextual, non-salesy nudges\n"
            "• Better site navigation & filters\n"
            "• Loyalty point visibility\n"
            "• Utility-driven email triggers\n"
            "• Minimize promotional noise"
        ),
        1: (
            "• Personalized dynamic content\n"
            "• Flash sales & VIP access\n"
            "• Retargeting with social proof\n"
            "• Upsell at checkout"
        )
    },
    "colors": {
    "background": "#FFF3E0",      
    "text_primary": "#333333",    
    "accent": "#FF9800",          
    "accent_hover": "#E65100",    
    "card_bg": "#FFFFFF",       
    "border": "#E0E0E0"           
}}

FEATURES = CONFIG["features"]
COLORS = CONFIG["colors"]
CARD_STYLE = f"""
<div style="
    background-color: {COLORS['card_bg']};
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    border: 1px solid {COLORS['border']};
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
">
"""

CLOSE_CARD = "</div>"

# Set page config with new background

st.set_page_config(
    page_title="Customer Segmentation System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
f"""
<style>
/* GLOBAL COLORS & COMPACT BASE LAYOUT */
.stApp {{
    background-color: {COLORS['background']} !important;
    padding-top: 16px !important;  /* 🔼 Reduced from 60px → just enough to clear top bar */
}}

h1, h2, h3, h4, h5, h6, p, div {{
    color: {COLORS['text_primary']} !important;
}}

/* TIGHTEN MAIN CONTAINER */
.block-container {{
    padding-top: 8px !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
}}

/* COLUMN LAYOUT — same logic, tighter spacing */
[data-testid="column"],
[data-testid="column"] > div,
[data-testid="column"] > div > div {{
    display: flex !important;
    flex-direction: column !important;
    align-items: stretch !important;
    gap: 0.4rem !important;  /* Slightly tighter */
}}

[data-testid="column"] > div {{
    background-color: #fffaf0 !important;
    border-radius: 12px !important;  /* Slightly softer */
    padding: 16px !important;        /* Reduced from 18px */
    border: 1px solid #e5decf !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;  /* Softer shadow */
}}

[data-testid="column"] > div > div:first-child {{
    margin-top: 0 !important;
    padding-top: 0 !important;
}}

/* EDA CARD — leaner */
.eda-card {{
    background-color: #fffaf0 !important;
    border-radius: 12px !important;
    padding: 18px !important;        /* Reduced from 20px */
    margin-bottom: 18px !important;  /* Reduced from 22px */
    border: 1px solid #e5decf !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 10px !important;            /* Reduced from 12px */
    width: 100% !important;
}}

h3.cluster-header {{
    line-height: 1.35 !important;
    margin-bottom: 0.5rem !important;  /* Slightly tighter */
    font-weight: 700 !important;
    font-size: 1.35rem !important;
}}
h3.cluster-header span {{
    display: block !important;
}}

/* EXPANDER — compact but clear */
.stExpander {{
    background-color: #fff7e6 !important;
    border-radius: 10px !important;
    border: 1px solid #f0e6d6 !important;
    margin-bottom: 0.8rem !important;  /* Reduced from 1rem */
    position: relative !important;
}}

.stExpander::before {{
    content: "";
    position: absolute;
    inset: 0;
    width: 4px;                        /* Thinner accent */
    background-color: #E8C16D;
    border-radius: 10px 0 0 10px;
}}

.stExpander > details > summary {{
    background-color: #fff7e6 !important;
    font-weight: 700 !important;
    color: #5a4635 !important;
    padding: 12px 16px !important;     /* Reduced padding */
    font-size: 1.05rem !important;
    list-style: none !important;
}}
.stExpander summary::-webkit-details-marker {{
    display: none !important;
}}

.stExpander > details > summary + div,
.st-expanderContent {{
    background-color: #fffaf0 !important;
    padding: 14px 18px !important;     /* Reduced */
    margin: 0 !important;
    border-radius: 0 0 10px 10px !important;
    font-size: 0.95rem !important;
}}

/* CTA — subtle */
.cta-box {{
    background-color: #C8E5EE !important;
    padding: 18px !important;          /* Reduced from 22px */
    border-radius: 10px !important;
    border: 1px solid #b8e6b8 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
    text-align: center !important;
    margin: 16px 0 !important;         /* Reduced */
    font-size: 1.05rem !important;
}}

/* HEADER BANNER — REMOVED inline style; control here */
.header-banner {{
    background-size: cover !important;
    background-position: center !important;
    padding: 20px 0 !important;        /* Balanced vertical space */
    border-radius: 10px !important;
    text-align: center !important;
    margin: 0 auto 14px !important;    /* Minimal bottom gap */
    position: relative !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1) !important;
    max-width: 100% !important;
}}
.header-banner h1 {{
    margin: 0 !important;
    font-size: 2.4rem !important;
}}
.header-banner p {{
    margin: 6px 0 0 !important;
    font-size: 1.2rem !important;
}}

/* MOBILE — even tighter */
@media (max-width: 900px) {{
    .stApp {{
        padding-top: 12px !important;
    }}
    [data-testid="column"] {{
        width: 100% !important;
        flex-direction: column !important;
    }}
    h3.cluster-header {{
        font-size: 1.25rem !important;
    }}
    .header-banner h1 {{
        font-size: 2.0rem !important;
    }}
    .header-banner p {{
        font-size: 1.1rem !important;
    }}
}}
</style>
""", unsafe_allow_html=True
)

# User Authentication
USERS = {
    "admin": "segment2024",
    "analyst1": "analyze2024",
    "manager": "lead2024"
}

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""


# Load Raw Data
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

# Load Models
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
    
    

# Login Page
if not st.session_state.logged_in:
    st.markdown("---")
    st.markdown("""
    <h1 style='font-size:1.8rem; font-weight:600;'>
    🔐 CoreCart <span style='font-size:1rem; color:#6b6b6b;'>| Your core customers. Your competitive edge.</span>
    </h1>
    """, unsafe_allow_html=True)

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

# GLOBAL HEADER BANNER
if st.session_state.logged_in:
    header_image_path = "header.jpeg"
    try:
        with open(header_image_path, "rb") as f:
            img_data = f.read()
        img_base64 = base64.b64encode(img_data).decode()
        bg_image = f"data:image/jpeg;base64,{img_base64}"
        header_bg = f"url('{bg_image}')"
    except FileNotFoundError:
        st.warning("⚠️ `header.jpeg` not found — using fallback color.")
        header_bg = "#d88c4f"
    except Exception as e:
        st.warning(f"⚠️ Error loading header: {e}")
        header_bg = "#d88c4f"

    st.markdown(f""" ... """, unsafe_allow_html=True)

    st.markdown(f"""
    <style>
    .header-banner {{
        background-image: {header_bg};
        background-size: cover;
        background-position: center;
        padding: 40px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
        position: relative;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}

    .header-banner::before {{
        content: "";
        position: absolute;
        inset: 0;
        background: rgba(0, 0, 0, 0.25);
        border-radius: 12px;
        z-index: 0;
    }}

    .header-banner h1 {{
        font-size: 3rem !important;
        font-weight: 800 !important;
        color: #FFF8E1 !important;
        position: relative;
        z-index: 2;
    }}

    .header-banner p {{
        font-size: 1.35rem !important;
        font-weight: 500 !important;
        color: #FFF8E1 !important;
        position: relative;
        z-index: 2;
    }}
    </style>

    <div class="header-banner">
        <h1>CoreCart</h1>
        <p>- Turn Exploration into Actions -</p>
    </div>
    <p class="made-by">Made by Sara Fuah Jin-Yin | Final Year Computer Science Student| University of Wollongong Malaysia </p>
    """, unsafe_allow_html=True)

# Preprocessing Function
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

def plot_monthly_spending_line_chart(filtered_df, selected_category="All"):
    """
    Plot simulated monthly spending by grocery category.
    """
    import pandas as pd
    import numpy as np
    import plotly.express as px

    # Map display names to column names
    category_map = {
        "Wines": "MntWines",
        "Fruits": "MntFruits",
        "Meat": "MntMeatProducts",
        "Fish": "MntFishProducts",
        "Sweets": "MntSweetProducts",
        "Gold": "MntGoldProds"
    }

    # Filter columns based on selection
    if selected_category == "All":
        cols_to_plot = list(category_map.values())
        display_names = list(category_map.keys())
    else:
        col_name = category_map.get(selected_category)
        if col_name and col_name in filtered_df.columns:
            cols_to_plot = [col_name]
            display_names = [selected_category]
        else:
            st.info("Selected category data not available.")
            return

    # Check if columns exist
    available_cols = [col for col in cols_to_plot if col in filtered_df.columns]
    if not available_cols:
        st.info("Spending category data not available.")
        return

    # Aggregate total spending per category
    totals = {}
    for col, name in zip(available_cols, display_names):
        totals[name] = filtered_df[col].sum()

    # Simulate monthly data
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    data = []
    for category, total in totals.items():
        if total == 0:
            continue
        monthly_avg = total / 12
        np.random.seed(42)  # For consistent noise
        noise = np.random.normal(0, monthly_avg * 0.1, len(months))
        monthly_vals = np.clip(monthly_avg + noise, 0, None)
        for month, val in zip(months, monthly_vals):
            data.append({"Month": month, "Category": category, "Spending": val})

    if not data:
        st.info("No spending data to display.")
        return

    df_plot = pd.DataFrame(data)

    # Create line chart
    fig = px.line(
        df_plot,
        x='Month',
        y='Spending',
        color='Category',
        title="Monthly Spending by Grocery Category",
        markers=True
    )
    fig.update_layout(
        plot_bgcolor=COLORS["card_bg"],
        paper_bgcolor=COLORS["card_bg"],
        font_color=COLORS["text_primary"],
        xaxis_title="Month",
        yaxis_title="Spending ($)",
        legend_title="Category"
    )
    fig.update_traces(
        hovertemplate="<b>%{fullData.name}</b><br>Month: %{x}<br>Spending: $%{y:,.0f}<extra></extra>"
    )
    st.plotly_chart(fig, use_container_width=True)

# Navigation
st.sidebar.header("Navigation")
current_page = st.session_state.get("page", "Home")

page = st.sidebar.radio(
    "Go to",
    ["Home", "Customer Dashboard", "Predict Customer Segment"],
    index=["Home", "Customer Dashboard", "Predict Customer Segment"].index(current_page)
)

# keep sidebar and session_state in sync
st.session_state["page"] = page

# HOME PAGE
if page == "Home":
    st.markdown("""
    ## 🎯 Our Aim
    To provide a **data-driven customer segmentation and prediction** platform that helps businesses
     understand their customers more effectively. By analyzing key demographic, behavioral, and campaign response data
     this dashboard **identifies meaningful customer clusters** and predicts clusters of new customers with **confidence**. 
     Through clear visualizations and actionable insights,
     we empower marketers and business teams to make informed decisions, personalize marketing strategies, and 
     improve overall customer engagement and loyalty.
                
    - Ultimately,  transform raw data into valuable intelligence for smarter, more targeted business growth.
    """)

    st.markdown("---") 

    st.markdown("""
    ### 🔎 Core Features of CoreCart

    <div style='display: flex; gap: 20px; flex-wrap: wrap; justify-content: center;'>

    <div style='
        flex: 1; 
        min-width: 260px; 
        background: #fffef8; 
        padding: 22px; 
        border-radius: 14px; 
        border: 2px solid #E8C16D; 
        box-shadow: 0 3px 10px rgba(0,0,0,0.07); 
        text-align: center;
    '>
        <h4 style='margin-top: 0;'> Segment Shoppers Instantly </h4>
        <p style='margin-bottom: 0;'>Identify families, premium spenders, deal-seekers, and highly engaged digital users in seconds.</p>
    </div>

    <div style='
        flex: 1; 
        min-width: 260px; 
        background: #fffef8; 
        padding: 22px; 
        border-radius: 14px; 
        border: 2px solid #E8C16D; 
        box-shadow: 0 3px 10px rgba(0,0,0,0.07); 
        text-align: center;
    '>
        <h4 style='margin-top: 0;'> Optimize Promotions </h4>
        <p style='margin-bottom: 0;'>Send the right campaign to the right customer, avoiding wasted promos and discount abuse.</p>
    </div>

    <div style='
        flex: 1; 
        min-width: 260px; 
        background: #fffef8; 
        padding: 22px; 
        border-radius: 14px; 
        border: 2px solid #E8C16D; 
        box-shadow: 0 3px 10px rgba(0,0,0,0.07); 
        text-align: center;
    '>
        <h4 style='margin-top: 0;'> Predict New Customer Personas </h4>
        <p style='margin-bottom: 0;'>Instantly categorize new customers using our Machine Learning prediction tool.</p>
    </div>

    <div style='
    flex: 1; 
        min-width: 260px; 
        background: #fffef8; 
        padding: 22px; 
        border-radius: 14px; 
        border: 2px solid #E8C16D; 
        box-shadow: 0 3px 10px rgba(0,0,0,0.07); 
        text-align: center;
    '>
        <h4 style='margin-top: 0;'> Strengthen Marketing Strategy </h4>
        <p style='margin-bottom: 0;'>Understand how browsing, store visits, and purchases connect across channels.</p>
    </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📌 Dataset Overview")
    df_raw = load_raw_data()
    if df_raw is not None:
        df_home = preprocess_for_eda(df_raw)
        st.write(f"Records after cleaning: **{len(df_home)}**")
        st.dataframe(df_home[FEATURES].head(3))
    else:
        st.error("Dataset not found. Place `marketing_campaign.csv` in the project root.")
    st.markdown("""
        - **Source:** Kaggle – Customer Personality Analysis Dataset  
        - **Timeline:** last 2 years  
          (campaign responses, purchases, and spending amounts)
    """)

    with st.expander("💡**Learn More About the Dataset**", expanded=False):
        st.markdown("""

        ### 📊 Features Table

        | **Feature** | **Meaning** | **Why It Matters** |
        |-------------|-------------|---------------------|
        | **Age** | Calculated as `2024 - Year_Birth` | Helps segment younger vs older shopper groups. |
        | **Has_Children** | 1 if Kidhome + Teenhome > 0 | Identifies family-oriented customers with different needs. |
        | **TotalMnt** | Total amount spent across all product categories (2 yrs) | Measures overall customer value & spending intensity. |
        | **TotalPurchases** | Sum of Web, Catalog, and Store purchases | Captures total shopping activity across all channels. |
        | **TotalAcceptedCmp** | Total number of campaigns accepted | Reveals how responsive a customer is to promotions. |
        | **Days_Customer** | Days since customer joined | Shows tenure: new vs long-term customers. |

        These features were created to improve segmentation quality and make patterns easier to understand.
        """)
    
    df_full = preprocess_for_eda(df_raw)

    st.markdown("---") 

    st.markdown("""
    ### 📦 Our Data Types

    <div style='display: flex; gap: 20px; flex-wrap: wrap; justify-content: center;'>

    <div style='
        flex: 1; 
        min-width: 260px; 
        background: #fffef8; 
        padding: 22px; 
        border-radius: 14px; 
        border: 2px solid #E8C16D; 
        box-shadow: 0 3px 10px rgba(0,0,0,0.07); 
        text-align: center;
    '>
        <h4 style='margin-top: 0;'>🛍️ Purchases</h4>
        <p style='margin-bottom: 0;'>Store, online, and catalog transactions.</p>
    </div>

    <div style='
        flex: 1; 
        min-width: 260px; 
        background: #fffef8; 
        padding: 22px; 
        border-radius: 14px; 
        border: 2px solid #E8C16D; 
        box-shadow: 0 3px 10px rgba(0,0,0,0.07); 
        text-align: center;
    '>
        <h4 style='margin-top: 0;'>📊 Demographics</h4>
        <p style='margin-bottom: 0;'>Age, income, education, family size, household type.</p>
    </div>

    <div style='
    flex: 1; 
        min-width: 260px; 
        background: #fffef8; 
        padding: 22px; 
        border-radius: 14px; 
        border: 2px solid #E8C16D; 
        box-shadow: 0 3px 10px rgba(0,0,0,0.07); 
        text-align: center;
    '>
        <h4 style='margin-top: 0;'>🌐 Web Engagement</h4>
        <p style='margin-bottom: 0;'>Monthly visits, browsing intensity, digital activity.</p>
    </div>

    <div style='
        flex: 1; 
        min-width: 260px; 
        background: #fffef8; 
        padding: 22px; 
        border-radius: 14px; 
        border: 2px solid #E8C16D; 
        box-shadow: 0 3px 10px rgba(0,0,0,0.07); 
        text-align: center;
    '>
        <h4 style='margin-top: 0;'>📣 Campaign Responses</h4>
        <p style='margin-bottom: 0;'>Historical acceptance patterns across campaigns.</p>
    </div>

    <div style='
        flex: 1; 
        min-width: 260px; 
        background: #fffef8; 
        padding: 22px; 
        border-radius: 14px; 
        border: 2px solid #E8C16D; 
        box-shadow: 0 3px 10px rgba(0,0,0,0.07); 
    text-align: center;
    '>
        <h4 style='margin-top: 0;'>🛒 Basket Spend Categories</h4>
        <p style='margin-bottom: 0;'>Wine, meat, fruits, sweets, fish, and premium products.</p>
    </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🚀 Getting Started")
    st.markdown("""
        This system uses machine learning to segment grocery retail customers into **two clear and meaningful groups** based on spending behavior, engagement, and promotion responsiveness:

        - **👨‍👩‍👧‍👦 Budget Conscious Parent** – Value-conscious, price-sensitive shoppers focused on essentials  
        - **💎 Responsive High Spenders** – Higher-income, brand-loyal shoppers who respond well to promotions and premium offers

        This two segment approach:
        - Simplifies decision-making for marketing teams  
        - Reflects the natural divide found in the customer dataset (based on spending median)  
        - Enables highly targeted communication, loyalty programs, and product recommendations  
        - Helps improve retention, boost ROI, and optimize promotional investments  

        By understanding the differences between these customer groups, businesses can turn complex customer data into **clear, actionable insights**.
    """)

    
# CUSTOMER DASHBOARD
elif page == "Customer Dashboard":
    st.title("📈 Customer Dashboard")
    st.markdown("Ready to **understand** your customers better?")
    
    df_raw = load_raw_data()
    if df_raw is None:
        st.error("Dataset not found.")
        st.stop()
    
    df_full = preprocess_for_eda(df_raw)

    # DEFINE SIDEBAR FILTERS FIRST =====
    st.sidebar.header("🔍 Filters")
    
    # Age Group
    age_bins = [18, 30, 45, 60, 100]
    age_labels = ["18-29", "30-44", "45-59", "60+"]
    if 'Age' in df_full.columns:
        df_full['AgeGroup'] = pd.cut(df_full['Age'], bins=age_bins, labels=age_labels, right=False)
    else:
        df_full['AgeGroup'] = "Unknown"
    selected_age_group = st.sidebar.selectbox("Age Group", ["All"] + list(age_labels))
    
    # Grocery Category Filter
    category_options = ["All", "Wines", "Fruits", "Meat", "Fish", "Sweets", "Gold"]
    selected_category = st.sidebar.selectbox("Grocery Category", category_options)
    
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

    # APPLY CATEGORY FILTER 
    category_col_map = {
        "Wines": "MntWines",
        "Fruits": "MntFruits",
        "Meat": "MntMeatProducts",
        "Fish": "MntFishProducts",
        "Sweets": "MntSweetProducts",
        "Gold": "MntGoldProds"
    }

    if selected_category != "All":
        col_name = category_col_map[selected_category]
        if col_name in df_full.columns:
            df_full = df_full[df_full[col_name] > 0].copy()
        else:
            st.warning(f"Category {selected_category} not found in data.")

    #  APPLY OTHER FILTERS 
    filtered_df = df_full.copy()
    if selected_age_group != "All":
        filtered_df = filtered_df[filtered_df['AgeGroup'] == selected_age_group]
    if 'TotalMnt' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['TotalMnt'] >= spending_range[0]) & (filtered_df['TotalMnt'] <= spending_range[1])
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

    # PLOT MONTHLY SPENDING CHART
    plot_monthly_spending_line_chart(filtered_df, selected_category)

    # KPIs 
    avg_spent = filtered_df['TotalMnt'].mean() if 'TotalMnt' in filtered_df.columns else 0
    avg_age = filtered_df['Age'].mean() if 'Age' in filtered_df.columns else 0
    avg_income = filtered_df['Income'].mean() if 'Income' in filtered_df.columns else 0
    col1, col2, col3 = st.columns(3)
    kpi_style = "background-color: #FFFFFF; padding: 15px; border-radius: 8px; border-left: 5px solid #FF9800; color: #333333; box-shadow: 0 2px 4px rgba(0,0,0,0.05);"
    with col1:
        st.markdown(f"""
        <div style="{kpi_style}">
            <h4>💰 Average Spending</h4>
            <p style="font-size: 20px; font-weight: bold;">${avg_spent:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="{kpi_style}">
            <h4>👤 Average Age</h4>
            <p style="font-size: 20px; font-weight: bold;">{avg_age:.1f}</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div style="{kpi_style}">
            <h4>💳 Average Income</h4>
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
                plot_bgcolor=COLORS["card_bg"],
                paper_bgcolor=COLORS["card_bg"],
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
                plot_bgcolor=COLORS["card_bg"],
                paper_bgcolor=COLORS["card_bg"],
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
                plot_bgcolor=COLORS["card_bg"],
                paper_bgcolor=COLORS["card_bg"],
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
                plot_bgcolor=COLORS["card_bg"],
                paper_bgcolor=COLORS["card_bg"],
                font_color=COLORS["text_primary"],
                xaxis_title="Marital Status",
                yaxis_title="Customer Count"
            )
            st.plotly_chart(fig, use_container_width=True)


        # Bottom expanders for Demographics tab
        with st.expander("📌 **Key Takeaways**", expanded=False):
            st.markdown("""
            👥 Most consumers are **50 years old**  
            💰 Most common income: **$38.5k**  
            🎓 Highest education level: **High school graduate**  
            💍 Most consumers are **married**  
            """)

        with st.expander("🚀 **Actionable Insights**", expanded=False):
            st.markdown("""
            📣 Tailor messaging to **midlife priorities** :
             - financial security
             - family
             - value driven purchases
                          
            💰 **Price products** in  affordability thresholds (e.g. payment plans, bundles).  
                        
            🎯 **Focus ad targeting** on married, 45–55-year-old audiences on platforms like Facebook or banners.  
                          
            📚 **Simplify communication** : avoid jargon, use clear language aligned with high school literacy levels.  
                        
            🏡 **Highlight stability & trust** in branding : audience values reliability over novelty.  
            """)


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
                plot_bgcolor=COLORS["card_bg"],
                paper_bgcolor=COLORS["card_bg"],
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
                plot_bgcolor=COLORS["card_bg"],
                paper_bgcolor=COLORS["card_bg"],
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
                plot_bgcolor=COLORS["card_bg"],
                paper_bgcolor=COLORS["card_bg"],
                font_color=COLORS["text_primary"],
                xaxis_title="Income ($)",
                yaxis_title="Total Spending ($)"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:  
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
                plot_bgcolor=COLORS["card_bg"],
                paper_bgcolor=COLORS["card_bg"],
                font_color=COLORS["text_primary"]
            )
            st.plotly_chart(fig, use_container_width=True)

        # Bottom expanders for begaviour tab
        with st.expander("📌 **Key Takeaways**", expanded=False):
            st.markdown("""
            🍷 **Top Grocery Category**: **Wine**  
                         
            🛒 **Channel Preference**:  
            - **In-store** purchases lead, but **online** is close behind  
                        
            📈 Higher income strongly correlates with higher spending  
                        
            ⚖️ **Spend Distribution**: Highly uneven, a small group drives most revenue  
            """)
            
        with st.expander("🚀 **Actionable Insights**", expanded=False):
            st.markdown("""
            🍷 **Leverage wine dominance**:  
            - Create curated bundles (e.g., wine + cheese)  
            - a loyalty tier for wine buyers  
                        
            🌐 **Bridge online & offline**:  
            - Enable “buy online, pick up in-store” to capture hybrid shoppers.  
  
            🎯 **Target high-income segments**:  
            -  tailored ads showcasing premium products and services.   
                        
            📊 **Implement loyalty programs**:
            - encourage repeat purchases among top spenders.  
            """)         

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
            plot_bgcolor=COLORS["card_bg"],
            paper_bgcolor=COLORS["card_bg"],
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
                plot_bgcolor=COLORS["card_bg"],
                paper_bgcolor=COLORS["card_bg"],
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
                plot_bgcolor=COLORS["card_bg"],
                paper_bgcolor=COLORS["card_bg"],
                font_color=COLORS["text_primary"],
                xaxis_title="Monthly Web Visits",
                yaxis_title="Total Purchases"
            )
            st.plotly_chart(fig, use_container_width=True)
            

        with st.expander("📌 **Key Takeaways**", expanded=False):
            st.markdown("""
            📊 Higher spenders tend to accept more campaigns.  
            🌐 High web traffic does NOT guarantee high purchases.  
            ✅ **Browsers** are not **buyers**.  
            🛒 **Loyal but rare web browser** buyers are dominant.  
            """)
            
        with st.expander("🚀 **Actionable Insights**", expanded=False):
            st.markdown("""
            🌐For **browsers**: 
            - high visits, low purchases
            - Use retargeting with product recommendations based on browsing history 
            - e.g. “You viewed this last week still interested? 👀 ”  
                        

            🌟 For **loyal** buyers: 
            - low visits, high purchases
            - Reward them with fast checkout, loyalty points, or surprise gifts to reinforce their behavior.  
                        

            🛍️ For **mid range** shoppers: 
            - Test limited-time offers or “free shipping on your next order” to nudge them toward more frequent purchases.
            """)


   #Clustering & Marketing tab
    with tab4:
        scaler, pca, clusterer, cluster_profiles, supervised_model = load_models()
        if cluster_profiles is not None and len(cluster_profiles) == 2:
            st.markdown("## 🧩 Customer Persona")

            # Define rich, actionable insights per cluster
            cluster_insights = {
                0: {
                    "name": "👨‍👩‍👧‍👦 Budget Conscious Parent",
                    "profile": [
                        "  **Family-oriented**, often with children at home",
                        "  **Frequent but low-spend** transactions — necessity-driven",
                        "  **Long-term, loyal** customer ",
                        "🌐 **Visits website often** — uses it for research/comparison",
                        "🔕 **Ignores most promotional campaigns** — filters out “noise”"
                    ],
                    "why_it_matters": [
                        " They’re **not disengaged** —  **actively browsing** but avoiding hype",
                        " Promotions fail because they feel **irrelevant or overwhelming**",
                        " They want **utility, trust, and convenience** — not flashy banners",
                        " They’re your **retention core** — but need the right triggers to convert"
                    ],
                    "marketing": [
                        " Use **contextual nudges**: “Based on your recent views…”, “Top picks for families like yours”",
                        "🔍 **Improve filtering & search** — help them find what they need in <10 seconds",
                        "🎖️ **Show loyalty points** on every visit/purchase — reinforce long-term value",
                    "📧 Trigger **helpful emails**: restock alerts, “you viewed this last week”, abandoned cart",
                    "🚫 **Avoid discount-heavy messaging** — focus on **practicality, reliability, and ease**"
                    ]
                },
                1: {
                    "name": " 💰Responsive High Spenders",
                    "profile": [
                        "  **Higher income** and **younger** demographic",
                        "  **Actively engages** with promotional campaigns",
                        " 💵 **High total spending** and **responds to 2+ campaigns** on average",
                        " ✨ Prefers **experiential, premium** offerings",
                        " 📣 Open to **experimentation** and new product discovery"
                    ],
                    "why_it_matters": [
                        " Campaigns matches their lifestyle, seeing offers as **value**",
                        " Digital engagement **directly converts** — ideal for retargeting",
                        " **Growth engine** for new premium launches"
                    ],
                    "marketing": [
                        "🎯 **Personalize dynamically**: “Recommended for you”, “Based on your style”",
                        "✨ Offer **flash sales, early access, VIP tiers** — exclusivity drives action",
                        "📣 Use **social proof**: bestsellers, reviews, influencer tags",
                        "🛒 **Upsell/cross-sell** at checkout ",
                        "📲 Retarget with **lifestyle-aligned** creatives"
                    ]
                }
            }

        def display_cluster(cluster_id):
            import re
            insight = cluster_insights[cluster_id]
            model_id = cluster_id + 1

            # convert **bold** → <strong>…</strong>
            def md_bold_to_html(s: str) -> str:
                return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)

            # list items
            profile_items = "".join(
                f'<li style="margin:6px 0; line-height:1.45;">{md_bold_to_html(p)}</li>'
                for p in insight["profile"]
            )

            # style attributes are single-line
            card_html = f'''
        <div class="eda-card" style="padding-top:15px !important;">
        <h3 style="margin-bottom:10px;">
            Cluster {model_id}:<br>
            <span style="font-weight:700;">{insight['name']}</span>
        </h3>

        <h4 style="margin:0 0 6px 0;">👤 Behavioral Profile</h4>
        <ul style="margin-top:4px; margin-bottom:0; padding-left:1.2rem; list-style:disc; color:{COLORS['text_primary']};">
            {profile_items}
        </ul>
        </div>
        '''
            st.markdown(card_html, unsafe_allow_html=True)

            # Expanders
            with st.expander("💡 **Why This Matters**", expanded=False):
                for point in insight["why_it_matters"]:
                    st.markdown(f"- {point}")
            with st.expander("💼 **Recommended Marketing Strategy**"):
                st.markdown("#### Actionable Tactics:")
                for tactic in insight["marketing"]:
                    st.markdown(f"- {tactic}")

        # Render Columns Cleanly 
        col1, col2 = st.columns(2)

        with col1:
            display_cluster(0)  

        with col2:
            display_cluster(1)  

        # MAIN COMPARISONS SECTION
        st.markdown("### 🔍 Main Comparisons")

        # Create two columns: radar chart + description
        col_radar, col_desc = st.columns([2, 1])

        # Radar chart (overlapping)
        with col_radar:
            categories = list(cluster_profiles.columns)
            fig = go.Figure()
            for cluster_id in [0, 1]:
                values = cluster_profiles.loc[cluster_id].values.flatten().tolist()
                values += values[:1]  # close the loop
                theta = categories + [categories[0]]
                color = COLORS["accent"] if cluster_id == 0 else "#4CAF50"
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=theta,
                    fill='toself',
                    name=f'Cluster {cluster_id}',
                    line_color=color
                ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1]),
                    angularaxis=dict(direction="clockwise")
                ),
                showlegend=True,
                title="Cluster Comparison",
                plot_bgcolor=COLORS["card_bg"],
                paper_bgcolor=COLORS["card_bg"],
                font_color=COLORS["text_primary"]
            )
            st.plotly_chart(fig, use_container_width=True)

        # Description
        with col_desc:
            st.markdown("""
            **Key Differences:**
            - **Cluster 1**: Family-focused, loyal, browses often but ignores flashy campaigns.
            - **Cluster 2**: Higher income, younger, responds to promotions, spends more.

           **Strategic Implication:**  
            Tailor messaging:  
            → **Cluster 1**: Practicality, trust, convenience  
            → **Cluster 2**: Exclusivity, personalization  
            """)
        # Heatmap 
        st.markdown("### 🌡️ Feature Normalization Heatmap")
        df_heatmap = cluster_profiles.reset_index()
        df_heatmap.columns.name = None
        df_melted = df_heatmap.melt(id_vars=df_heatmap.columns[0], var_name='Feature', value_name='Value')
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
        plot_bgcolor=COLORS["card_bg"],
        paper_bgcolor=COLORS["card_bg"],
        font_color=COLORS["text_primary"]
        )
            
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("ℹ️ **Understanding the Heatmap**", expanded=False):
            st.markdown("""
            ##### 🧩 What Is This?

            - **Purpose:** Compare how different customer clusters behave across key demographic  
              and behavioral features.
            - **Rows:** Represent customer clusters.
            - **Columns:** Represent features.
            - **Color Intensity:** The deeper the color, the higher the normalized value of that  
              feature within the cluster.

            ##### 🔎 Key Defining Characteristics Shown
            1. **Children**  
            Indicates whether the customer has children in the household.
            2. **Total Purchases**  
               Shows overall purchase volume across all channels (web, store, catalog).
            3. **Monthly Number of Website Visits**  
               Reflects browsing frequency and digital engagement level.
            """)


        # ✅ CTA Container
        cta = st.container()
        with cta:
            st.markdown("""
            <div style="
                background-color: #C8E5EE;
                padding: 20px;
                border-radius: 12px;
                border: 1px solid #b8e6b8;
                box-shadow: 0 2px 6px rgba(0,0,0,0.05);
                text-align: center;
                margin-top: 15px;
                margin-bottom: 25px;
            ">
                <h3 style="
                    color: #2a5d2a;
                    font-weight: 700;
                    margin-bottom: 6px;
                ">
                    Can't predict the customer profile of a new shopper?
                </h3>
                <p style="
                    color: #3d6e3d;
                    font-size: 1.05rem;
                    margin-bottom: 16px;
                ">
                    We can help you.
                </p>
            </div>
            """, unsafe_allow_html=True)

        go_predict = st.button(" Try the Prediction Tool 😉")

        if go_predict:
            st.session_state["page"] = "Predict Customer Segment"
            st.rerun()   

elif page == "Predict Customer Segment":
    st.title("🔮 Predict Customer Segment")

    scaler, pca, clusterer, cluster_profiles, supervised_model = load_models()
    if hasattr(scaler, 'feature_names_in_'):
        expected_features = list(scaler.feature_names_in_)
        FEATURES = [f for f in expected_features if f in CONFIG["features"]]
    else:
        FEATURES = CONFIG["features"]

    if scaler is None or supervised_model is None or cluster_profiles is None:
        st.error("❌ Models or cluster profiles not loaded.")
        st.stop()

    #  INSIGHTS FOR CLUSTER (unchanged — kept for reference)
    cluster_insights = {
        0: {
            "name": "👨‍👩‍👧‍👦 Budget Conscious Parent",
            "profile": [
                "**Family-oriented** with children at home",
                "**Frequent but low-spend** shopping",
                "**Loyal** long-term customers",
                "🌐 Visits website often for comparison",
                "🔕 Ignores most promotional campaigns"
            ],
            "why_it_matters": [
                "Not disengaged — just **filtering noise**",
                "Promos fail because they feel irrelevant",
                "They value **utility, trust, convenience**",
                "They're a **retention cornerstone**"
            ],
            "marketing": [
                "Contextual nudges (recently viewed items)",
                "🔍 Improve filtering & search",
                "🎖️ Display loyalty points frequently",
                "📧 Helpful emails: restocks, reminders",
                "🚫 Avoid heavy discount spam"
            ]
        },
        1: {
            "name": "💰 Responsive High Spenders",
            "profile": [
                "**Higher income** and **younger** demographic",
                "**Actively engages** with promotional campaigns",
                "💵 High total spending & responds to **2+ campaigns**",
                "✨ Prefers experiential & premium offerings",
                "📣 Open to trying new products"
            ],
            "why_it_matters": [
                "They view campaigns as **value**, not noise",
                "Digital engagement **converts strongly**",
                "Core group for premium product launches"
            ],
            "marketing": [
                "🎯 Personalized recommendations",
                "✨ VIP tiers, early access, exclusives",
                "📣 Use reviews, influencers, social proof",
                "🛒 Upsell/cross-sell at checkout",
                "📲 Retarget with lifestyle content"
            ]
        }
    }

    #  FORM
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
            total_cmp = st.number_input("Accepted Campaigns (0–6)", min_value=0, max_value=6, value=1)
            days_customer = st.number_input("Days as Customer", min_value=0, value=365)

        submitted = st.form_submit_button("🚀 Predict Segment")

    # PERFORM PREDICTION
    if submitted:
        try:
            input_dict = {
                'Age': age,
                'Income': income,
                'Has_Children': has_children,
                'TotalMnt': total_mnt,
                'TotalPurchases': total_purchases,
                'NumWebVisitsMonth': web_visits,
                'TotalAcceptedCmp': total_cmp,
                'Days_Customer': days_customer
            }

            if hasattr(scaler, 'feature_names_in_'):
                ordered_features = list(scaler.feature_names_in_)
            else:
                ordered_features = FEATURES
                
            reordered_input = {feat: input_dict[feat] for feat in FEATURES}
            X_input = pd.DataFrame([reordered_input])

            # Double-check: all features present
            missing = set(FEATURES) - set(input_dict.keys())
            extra = set(input_dict.keys()) - set(FEATURES)
            if missing:
                st.error(f"❌ Missing features: {missing}")
                st.stop()
            if extra:
                st.warning(f"⚠️ Extra features ignored: {extra}")

            # Optional: safety check
            if hasattr(scaler, 'feature_names_in_') and list(X_input.columns) != list(scaler.feature_names_in_):
                st.error("Feature order mismatch!")
                st.stop()

            scaled = scaler.transform(X_input)
            pred_label = int(supervised_model.predict(scaled)[0])

            display_cluster_id = pred_label + 1
            insight = cluster_insights[pred_label]
            segment_name = insight["name"]

            confidence_msg = ""
            if hasattr(supervised_model, "predict_proba"):
                proba = supervised_model.predict_proba(scaled)[0]
                confidence = proba[pred_label] * 100  # ← Use prob for predicted class
                confidence_msg = f" (Confidence: **{confidence:.1f}%**)"
            else:
                confidence_msg = " (Confidence: N/A)"

            #  Display Result
            st.success(
                f"### 🎯 Predicted Segment:\n"
                f"**Cluster {display_cluster_id} – {segment_name}**\n"
                f"{confidence_msg}"
            )

            #  PROFILE
            st.markdown("### 👤 Customer Profile")
            for point in insight["profile"]:
                st.markdown(f"- {point}")

            #  RADAR CHART
            st.markdown("### 📊 Behavioral Radar Chart")

            categories = list(cluster_profiles.columns)
            values = cluster_profiles.loc[pred_label].tolist()
            values += values[:1]

            fig = go.Figure(
                data=go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    fill='toself',
                    line_color=COLORS["accent"],
                )
            )

            fig.update_layout(
                title=f"Cluster {display_cluster_id} Behavioral Profile",
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False,
                paper_bgcolor=COLORS["background"],
                plot_bgcolor=COLORS["background"],
                font_color=COLORS["text_primary"]
            )

            st.plotly_chart(fig, use_container_width=True)

            #  WHY IT MATTERS
            with st.expander("💡 **Why This Matters**"):
                for point in insight["why_it_matters"]:
                    st.markdown(f"- {point}")

            # MARKETING STRATEGY
            with st.expander("💼 **Recommended Marketing Strategy**"):
                st.markdown("#### Actionable Tactics:")
                for tactic in insight["marketing"]:
                    st.markdown(f"- {tactic}")

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            st.exception(e)  
 

# FOOTER 
st.markdown("---")
col1, col2 = st.columns([5, 1])
with col1:
    st.caption("💡 Powered by binary customer segmentation using Logistic Regression and KMeans.")
with col2:
    if st.button("🚪 Logout", key="logout_footer"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()