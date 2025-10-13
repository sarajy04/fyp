# ================================================================
# 🧩 CUSTOMER SEGMENTATION DASHBOARD (with LOGIN)
# ================================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from math import pi
import os

# ================================================================
# 1️⃣ PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide",
    page_icon="🧩"
)

# ================================================================
# 2️⃣ AUTHORIZED USER CREDENTIALS
# ================================================================
# ✅ Only authorized users can log in (no registration allowed)
AUTHORIZED_USERS = {
    "admin": "password123",     # you can change these
    "manager": "securepass456"
}

# ================================================================
# 3️⃣ SESSION STATE MANAGEMENT
# ================================================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# ================================================================
# 4️⃣ LOGIN FUNCTION
# ================================================================
def login():
    st.title("🔐 Secure Login")
    st.markdown("Only authorized users can access the segmentation dashboard.")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_btn = st.button("Login")

    if login_btn:
        if username in AUTHORIZED_USERS and AUTHORIZED_USERS[username] == password:
            st.session_state.authenticated = True
            st.success("✅ Login successful!")
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

# ================================================================
# 5️⃣ LOGOUT FUNCTION
# ================================================================
def logout():
    st.session_state.authenticated = False
    st.success("👋 You have been logged out.")
    st.rerun()

# ================================================================
# 6️⃣ MAIN DASHBOARD FUNCTION
# ================================================================
def main_dashboard():
    st.title("🧩 Customer Segmentation System")
    st.sidebar.button("🚪 Logout", on_click=logout)

    st.markdown("""
    This dashboard identifies customer segments based on purchasing behavior and demographics.
    Upload new data to predict segment membership and explore existing cluster profiles.
    """)

    # Load saved models
    model_dir = "models"
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    pca = joblib.load(os.path.join(model_dir, "pca.pkl"))
    clusterer = joblib.load(os.path.join(model_dir, "clusterer.pkl"))
    feature_order = open(os.path.join(model_dir, "feature_order.txt")).read().splitlines()
    cluster_profiles = pd.read_csv(os.path.join(model_dir, "cluster_profiles.csv"), index_col=0)

    st.sidebar.header("Model Info")
    st.sidebar.success(f"✅ Loaded model with {len(cluster_profiles)} clusters")

    # File upload
    uploaded_file = st.file_uploader("📤 Upload a customer CSV file for segmentation", type=["csv"])

    if uploaded_file:
        df_new = pd.read_csv(uploaded_file)
        st.subheader("Preview of Uploaded Data")
        st.dataframe(df_new.head())

        X_new = df_new[feature_order]
        X_scaled = scaler.transform(X_new)
        X_pca = pca.transform(X_scaled)
        preds = clusterer.predict(X_pca)
        df_new["PredictedCluster"] = preds

        st.success("✅ Clusters assigned successfully!")
        st.dataframe(df_new.head())

        # Cluster distribution
        st.subheader("📊 Cluster Distribution (Uploaded Data)")
        cluster_counts = df_new["PredictedCluster"].value_counts().sort_index()
        st.bar_chart(cluster_counts)

        # Radar chart
        st.subheader("🕸️ Cluster Profile Radar Chart")
        scaler_radar = MinMaxScaler()
        cluster_profiles_scaled = pd.DataFrame(
            scaler_radar.fit_transform(cluster_profiles),
            columns=cluster_profiles.columns,
            index=cluster_profiles.index
        )

        cluster_to_plot = st.selectbox(
            "Select a Cluster to View Profile:",
            cluster_profiles_scaled.index
        )

        def radar_chart(data, title):
            categories = list(data.index)
            values = data.values.flatten().tolist()
            values += values[:1]
            angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
            angles += angles[:1]

            fig, ax = plt.subplots(subplot_kw={'polar': True}, figsize=(6, 6))
            ax.set_theta_offset(pi / 2)
            ax.set_theta_direction(-1)
            plt.xticks(angles[:-1], categories)
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=title)
            ax.fill(angles, values, alpha=0.25)
            plt.title(title, size=15, y=1.1)
            st.pyplot(fig)

        radar_chart(cluster_profiles_scaled.loc[cluster_to_plot], f"Cluster {cluster_to_plot} Profile")

        # Download segmented data
        st.subheader("📥 Download Predictions")
        csv = df_new.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Clustered Data as CSV",
            data=csv,
            file_name="segmented_customers.csv",
            mime="text/csv"
        )

    else:
        st.info("⬆️ Please upload a CSV file to start segmentation.")

    # Cluster summary
    st.markdown("---")
    st.header("📈 Cluster Behavior Summary")
    st.dataframe(cluster_profiles)
    st.caption("These represent the normalized behavior of each customer cluster.")

# ================================================================
# 7️⃣ LOGIN LOGIC
# ================================================================
if not st.session_state.authenticated:
    login()
else:
    main_dashboard()
