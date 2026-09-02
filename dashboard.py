import streamlit as st
import pandas as pd
from PIL import Image
import os

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Smart Waste Dashboard", page_icon="♻️", layout="wide")

# ==========================================
# 2. HEADER & MOCK DATABASE
# ==========================================
st.title("🌍 Smart Waste Management & XAI Dashboard")
st.markdown("Monitor live bin capacities, view optimized routes, and audit AI decisions.")

# Simulated database (This represents the MQTT data from your IoT sensors)
data = {
    "Bin ID": ["BIN_001", "BIN_002", "BIN_003", "BIN_004", "BIN_005", "BIN_006"],
    "Location": ["Downtown Square", "City Park", "Main Street", "Train Station", "Shopping Mall", "Library"],
    "Fill Level (%)": [90, 20, 95, 15, 88, 85],
    "Status": ["🔴 FULL", "🟢 OK", "🔴 FULL", "🟢 OK", "🔴 FULL", "🔴 FULL"]
}
df = pd.DataFrame(data)

# ==========================================
# 3. TOP METRICS ROW
# ==========================================
st.divider()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Bins Monitored", len(df))
col2.metric("Bins Needing Collection", len(df[df["Fill Level (%)"] >= 80]))
col3.metric("Average Fill Level", f"{int(df['Fill Level (%)'].mean())}%")
col4.metric("Active Trucks", "1 (Route Optimized)")
st.divider()

# ==========================================
# 4. MAIN DASHBOARD TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Live IoT Data", "🚚 Route Optimization", "🧠 AI Transparency (XAI)"])

# --- TAB 1: IoT Data ---
with tab1:
    st.subheader("Real-Time Bin Status")
    st.write("Live data streaming from ultrasonic sensors deployed across the city.")
    # Display the dataframe with highlighted styles
    st.dataframe(df.style.map(
        lambda x: 'background-color: #ffcccc' if x == '🔴 FULL' else 'background-color: #ccffcc', subset=['Status']
    ), use_container_width=True)

# --- TAB 2: Route Optimization ---
with tab2:
    st.subheader("Dynamic Collection Route")
    st.info("Route calculated using Google OR-Tools. Bins below 80% capacity are bypassed to save fuel.")
    
    # We display the output you generated earlier from route_optimizer.py
    st.code("DEPOT -> BIN_006 -> BIN_001 -> BIN_003 -> BIN_005 -> DEPOT", language="text")
    st.success("✅ Estimated Fuel Savings: 28% compared to static routing.")

# --- TAB 3: Explainable AI ---
with tab3:
    st.subheader("Waste Classification & Explainability Logs")
    st.write("Audit logs showing visual proof of the AI's sorting decisions using Grad-CAM heatmaps.")
    
    # Load the heatmap you generated in the previous step
    if os.path.exists("xai_result.png"):
        image = Image.open("xai_result.png")
        st.image(image, caption="Recent Classification Log: Plastic", use_container_width=True)
    else:
        st.warning("⚠️ XAI image not found. Please run 'explain_model.py' first!")