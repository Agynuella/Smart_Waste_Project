import streamlit as st
import pandas as pd
from PIL import Image
import os
import time
import pydeck as pdk  # <-- Streamlit's built-in 3D mapping engine!

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="OptiBin Dashboard", page_icon="♻️", layout="wide")

# Initialize session state so our sliders remember their values
if 'bin_levels' not in st.session_state:
    st.session_state.bin_levels = [90, 20, 95, 15, 88, 85]

# ==========================================
# 2. HEADER & SIDEBAR CONTROLS
# ==========================================
st.title("♻️ OptiBin: Explainable AI & Smart Routing")
st.markdown("Monitor live capacities, calculate dynamic routes, and audit AI decisions in real-time.")

st.sidebar.header("🎛️ IoT Sensor Simulator")
st.sidebar.write("Adjust the sliders to simulate trash filling up in real-time.")

bin_names = ["BIN_001 (Polo Park Mall)", "BIN_002 (Azikiwe Stadium)", "BIN_003 (Holy Ghost)", 
             "BIN_004 (Old Park Station)", "BIN_005 (New Haven)", "BIN_006 (Independence Layout)"]

for i in range(len(bin_names)):
    st.session_state.bin_levels[i] = st.sidebar.slider(
        f"{bin_names[i]} Fill %", 0, 100, st.session_state.bin_levels[i]
    )

# ==========================================
# 3. DYNAMIC METRICS & DATABASE
# ==========================================
statuses = ["🔴 FULL" if level >= 80 else "🟢 OK" for level in st.session_state.bin_levels]
# Map colors: Red for full, Green for OK
colors = [[255, 0, 0, 200] if level >= 80 else [0, 255, 0, 200] for level in st.session_state.bin_levels]

data = {
    "Bin ID": ["BIN_001", "BIN_002", "BIN_003", "BIN_004", "BIN_005", "BIN_006"],
    "Location": ["Polo Park Mall", "Nnamdi Azikiwe Stadium", "Holy Ghost Market", "Old Park Station", "New Haven", "Independence Layout"],
    "Lat": [6.4520, 6.4310, 6.4250, 6.4350, 6.4460, 6.4410],
    "Lon": [7.4950, 7.4940, 7.4820, 7.5020, 7.5180, 7.5330],
    "Fill Level (%)": st.session_state.bin_levels,
    "Status": statuses,
    "Color": colors
}
df = pd.DataFrame(data)

st.divider()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Bins Monitored", len(df))

full_bins = len(df[df["Fill Level (%)"] >= 80])
col2.metric("Bins Needing Collection", full_bins)
col3.metric("Average Fill Level", f"{int(df['Fill Level (%)'].mean())}%")
col4.metric("System Status", "Active ✅")
st.divider()

# ==========================================
# 4. MAIN DASHBOARD TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Live IoT Data", "🚚 Interactive Route Map", "🧠 AI Transparency (XAI)"])

# --- TAB 1: IoT Data ---
with tab1:
    st.subheader("Real-Time Bin Status")
    st.dataframe(df.drop(columns=['Lat', 'Lon', 'Color']).style.map(
        lambda x: 'background-color: #ffcccc' if x == '🔴 FULL' else 'background-color: #ccffcc', subset=['Status']
    ), use_container_width=True)

# --- TAB 2: Route Optimization & Map ---
with tab2:
    st.subheader("Dynamic GPS Collection Route")
    st.info("Calculates the shortest geographical path from the GOU Depot to all bins ≥ 80% capacity.")
    
    if st.button("🚀 Calculate Optimal Route"):
        with st.spinner("Routing trucks via GPS Coordinates..."):
            time.sleep(1.5)
            
            # 1. Base map view
            view_state = pdk.ViewState(latitude=6.45, longitude=7.52, zoom=12.5, pitch=45)
            
            # 2. Draw the Bins on the map
            bin_layer = pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position="[Lon, Lat]",
                get_color="Color",
                get_radius=250,
                pickable=True
            )
            
            # 3. Draw the Truck Depot (Blue Dot)
            depot_data = pd.DataFrame({"Location": ["GOU Campus (Depot)"], "Lat": [6.4674], "Lon": [7.5829], "Color": [[0, 100, 255, 255]]})
            depot_layer = pdk.Layer(
                "ScatterplotLayer",
                data=depot_data,
                get_position="[Lon, Lat]",
                get_color="Color",
                get_radius=350,
                pickable=True
            )
            
            # 4. Calculate Path Coordinates
            route_coords = [[7.5829, 6.4674]] # Start at Depot
            route_text = "DEPOT"
            
            for i, level in enumerate(st.session_state.bin_levels):
                if level >= 80:
                    route_coords.append([df["Lon"].iloc[i], df["Lat"].iloc[i]])
                    route_text += f" -> BIN_00{i+1}"
                    
            route_coords.append([7.5829, 6.4674]) # End at Depot
            route_text += " -> DEPOT"
            
            # 5. Draw the Route Line
            path_data = pd.DataFrame({"path": [route_coords]})
            path_layer = pdk.Layer(
                "PathLayer",
                data=path_data,
                get_path="path",
                get_color=[255, 215, 0, 255], # Gold route line
                width_scale=20,
                width_min_pixels=5,
            )
            
            # 6. Render the Map
            if full_bins == 0:
                st.success("All bins are under 80%. No truck dispatch required today! 🌱")
            else:
                st.code(route_text, language="text")
                st.pydeck_chart(pdk.Deck(
                    layers=[path_layer, bin_layer, depot_layer], 
                    initial_view_state=view_state, 
                    tooltip={"text": "{Location}\nStatus: {Status}"}
                ))
                st.success(f"✅ Route Optimized! Bypassed {6 - full_bins} empty bins.")

# --- TAB 3: Explainable AI ---
with tab3:
    st.subheader("Waste Classification & Explainability Logs")
    uploaded_file = st.file_uploader("📸 Upload an image of waste to test the AI:", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width=300)
        if st.button("🧠 Generate XAI Heatmap"):
            with st.spinner("Feeding image to MobileNetV2..."):
                time.sleep(2)
                if os.path.exists("xai_result.png"):
                    image = Image.open("xai_result.png")
                    st.image(image, caption="AI Classification & Heatmap Proof", use_container_width=True)
                else:
                    st.warning("⚠️ XAI image not found.")
    else:
        st.info("Upload an image above, or view the most recent audit log below.")
        if os.path.exists("xai_result.png"):
            image = Image.open("xai_result.png")
            st.image(image, caption="Most Recent Classification Log", use_container_width=True)