import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="PPM Detection Models Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal default data
default_model_data = {
    "name": ["PPM-v8l", "PPM-v8s_v1", "PPM-v8s_v2", "PPM-v9s", "PPM-v8m"],
    "map50": [0.84, 0.86, 0.96, 0.85, 0.71],
    "map50_95": [0.57, 0.52, 0.76, 0.59, 0.49],
}

# File path
MODEL_DATA_FILE = ".streamlit/min_model_data.json"
os.makedirs(".streamlit", exist_ok=True)

# Load model data
def load_model_data():
    if os.path.exists(MODEL_DATA_FILE):
        return pd.read_json(MODEL_DATA_FILE)
    else:
        df = pd.DataFrame(default_model_data)
        df.to_json(MODEL_DATA_FILE)
        return df

# Save model data
def save_model_data(df):
    df.to_json(MODEL_DATA_FILE)

# Load existing or default data
df = load_model_data()

# Title
st.title("NTRM Object Detection Models Dashboard")
st.markdown("Minimal view: only Model Name, mAP@0.5, mAP@0.5:0.95")

# Display model data
display_df = df.copy()
display_df["map50"] = display_df["map50"].apply(lambda x: f"{x:.2f}")
display_df["map50_95"] = display_df["map50_95"].apply(lambda x: f"{x:.2f}")
display_df.columns = ["Model Name", "mAP@0.5", "mAP@0.5:0.95"]

st.dataframe(display_df, use_container_width=True)

# Sidebar: Add new model
st.sidebar.markdown("### ➕ Add New Model")

with st.sidebar.form("add_model_form"):
    model_name = st.text_input("Model Name", value="NTRM-")
    map50 = st.number_input("mAP@0.5", min_value=0.0, max_value=1.0, value=0.80, format="%.2f")
    map50_95 = st.number_input("mAP@0.5:0.95", min_value=0.0, max_value=1.0, value=0.65, format="%.2f")
    submitted = st.form_submit_button("Add Model")

    if submitted:
        if model_name:
            if model_name in df["name"].values:
                st.error("A model with this name already exists.")
            else:
                new_row = {"name": model_name, "map50": map50, "map50_95": map50_95}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                save_model_data(df)
                st.success("Model added successfully!")
                st.rerun()
        else:
            st.error("Model name is required.")

# Sidebar: Reset data
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔄 Reset to Default Data")

if st.sidebar.button("Reset Data"):
    confirm = st.sidebar.checkbox("Confirm Reset")
    if confirm:
        df = pd.DataFrame(default_model_data)
        save_model_data(df)
        st.success("Dashboard reset to default!")
        st.rerun()
    else:
        st.warning("Please confirm the reset before proceeding.")

