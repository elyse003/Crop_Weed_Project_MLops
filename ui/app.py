import streamlit as st
import requests
import pandas as pd
import os
import numpy as np

API_URL = "http://fastapi:8000"

st.title("🌿 Crop vs Weed Classifier")

tab1, tab2, tab3 = st.tabs(["Prediction", "Dashboard", "Admin"])

with tab1:
    st.header("Upload Image")
    uploaded = st.file_uploader("Choose an image", type=["jpg", "png"])
    if uploaded:
        st.image(uploaded, width=250)
        if st.button("Predict"):
            files = {"file": uploaded.getvalue()}
            res = requests.post(f"{API_URL}/predict", files=files)
            st.json(res.json())

with tab2:
    st.header("Data Insights")
    # Simple logic to count files for visualization
    try:
        crop_count = len(os.listdir('dataset/train/crop'))
        weed_count = len(os.listdir('dataset/train/weed'))
        st.bar_chart(pd.DataFrame({'Count': [crop_count, weed_count]}, index=['Crop', 'Weed']))
    except:
        st.warning("Could not access data folder directly for counting.")

with tab3:
    st.header("Retraining")
    if st.button("Trigger Retraining"):
        res = requests.post(f"{API_URL}/retrain")
        st.success(res.json()['message'])
    
    st.header("Bulk Upload")
    label = st.selectbox("Class", ["crop", "weed"])
    files = st.file_uploader("Upload new training images", accept_multiple_files=True)
    if st.button("Upload"):
        if files:
            files_list = [('files', (f.name, f.getvalue(), f.type)) for f in files]
            requests.post(f"{API_URL}/upload-data?label={label}", files=files_list)
            st.success("Uploaded!")