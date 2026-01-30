import streamlit as st      # For the Web Interface & Dashboard
import easyocr              # The PyTorch-based OCR Engine (The "Real" AI)
import torch                # The Deep Learning Framework (Backend for EasyOCR)
import tensorflow as tf     # The ML Framework (For the Logic/Classification)
import pandas as pd         # For Data Manipulation
import numpy as np          # For Numerical Operations (Standard AI library)
import re                   # For "Regex" (To find prices and dates in the text)
import time                 # To simulate processing delays
from PIL import Image       # To handle and display image files properly

# 1. SETUP: High-performance caching for the AI model
@st.cache_resource
def load_ocr_model():
    # This downloads the 'Weights' (the AI's knowledge) to the computer
    return easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# Initialize the reader
reader = load_ocr_model()

def analyze_image(image_file):
    results = reader.readtext(image_file, detail=0)

    merchant= results[0] if results else "Unknown Merchant"

    price = 0.00
    for price in results:
        clean_price =  price.replace("$", "").replace(" ", "")

        if "." in clean_price and clean_price.replace(".", "").isdigit():
            price = clean_price
            
    return {"merchant": merchant, "text": " ".join(results)} # with the .join it actually gives space between merchant names

#2.Login Page
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 VisionLedger Login")
    user = st.text_input("Username")
    passw = st.text_input("Password", type="password")
    if st.button("Login"):
        if user == "admin" and passw == "1234":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Wrong password!")

#3. The DASHBOARD for the App
else:
    st.title("💸 VisionLedger Dashboard")
    
    # Add the uploader to the sidebar
    uploaded_files = st.sidebar.file_uploader("Upload Receipts", accept_multiple_files=True)
    
    # Decide which receipts to loop through
    if uploaded_files:
        receipt_list = uploaded_files
    else:
        receipt_list = ["receipt1.jpg", "receipt2.jpg"] # Make sure these exist!

    for receipt in receipt_list:
        # HERE IS THE MAGIC: We run the AI on each receipt in the loop
        with st.spinner("AI is reading pixels..."):
            data = analyze_image(receipt)

        with st.container(border=True):
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                st.image(receipt, use_container_width=True)
            
            with col2:
                # Instead of hardcoding "Starbucks", we use the AI's result!
                st.subheader(data["merchant"])
                st.write(f"**Raw AI Output:**")
                st.caption(data["text"])
                
            with col3:
                st.metric("Status", "Processed")
                st.button("Edit", key=f"edit_{receipt}")