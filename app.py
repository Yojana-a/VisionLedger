from os import replace
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
    #asks if the image has read method
    #if its an uploaded file
    if hasattr(image_file, "read"): #hasattr is a python function to check if an obj has specific attribute
        image = Image.open(image_file)
        image_np = np.array(image) #convert image to numpy array 
        results = reader.readtext(image_np, detail=0)

        #if its a file path
    else:
        results=reader.readtext(image_file, detail=0)

    tight_text = "".join(results).replace(" ", "") #joins all the text together without spaces for the regex search or for AI to read better
    full_text=" ".join(results)# for the UI dashboard for human reading
    merchant = results[0] if results else "unknown merchant"

    detected_price = 0.0#assuming we havent found a price yet

    # STRATEGY 1: Look for "Total" line specifically (most reliable)
    # Find the LAST occurrence of standalone "Total" (not S-Total, Subtotal, Sub-Total)
    
    total_matches = []
    
    # Find all "Total" occurrences with some context before AND after
    for match in re.finditer(r'([^\n\r]{0,50})\b(?<!S-)(?<!Sub-)(?<!sub-)Total\b[:\s]*([^\n\r]{0,50})', full_text, re.IGNORECASE):
        #Capture Group 1:([^\n\r]{0,50} Grabs up to 50 characters before the word Total, but stops if it hits a new line ([^\n\r]).
        #Word Boundary:\b Ensures it matches the whole word "Total" and not "Totally" or "Subtotaling". * 0 or motre than 0 spaces or colons after Total [:s]*
        #Capture Group 2:][^\n\r]{0,50} Grabs up to 50 characters after the word Total on the same line.
        total_matches.append(match)
    
    if total_matches:
        # Take the LAST "Total" match (the final total, not subtotal)
        last_match = total_matches[-1]
        before_total = last_match.group(1).strip()
        after_total = last_match.group(2).strip()
        
        # Try to find price AFTER "Total" first (most common)
        # Pattern 1: Normal order like "20.00" or "$20.00" or "820.00"
        # Pattern 2: Reversed order like ".00 820" (MUST have 8 before the number)
        price_matches_after = list(re.finditer(r'[8$]?(\d+)[.,](\d{2})|[.,](\d{2})\s*8(\d+)', after_total))
        price_match = price_matches_after[0] if price_matches_after else None
        
        # If not found after, look BEFORE "Total" - take the LAST price (closest to Total)
        if not price_match:
            price_matches_before = list(re.finditer(r'[8$]?(\d+)[.,](\d{2})|[.,](\d{2})\s*8(\d+)', before_total))
            price_match = price_matches_before[-1] if price_matches_before else None
        
        if price_match:
            groups = price_match.groups()
            # Check which groups matched (normal order vs reversed)
            if groups[0] is not None:  # Normal: "20.00"
                dollars = groups[0]
                cents = groups[1]
            else:  # Reversed: ".00 820" (with 8)
                cents = groups[2]
                dollars = groups[3]
            
            # Remove leading '8' if it looks like a misread '$'
            if dollars.startswith('8') and len(dollars) >= 2:
                dollars = dollars[1:]
            
            detected_price = float(f"{dollars}.{cents}")
    
    # STRATEGY 2: No "Total" found - find the LARGEST price (likely the total)
    if detected_price == 0.0:
        all_prices = []
        
        # Find all normal price patterns
        for match in re.finditer(r'[8$]?(\d+)[.,](\d{2})', tight_text):
            val = match.group(0)
            # Remove $ or leading 8
            if val.startswith('$'):
                val = val[1:]
            elif val.startswith('8') and len(val) > 4:
                val = val[1:]
            
            try:
                price = float(val.replace(',', '.'))
                # Filter out unrealistic prices (too small or too large)
                if 0.50 < price < 10000:
                    all_prices.append(price)
            except ValueError:
                continue
        
        # Take the largest price (usually the total)
        if all_prices:
            detected_price = max(all_prices)
    
    # STRATEGY 3: Still nothing? Look for numbers without decimals (rare fallback)
    if detected_price == 0.0:
        all_nums = re.findall(r'(\d{3,})', tight_text)
        if all_nums:
            # Take the last number, assume it's in cents
            last_num = all_nums[-1]
            if last_num.startswith('8') and len(last_num) >= 3:
                last_num = last_num[1:]
            try:
                detected_price = float(last_num) / 100
            except ValueError:
                detected_price = 0.0

    #Enhanced price cleaning - replace '8' with '$' in the full text for display
    cleaned_text = full_text
    # Look for patterns like "8XX.XX" or "Total 8XX.XX" and suggest they might be prices
    if '8' in tight_text:
        # This helps users see where the OCR might have misread $
        cleaned_text = full_text.replace("8", "$ (or 8)")

    return {
        "merchant": merchant, 
        "text": full_text,
        "cleaned_text": cleaned_text,
        "price": detected_price
    } #returning a dictionary with all the info

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
                
                # Display the cleaned price found by your Regex logic
                # We use number_input so you can fix it if the AI is still off
                final_price = st.number_input(
                    "Verified Total",
                    value=float(data["price"]),
                    step=0.01,
                    format="%.2f",
                    key=f"price_{receipt}" # Unique key for each receipt
                )
                
                if st.button("Save Data", key=f"save_{receipt}"):
                    st.success(f"Saved ${final_price:.2f}")