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
import plotly.express as px
import pickle

# Page configuration
st.set_page_config(
    page_title="VisionLedger - AI Receipt Analyzer",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
            <style>
                /* Main app background - Light coffee gradient */
                .stApp {
                    background: linear-gradient(135deg, #C7B7A3 0%, #E8D8C4 50%, #F5F0E8 100%);
                }
    
                /* Sidebar styling - Dark for contrast */
                section[data-testid="stSidebar"] {
                    background: linear-gradient(180deg, #561C24 0%, #6D2932 100%);
                }
                section[data-testid="stSidebar"] * {
                    color: #E8D8C4 !important;
                }
    
                /* Header styling - Dark text on light background */
                .main-header {
                    font-size: 2.5rem;
                    font-weight: bold;
                    color: #561C24;
                    margin-bottom: 0;
                    text-shadow: 1px 1px 2px rgba(255,255,255,0.5);
                }
                .sub-header {
                    font-size: 1.2rem;
                    color: #6D2932;
                    margin-top: 0;
                }
            
                /* Receipt card styling - White for maximum contrast */
                .receipt-card {
                    background: white;
                    padding: 20px;
                    border-radius: 15px;
                    box-shadow: 0 8px 16px rgba(86, 28, 36, 0.2);
                    margin-bottom: 20px;
                    border: 2px solid #C7B7A3;
                }

                /* Category badge - for confidence showcase*/
                .category-badge-high {
                    background: linear-gradient(135deg, #6D2932 0%, #561C24 100%);
                    color: #E8D8C4;
                    padding: 8px 20px;
                    border-radius: 25px;
                    font-size: 0.95rem;
                    display: inline-block;
                    font-weight: 600;
                    box-shadow: 0 2px 8px rgba(86, 28, 36, 0.4);
                }
                .category-badge-medium {
                    background: linear-gradient(135deg, #A0826D 0%, #8B6F47 100%);
                    color: white;
                    padding: 8px 20px;
                    border-radius: 25px;
                    font-size: 0.95rem;
                    display: inline-block;
                    font-weight: 600;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                }
                 
                /* Button styling */
                .stButton>button {
                    background: linear-gradient(135deg, #6D2932 0%, #561C24 100%);
                    color: #E8D8C4;
                    border: 2px solid #C7B7A3;
                    border-radius: 10px;
                    padding: 12px 28px;
                    font-weight: 600;
                    transition: all 0.3s;
                    box-shadow: 0 4px 12px rgba(86, 28, 36, 0.3);
                }
                .stButton>button:hover {
                    background: linear-gradient(135deg, #561C24 0%, #6D2932 100%);
                    box-shadow: 0 6px 20px rgba(86, 28, 36, 0.5);
                    transform: translateY(-2px);
                    border-color: #E8D8C4;
                }

                /* Metrics styling - Dark text on light background */
                div[data-testid="stMetricValue"] {
                    font-size: 2rem;
                    font-weight: bold;
                    color: #561C24;
                    text-shadow: 1px 1px 2px rgba(255,255,255,0.3);
                }
                div[data-testid="stMetricLabel"] {
                    color: #6D2932;
                    font-size: 1rem;
                    font-weight: 500;
                }
                
                /* Input fields */
                .stNumberInput input {
                    background: white;
                    border: 2px solid #C7B7A3;
                    border-radius: 8px;
                    color: #561C24;
                }
    
                /* Checkboxes */
                .stCheckbox {
                    color: #561C24;
                }
            
                /* Horizontal rule */
                hr {
                    border-color: #C7B7A3;
                    opacity: 0.3;
                }
            
                /* File uploader */
                section[data-testid="stFileUploader"] {
                    background: rgba(255, 255, 255, 0.5);
                    border: 2px dashed #C7B7A3;
                    border-radius: 10px;
                    padding: 20px;
                }
            
                /* Dataframe styling */
                .stDataFrame {
                    border: 2px solid #C7B7A3;
                    border-radius: 8px;
                }
                
            </style>
            """, unsafe_allow_html=True)


# 1. SETUP: High-performance caching for the AI model
@st.cache_resource
def load_ocr_model():
    # This downloads the 'Weights' (the AI's knowledge) to the computer
    return easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# Initialize the reader
reader = load_ocr_model()

@st.cache_resource
def load_category_model():
    #Load the trained ML categorization model
    try:
        with open('category_model.pkl', 'rb') as f: #pkl, jpg, pdf uses read binary(rb)
            vectorizer, model = pickle.load(f)
        return vectorizer, model
    except FileNotFoundError:
        st.error("Model not found! Please run 'python trainModel.py' first.")
        st.stop()
    
def categorize_merchant(merchant_name):
    #use trained model to predict category from merchant name
    #merchant_name the merchant/store name from ocr
    #returns tuple:(category,confidence) eg("food&dining", 0.85)
    try:
        vectorizer, model = load_category_model()

        X=vectorizer.transform([merchant_name]) #Convert merchnat name to numerical values
        category=model.predict(X)[0]#predict category
        confidence=model.predict_proba(X).max()
        return category, confidence
    
    except Exception as e:
        return "Other", 0.0
    #fallback if model fails
    
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
        before_total = last_match.group(1).strip()#to clean captured chunk of text
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

    #categorize merchnat using ML
    category, confidence = categorize_merchant(merchant)

    return {
        "merchant": merchant, 
        "text": full_text,
        "cleaned_text": cleaned_text,
        "price": detected_price,
        "category": category,
        "confidence": confidence
    } #returning a dictionary with all the info

#2.Login Page
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_db" not in st.session_state:
    st.session_state.user_db = {"admin": "1234"} # Keep your default admin

if not st.session_state.logged_in:
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.markdown("<h1 style='text-align: center; color: #1f77b4;'>💸 VisionLedger</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #666;'>AI-Powered Receipt Analyzer</p>", unsafe_allow_html=True)

        # CHANGE STARTS HERE: Use tabs for Login and Registration
        tab_login, tab_reg = st.tabs(["🔐 Login", "📝 Register"])

        with tab_login:
            user = st.text_input("Username", placeholder="Enter username", key="login_user")
            passw = st.text_input("Password", type="password", placeholder="Enter password", key="login_pass")
            
            if st.button("Login", use_container_width=True):
                # Check against our dynamic user_db instead of hardcoded strings
                if user in st.session_state.user_db and st.session_state.user_db[user] == passw:
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Invalid credentials!")

        with tab_reg:
            st.markdown("### Create New Account")
            new_user = st.text_input("Choose Username", placeholder="e.g. JohnDoe", key="reg_user")
            new_pass = st.text_input("Choose Password", type="password", key="reg_pass")
            
            if st.button("Create Account", use_container_width=True):
                if new_user in st.session_state.user_db:
                    st.error("Username already exists!")
                elif new_user and new_pass:
                    st.session_state.user_db[new_user] = new_pass
                    st.success("Registration successful! Go to Login tab.")
                else:
                    st.warning("Please fill out both fields.")

#3. The DASHBOARD for the App
else:
    # Header
    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown("<h1 class='main-header'>💸 VisionLedger Dashboard</h1>", unsafe_allow_html=True)
        st.markdown("<p class='sub-header'>AI-powered receipt analysis and expense tracking</p>", unsafe_allow_html=True)
    with col2:
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.rerun()
    
    # Add the uploader to the sidebar
    with st.sidebar:
        st.markdown("### 📤 Upload Receipts")
        uploaded_files = st.sidebar.file_uploader("Upload Receipts", accept_multiple_files=True)
    
    # Decide which receipts to loop through
    if uploaded_files:
        receipt_list = uploaded_files
    else:
        receipt_list = ["receipt1.png"]

    if "ledger" not in st.session_state:
        st.session_state.ledger=pd.DataFrame(columns=["Merchant","Category","Amount"])

    
    all_data=[] #stores all receipt data
    category_totals={} #creating a dict

    for receipt in receipt_list:
        #We run the AI on each receipt in the loop
        with st.spinner("Loading..."):
            receipt_data = analyze_image(receipt)

            #add to category totals
            cat=receipt_data["category"]
            category_totals[cat]=category_totals.get(cat,0)+ receipt_data["price"] #finding total amount in a particular category
        
        with st.container(border=True):
            st.markdown("<div class='receipt-card'>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1.2, 2.5, 1.5])
            
            with col1:
                st.image(receipt, width='stretch')
            
            with col2:
                # Category badge
                confidence_color = "#4caf50" if receipt_data['confidence'] > 0.7 else "#ff9800" if receipt_data['confidence'] > 0.4 else "#f44336"
                st.markdown(f"""
                <div style='margin: 10px 0;'>
                    <span style='background: {confidence_color}; color: white; padding: 5px 15px; border-radius: 20px; font-size: 1.5rem;'>
                        📂 {receipt_data['category']} ({receipt_data['confidence']*100:.0f}% confident)
                    </span>
                </div>
                """, unsafe_allow_html=True)

                show_details= st.checkbox("Show raw AI output",  key=f"details_{receipt}") # helps us differentiate between two or more receipts

                if show_details:
                    st.write("**Raw AI Output:**")
                    st.caption(receipt_data["text"])
                else:
                    st.markdown("<span style='font-size:1rem; color:gray;'>Click to view AI output</span>",
                    unsafe_allow_html=True)
                    #st.write("Click to view AI output")
                    
            with col3:
                st.markdown("### 💰 Amount")
                st.metric("Status", "Processed")
                
                # Display the cleaned price found by your Regex logic
                # We use number_input so you can fix it if the AI is still off
                final_price = st.number_input(
                    "Verified Total",
                    value=float(receipt_data["price"]),
                    step=0.01,
                    format="%.2f",
                    key=f"price_{receipt}", # Unique key for each receipt
                    label_visibility="collapsed"
                )
                
                if st.button("Save Data", key=f"save_{receipt}"):
                    category_totals[cat] = category_totals.get(cat, 0) + final_price

                    new_entry=pd.DataFrame([{
                        "Merchant":receipt_data["merchant"],
                        "Category":cat,
                        "Amount":final_price
                    }])

                    st.session_state.ledger = pd.concat([st.session_state.ledger, new_entry], ignore_index=True) #saving to permanent session state
                    st.success(f"Saved ${final_price:.2f} to ledger")
    
    #only show if category_totals{} is not empty
    if not st.session_state.ledger.empty:
        st.subheader("**Spending by category**")
        
        # We create df_chart from the LEDGER so it remembers all saved receipts
        df_chart = st.session_state.ledger.groupby("Category")["Amount"].sum().reset_index()
        #it groups all the same category together and reset takes the column and rows back to regular columns
        df_chart = df_chart.sort_values('Amount', ascending=False)

        col_left, col_right = st.columns([1,1])

        with col_left:
            #4 metric with 2x2 grid
            metric_col1, metric_col2 = st.columns(2)

            with metric_col1:
                st.metric("Total Spending", 
                          f"${df_chart['Amount'].sum():.2f}",
                          help="Sum of all receipts") #gives total spending by adding amounts
                st.metric("Top Category",
                          df_chart.iloc[0]['Category'])
            
            with metric_col2:
                st.metric("Receipts Processed", len(receipt_list))
        
        # Create and display pie chart
        with col_right:
            fig = px.pie(
                df_chart,
                values='Amount',
                names='Category',
                title='Spending Piechart',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(
                    showlegend=True,
                    height=400,
                    font=dict(size=14),
                    title_font_size=18
                )
            st.plotly_chart(fig, width='stretch')