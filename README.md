# 💸 VisionLedger: AI-Powered Expense Pipeline
VisionLedger is a full-stack AI application that automates the transition from physical receipts to structured financial data. It combines Computer Vision (OCR), Natural Language Processing (NLP), and a dynamic web dashboard to provide users with instant spending analytics.

 | 
## 🛠️ The Tech Stack
**AI/ML:** EasyOCR (PyTorch-based text detection), Scikit-Learn (Random Forest classification), TF-IDF Vectorization.

**Backend/Interface:** Streamlit for real-time model serving and session state management.

**Data Science:** Pandas for ledger management, Plotly for interactive financial visualization.

**Deployment:** Version controlled via Git, deployed on Streamlit Community Cloud.

## 🚀 Key Features
- **AI Inference:** Utilizes a PyTorch-based CRNN (EasyOCR) to extract text from images.
- **Contextual Regex:** Scans for "Total" keywords and captures high-probability price patterns in surrounding text.
- **Fallback Logic:** If keywords are missing, the system identifies the largest numerical value as the likely total, filtering for realistic price ranges.
- **Interactive Dashboard:** Built with Streamlit for real-time receipt analysis and visualization.
- **Cleaning:** Implemented text normalization to handle OCR misreads (e.g., converting '8' characters to '$' where appropriate).
- **Problem Solved:** Reduces manual data entry time for expense tracking by approximately 80%.

## Synthetic Data Generation
**Generation:** Utilized LLMs (Claude 3.5) to bootstrap a dataset of 1,000+ labeled merchant entries.
**Normalization:** Implemented a pre-processing script to sanitize text, removing store numbers and special characters to improve model generalization.

## Future Roadmap
**Transformer Migration:** Transitioning classification from Random Forest to a Fine-tuned BERT model for enhanced semantic accuracy.
**Export Integration:** Adding API support for direct export to Google Sheets for professional accounting.


## 📋 How to Run
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
