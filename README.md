# 💸 VisionLedger: AI-Powered Expense Pipeline

A computer vision application that digitizes physical receipts and extracts financial data using Deep Learning.

## 🚀 Key Features
- **AI Inference:** Utilizes a PyTorch-based CRNN (EasyOCR) to extract text from images.
- **Data Sanitization:** Implemented custom Python heuristics to clean OCR artifacts (e.g., misidentified currency symbols).
- **Interactive Dashboard:** Built with Streamlit for real-time receipt analysis and visualization.

## 🛠️ Technical Stack
- **Language:** Python
- **AI Framework:** EasyOCR / PyTorch
- **UI Framework:** Streamlit
- **Environment:** Managed via Virtual Environments (.venv)

## 📋 How to Run
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt