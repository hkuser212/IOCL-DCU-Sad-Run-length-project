# 🔥 DCU SAD Forecasting using Machine Learning

This project forecasts **Shutdown and Decoking (SAD)** requirements for the **Delayed Coker Unit (DCU)** in a refinery, using **real-time furnace sensor data** and Machine Learning (XGBoost). The tool helps chemical engineers monitor conditions, anticipate shutdowns, and reduce operational risks.


---

## 📌 Features

- 📊 Predict **Days Until Next SAD**
- 🧠 Uses XGBoost and LSTM for accurate time-series forecasting
- 📈 Visual Timeline of SAD risk over time
- 🔍 Feature Importance & Parameter Drift Analysis (2021–2025)
- ⏱ Real-time input form and batch CSV upload support
- 📍 Alerts for Critical, Warning, and Safe zones

---

## 🛠 Tech Stack

| Component | Tools Used |
|----------|-------------|
| Language | Python |
| Dashboard | Streamlit |
| ML Models | XGBoost, LSTM (Keras/TensorFlow) |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Model Serialization | Joblib, H5 |
| Version Control | Git & GitHub |

---

---

## ⚙️ How It Works

1. **Data Preprocessing**
   - Historical SAD windows are labeled using known shutdown/feed-cut dates.
   - Furnace parameters are cleaned and normalized.

2. **Base Model Training**
   - XGBoost is trained on 2021–2024 data to learn patterns before each SAD.

3. **Pseudo-labeling for 2025**
   - 2025 data (Jan–June) is pseudo-labeled using the base model.
   - Retrained model adapts to 2025 operating behavior.

4. **Dashboard**
   - Interactive web app built with Streamlit.
   - Supports live sensor input and CSV batch prediction.

5. **Forecast**
   - Predicts `Days_to_SAD`, plots trends, alerts engineers, and shows likely SAD dates.

---

## 🚀 Quick Start

### ✅ 1. Clone the Repository

```bash
git clone https://github.com/your-username/dcu-sad-forecast.git
cd dcu-sad-forecast
✅ 2. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
⚠️ Also install TensorFlow if using LSTM:

bash
Copy
Edit
pip install tensorflow
✅ 3. Run the Dashboard
bash
Copy
Edit
streamlit run dashboard.py

💼 Use Cases
📍 Shutdown Planning
Predict optimal SAD dates months in advance.

🔬 Data Drift Detection
Visualize how new furnace behavior (2025) differs from the past.

📊 Parameter Analysis
Track how individual sensor readings impact shutdown forecasts.
