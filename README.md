# 🔥 DCU SAD Forecasting using Machine Learning

This project predicts **Shutdown and Decoking (SAD)** requirements for the **Delayed Coker Unit (DCU)** in a refinery using **furnace operating data** and advanced **machine learning models**. The system enables chemical engineers to monitor unit health, anticipate maintenance proactively, and optimize furnace runtime with minimal disruptions.

---

## 📌 Key Features

- ⏱ Predict **Days Until Next SAD** using sensor data
- 📊 Visual Timeline: Track SAD risk over time
- 🧠 Powered by **XGBoost Regression** for robust forecasting
- 📁 Supports real-time input and batch **CSV file uploads**
- 📍 Automatic alerts for **safe**, **warning**, and **critical** zones
- 📈 Interactive visualizations of parameter trends
- 🧪 Feature drift detection between past years and 2025

---

## 🛠️ Tech Stack

| Component       | Tool/Library                |
|----------------|-----------------------------|
| Language        | Python                      |
| Dashboard       | Streamlit                   |
| ML Model        | XGBoost                     |
| Data Handling   | Pandas, NumPy               |
| Visualization   | Matplotlib, Seaborn         |
| Model I/O       | Joblib                      |
| Deployment      | Localhost / GitHub Pages    |

---

## ⚙️ How It Works

### 🔹 Data Labeling

- The model uses **historical SAD event dates (2021–2024)**.
- Each day before a SAD is labeled with **`days_to_sad`**, counting down from ~365 to 0.
- Only periods **1 year before each SAD event** are considered valid for training.

### 🔹 Model Training Pipeline

1. **Base Model (2021–2024)**:
   - Trained using XGBoost on true `days_to_sad` values.
   - Captures pre-SAD behavior from past years.

2. **Pseudo-Labeling (2025)**:
   - 2025 furnace data is unlabeled (no SAD date yet).
   - The base model predicts `pseudo_days_to_sad` on 2025 data.

3. **Retraining on 2025**:
   - Model is retrained using the pseudo-labeled 2025 data to adapt to new conditions.

4. **Deployment**:
   - A user-friendly Streamlit dashboard takes live input or CSV uploads to make predictions and visualize insights.


Model Evaluation:-

| Model Type        | Training Dataset | R² Score | RMSE (days) |
| ----------------- | ---------------- | -------- | ----------- |
| XGBoost Base      | 2021–2024        | 0.96     | 12.5        |
| XGBoost Retrained | 2025 (pseudo)    | 0.94     | 17.3        |


🧠 Alerts & Thresholds:-

| Days to SAD | Status      | Action                   |
| ----------- | ----------- | ------------------------ |
| > 90 days   | ✅ Healthy   | Normal operation         |
| 30–90 days  | 🟡 Warning  | Start monitoring closely |
| < 30 days   | 🔴 Critical | Plan shutdown soon       |

💼 Use Cases:-
📍 Shutdown Planning
Predict optimal SAD dates months in advance.

🔬 Data Drift Detection
Visualize how new furnace behavior (2025) differs from the past.

📊 Parameter Analysis
Track how individual sensor readings impact shutdown forecasts.

🧠 Future Scope:-
Live integration with refinery SCADA systems

SMS/Email alerts for threshold breaches

Integration with refinery-wide dashboards (like PI System)

Streamlit images:-
![image](https://github.com/user-attachments/assets/56ef1e0b-6989-4461-9038-2333a88472bc)
![image](https://github.com/user-attachments/assets/96598682-da66-492e-b4d4-000cc0b97b6a)
![image](https://github.com/user-attachments/assets/2aa1f099-e25c-4782-9eed-d472bde845c0)
![image](https://github.com/user-attachments/assets/5ffaf54c-23df-42b4-bfc3-d567f63ec9c9)

---

## 📝 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).  
Feel free to use, modify, and distribute it with proper attribution.

---

## 👨‍💻 Author

**Harsh Kumar**  
_Machine Learning Intern, Indian Oil Corporation Limited (IOCL)_  
📫 [Email](mailto:harshk210804@gmail.com) · 🌐 [GitHub](https://github.com/hkuser212)


> Note: This project was independently designed and executed by Harsh Kumar during his 2025 summer internship at IOCL. 
> Initial results were shared with the Chemical Engineering team for validation, and the original forecasting pipeline was developed without external codebase assistance.

