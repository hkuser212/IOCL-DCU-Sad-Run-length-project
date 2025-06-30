# dashboard.py
# --------------------------------------------
# Project: DCU Furnace SAD Forecasting System
# Author: Harsh Kumar | IOCL Summer Intern 2025
# Description: Original model design, EDA, ML pipeline, and deployment done independently by the author.
# Tools Used: Python, Pandas, XGBoost, Streamlit, SHAP
# --------------------------------------------


# -------------------- Load Regression Model --------------------
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# -------------------- Load Regression Model --------------------
#2021-2024 model and features..
model = joblib.load('model_base_2021_24.pkl')
features = joblib.load('features_base_2021_24.pkl')
#2021-2024 model and features retrianed with 2025
# model = joblib.load('model_retrained_2025.pkl')
# features = joblib.load('features_base_2021_24.pkl')
# Fix column name typo if any
features = [f.replace("FeedFlow _Pass3", "FeedFlow_Pass3") for f in features]

st.set_page_config(page_title="DCU SAD Forecast", layout="wide")
st.title("Forecast DCU Shutdown (SAD) proactively using furnace telemetry data — powered by Machine Learning")
st.markdown("Predict **Days Until Next SAD** using real-time or uploaded furnace parameters")

# -------------------- Realtime Input Form --------------------
st.sidebar.header("📅 Enter Real-Time Furnace Readings")
input_data = {}
for feature in features:
    input_data[feature] = st.sidebar.number_input(f"{feature}", value=0.0, step=0.01)

if st.sidebar.button("🔮 Predict Using Realtime Input"):
    df_input = pd.DataFrame([input_data])
    predicted_days = model.predict(df_input)[0]
    sad_date = pd.Timestamp.today() + pd.Timedelta(days=predicted_days)

    st.success(f"⏰ Estimated Days Until SAD: **{predicted_days:.2f} days**")
    st.info(f"📍 Estimated SAD Date: **{sad_date:%d-%b-%Y}**")

    if predicted_days > 90:
        st.markdown("✅ **Safe Operation: No immediate SAD needed**")
    elif predicted_days > 30:
        st.markdown("🟡 **Monitor Conditions: SAD could occur in a month**")
    else:
        st.markdown("🔴 **Urgent: SAD may be required soon!**")
#
# # -------------------- CSV Upload Section --------------------
st.header("📁 Upload Furnace CSV Data")
uploaded_file = st.file_uploader("Upload a .csv file with furnace readings", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, header=4)


    st.subheader("📄 Raw Data Preview")
    st.write(df.head())


    column_mapping = {
        'Timestamp': "timestamp",
        'KG/CM2': "Coil_Inlet_Pressure_Pass1",
        'KG/CM2.1': "Coil_Inlet_Pressure_Pass2",
        'KG/CM2.2': "Coil_Inlet_Pressure_Pass3",
        'KG/CM2.3': "Coil_Inlet_Pressure_Pass4",
        'Unnamed: 7': 'FeedFlow_Pass1',
        'Unnamed: 8': 'FeedFlow_Pass2',
        'Unnamed: 9': 'FeedFlow_Pass3',
        'Unnamed: 10': 'FeedFlow_Pass4',
        'DEGC': 'COT_Pass1',
        'DEGC.1': 'COT_Pass2',
        'DEGC.2': 'COT_Pass3',
        'DEGC.3': 'COT_Pass4',
        'KG/CM2.4': 'COIL_dP_Pass1',
        'KG/CM2.5': 'COIL_dP_Pass2',
        'KG/CM2.6': 'COIL_dP_Pass3',
        'KG/CM2.7': 'COIL_dP_Pass4',
        'KG/HR': 'BFW_Injection_Pass1',
        'KG/HR.1': 'BFW_Injection_Pass2',
        'KG/HR.2': 'BFW_Injection_Pass3',
        'KG/HR.3': 'BFW_Injection_Pass4',
        'DEGC.4': 'MAX_SkinTemp_Pass1',
        'DEGC.5': 'Max_SkinTemp_Pass2',
        'DEGC.6': 'Max_SkinTemp_Pass3',
        'DEGC.7': 'Max_SkinTemp_Pass4',
    }
    df.rename(columns=column_mapping, inplace=True)
    print("✅ Available columns after rename:")
    print(df.columns.tolist())
    # Rename columns
    df.rename(columns=column_mapping, inplace=True)

    # Ensure required features are present
    missing = [f for f in features if f not in df.columns]
    if missing:
        st.error(f"❌ Missing required columns: {missing}")
    else:
        # Clean and predict
        for f in features:
            df[f] = pd.to_numeric(df[f], errors='coerce')

        df.dropna(subset=features, inplace=True)
        df['Timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=False, errors='coerce')
        st.write("🟢 Valid timestamps:", df['timestamp'].min(), "to", df['timestamp'].max())
        st.write("🔢 Rows retained after cleanup:", df.shape[0])

        if df.empty:
            st.warning("⚠️ No valid rows after cleaning. All rows had missing values.")
        else:
            # Make prediction
            df['Predicted_Days_to_SAD'] = model.predict(df[features])

            # Plotting
            df.set_index('timestamp', inplace=True)
            fig, ax = plt.subplots(figsize=(15, 6))
            ax.plot(df.index, df['Predicted_Days_to_SAD'], label='Predicted Days to SAD', color='blue')
            ax.axhline(30, color='orange', linestyle='--', label='Warning Threshold')
            ax.axhline(10, color='red', linestyle='--', label='Critical Threshold')
            ax.set_ylabel('Days to SAD')
            ax.set_title('SAD Prediction Timeline')
            ax.legend()
            ax.grid(True)
            # Improve x-axis readability
            ax.xaxis.set_major_locator(plt.MaxNLocator(10))  # Limit number of ticks
            fig.autofmt_xdate(rotation=45)  # Rotate dates

            st.pyplot(fig)
# parament
            selected_param = st.selectbox("📈 Select Parameter to Plot:", features)

            fig, ax1 = plt.subplots(figsize=(15, 5))
            ax1.plot(df.index, df[selected_param], color='green', label=selected_param)
            ax1.set_ylabel(selected_param, color='green')

            ax2 = ax1.twinx()
            ax2.plot(df.index, df['Predicted_Days_to_SAD'], color='blue', alpha=0.6, label='Predicted Days to SAD')
            ax2.set_ylabel('Predicted Days to SAD', color='blue')

            fig.legend(loc='upper right')

            st.pyplot(fig)

            import seaborn as sns



            importances = model.feature_importances_  # for XGBoost, RandomForest
            feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
            feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)



            # Plot
            import seaborn as sns

            plt.figure(figsize=(10, 6))
            sns.barplot(data=feat_imp_df, x='Importance', y='Feature', palette='viridis')
            plt.title("Top Features Affecting SAD Prediction")
            st.pyplot(plt)

            st.header("📊 Feature Drift Analysis (2021–2024 vs 2025)")
            st.markdown(
                "This helps you visualize how key furnace parameters have shifted between older data and 2025 conditions.")

            df_labeled = pd.read_csv('merged-csv-files.csv',header=4 , skiprows=[0, 1, 2, 3])
            df_2025 = pd.read_csv('DCU - 2025.csv',header=4 , skiprows=[0, 1, 2, 3])
            # Optional checkbox to toggle visibility
            if st.checkbox("📈 Show Feature Distribution Comparison"):
                st.markdown(
                    "Visual comparison of selected parameters between **2021–2024** and **2025** to detect data drift.")

                for col in features[:6]:  # First 6 key features
                    fig, ax = plt.subplots(figsize=(10, 4))

                    # Plot KDE for historical 2021–24 data
                    sns.kdeplot(df_labeled[col], label='2021–24', fill=True, ax=ax, color='green')

                    # Plot KDE for 2025 data
                    sns.kdeplot(df_2025[col], label='2025', fill=True, ax=ax, color='blue')

                    ax.set_title(f'Distribution Shift: {col}')
                    ax.set_xlabel(col)
                    ax.set_ylabel("Density")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)

            # Custom inputs
            input_data = {}
            for feat in features:
                input_data[feat] = st.number_input(f"{feat}", value=100.0, format="%.2f")

            if st.button("🔮 Predict for Custom Inputs"):
                pred_df = pd.DataFrame([input_data])
                pred_days = model.predict(pred_df)[0]
                sad_date = pd.Timestamp.today() + pd.Timedelta(days=pred_days)
                st.info(f"📍 Estimated SAD Date: **{sad_date:%d-%b-%Y}**")

            # 1. Clean and process all rows
            df['Predicted_Days_to_SAD'] = model.predict(df[features])


            # 2️⃣ Get earliest warning row
            # Apply rolling average with a window of 12
            # RAW CRITICAL METHOD
            min_pred_raw = df['Predicted_Days_to_SAD'].min()
            min_row_raw = df[df['Predicted_Days_to_SAD'] == min_pred_raw].iloc[0]

            if 'timestamp' in df.columns:
                min_ts_raw = pd.to_datetime(min_row_raw['timestamp'], errors='coerce')
            else:
                min_ts_raw = pd.to_datetime(min_row_raw.name, errors='coerce')

            est_sad_raw = min_ts_raw + pd.Timedelta(days=min_pred_raw)

            df['Rolling_SAD'] = df['Predicted_Days_to_SAD'].rolling(window=12, min_periods=1).mean()

            min_pred = df['Rolling_SAD'].min()
            min_index = df['Rolling_SAD'].idxmin()
            min_row = df.loc[min_index]
            min_index_dt = pd.to_datetime(min_index)

            # Extract timestamp from column or index
            if 'timestamp' in df.columns:
                min_ts = pd.to_datetime(min_row['timestamp'], errors='coerce')
            else:
                min_ts = pd.to_datetime(min_row.name, errors='coerce')

            # Validate earliest timestamp
            if pd.isna(min_ts):
                st.error("❌ Earliest timestamp (for SAD) is not valid.")
            else:
                est_sad_from_min = min_ts + pd.Timedelta(days=min_pred)

                # Display alert
                if min_pred > 250:
                    st.success("✅ DCU is operating in a healthy range. No SAD required soon.")
                elif min_pred > 90:
                    st.warning("🟡 Conditions are stable, but monitor for trends. SAD may approach within 2–3 months.")
                else:
                    st.error("🔴 Alert: SAD likely required soon! Take action.")
                st.info(f"📉 Minimum (smoothed) Days to SAD: **{min_pred:.1f}** (on {min_index_dt:%d-%b-%Y})")
                st.success(f"📍 Estimated SAD Date (Rolling Avg): **{est_sad_from_min.strftime('%d-%b-%Y')}**")
                st.subheader("🔴 Critical Risk Estimate (Raw Minimum)")
                st.info(f"📉 Most Critical Prediction: **{min_pred_raw:.1f} days to SAD** (from {min_ts_raw:%d-%b-%Y})")
                st.warning(f"📍 Estimated SAD Date (Raw): **{est_sad_raw:%d-%b-%Y}**")

                # column_mapping = {
                #   'Timestamp': "timestamp",
                #   'gwa.03Pi602a1.pv - Average': "Coil_Inlet_Pressure_Pass1",
                #   'gwa.03Pi602a2.pv - Average': "Coil_Inlet_Pressure_Pass2",
                #   'gwa.03Pi602B1.pv - Average': "Coil_Inlet_Pressure_Pass3",
                #   'gwa.03pi602B2.pv - Average': "Coil_Inlet_Pressure_Pass4",
                #   'gwa.03fic601a1.pv - Average': 'FeedFlow_Pass1',
                #     'gwa.03fic601a2.pv - Average': 'FeedFlow_Pass2',
                #     'gwa.03fic601b1.pv - Average': 'FeedFlow_Pass3',
                #     'gwa.03fic601b2.pv - Average': 'FeedFlow_Pass4',
                #     'gwa.03tic606a1.pv - Average': 'COT_Pass1',
                #     'gwa.03tic606a2.pv - Average': 'COT_Pass2',
                #     'gwa.03tic606b1.pv - Average': 'COT_Pass3',
                #     'gwa.03tic606b2.pv - Average': 'COT_Pass4',
                #     'gwa.03dpi613a1.pv - Average': 'COIL_dP_Pass1',
                #     'gwa.03dpi613a2.pv - Average': 'COIL_dP_Pass2',
                #     'gwa.03dpi613b1.pv - Average': 'COIL_dP_Pass3',
                #     'gwa.03dpi613b2.pv - Average': 'COIL_dP_Pass4',
                #     'gwa.03fic602a1.pv - Average': 'BFW_Injection_Pass1',
                #     'gwa.03fic602a2.pv - Average': 'BFW_Injection_Pass2',
                #     'gwa.03fic602b1.pv - Average': 'BFW_Injection_Pass3',
                #     'gwa.03fic602b2.pv - Average': 'BFW_Injection_Pass4',
                #     'gwa.03ti610g1.pv - Average': 'MAX_SkinTemp_Pass1',
                #     'gwa.03ti611g1.pv - Average': 'Max_SkinTemp_Pass2',
                #     'gwa.03ti612g1.pv - Average': 'Max_SkinTemp_Pass3',
                #     'gwa.03ti613g1.pv - Average': 'Max_SkinTemp_Pass4',
                # }

# # dashboard.py
#
# import streamlit as st
# import pandas as pd
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import json
#
# # Load model and feature names
# model = joblib.load('regression_model_2022.pkl')
# features = joblib.load('feature_columns_2022.pkl')
#
# # Optional: load saved mapping from file
# # try:
# #     with open('column_mapping.json', 'r') as f:
# #         saved_mapping = json.load(f)
# # except FileNotFoundError:
# #     saved_mapping = {}
#
# st.title("🔥 IOCL Furnace Optimization Dashboard")
# st.markdown("Predict Shutdown Risk (SAD) and Optimize Temperature")
#
# # Input form
# st.sidebar.header("📥 Input Parameters")
# input_data = {}
# for feature in features:
#     input_data[feature] = st.sidebar.number_input(f"{feature}", value=0.0)
#
# # Predict
# if st.sidebar.button("Predict"):
#     df_input = pd.DataFrame([input_data])
#     prediction = model.predict(df_input)[0]
#     st.success(f"Predicted Output: {prediction:.2f}")
#     if prediction < 24:
#         st.markdown("🟢 **Safe Operating Condition**")
#     else:
#         st.markdown("🔴 **High Risk: SAD Likely**")
#
# # Upload & Visualize
# st.header("📊 Upload Furnace Data (CSV)")
# uploaded_file = st.file_uploader("Upload furnace CSV", type=["csv"])
#
# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file, skiprows=4)  # Adjust as needed
#     st.subheader("🧾 Raw Uploaded Data")
#     st.write(df.head())
#
#     # Clean column names
#     df.columns = df.columns.str.strip().str.lower().str.replace(r"[^\w]", "", regex=True)
#
#     # Apply saved or hardcoded mapping
#     column_mapping = {
#                 'Unnamed: 7': 'FeedFlow_Pass1',
#                 'Unnamed: 8': 'FeedFlow_Pass2',
#                 'Unnamed: 9': 'FeedFlow_Pass3',
#                 'Unnamed: 10': 'FeedFlow_Pass4',
#                 'DEGC': 'COT_Pass1',
#                 'DEGC.1': 'COT_Pass2',
#                 'DEGC.2': 'COT_Pass3',
#                 'DEGC.3': 'COT_Pass4',
#                 'DEGC.4': 'MAX_SkinTemp_Pass1',
#                 'DEGC.5': 'Max_SkinTemp_Pass2',
#                 'DEGC.6': 'Max_SkinTemp_Pass3',
#                 'DEGC.7': 'Max_SkinTemp_Pass4',
#                 'KG/HR': 'BFW_Injection_Pass1',
#                 'KG/HR.1': 'BFW_Injection_Pass2',
#                 'KG/HR.2': 'BFW_Injection_Pass3',
#                 'KG/HR.3': 'BFW_Injection_Pass4',
#                 'KG/CM2.5': 'COIL_dP_Pass2',
#                 'KG/CM2.3': 'Coil_Inlet_Pressure_Pass4'
#             }
#     # column_mapping.update(saved_mapping)
#
#     # Try applying the mapping
#     df.rename(columns=column_mapping, inplace=True)
#
#     # Capitalize normalized names to match model features
#     df.columns = [col.upper() for col in df.columns]
#
#     # Handle missing features
#     available_features = [f for f in features if f.upper() in df.columns]
#     missing_features = [f for f in features if f.upper() not in df.columns]
#
#     if missing_features:
#         st.warning(f"⚠️ Missing or unmatched columns: {missing_features}")
#         st.subheader("🛠️ Manual Column Mapping")
#         for f in missing_features:
#             selected_col = st.selectbox(f"Select CSV column for: **{f}**", df.columns, key=f)
#             df.rename(columns={selected_col: f}, inplace=True)
#
#         # Save updated mapping for future use
#         # new_mapping = {col: f for f in missing_features for col in df.columns if col.upper() == f.upper()}
#         # saved_mapping.update(new_mapping)
#         # with open('column_mapping.json', 'w') as f:
#         #     json.dump(saved_mapping, f)
#
#     # Correlation Heatmap
#     st.subheader("📊 Correlation Heatmap")
#     fig, ax = plt.subplots()
#     try:
#         sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm', ax=ax)
#         st.pyplot(fig)
#     except KeyError as e:
#         st.error(f"Some features still not found in DataFrame: {e}")
#
#     # Time Series Trend
#     st.subheader("📈 Time Series Trend")
#     if 'timestamp' in df.columns:
#         df['timestamp'] = pd.to_datetime(df['timestamp'])
#         df.set_index('timestamp', inplace=True)
#         st.line_chart(df[available_features])
#     else:
#         st.info("No 'timestamp' column found. Skipping time series plot.")

