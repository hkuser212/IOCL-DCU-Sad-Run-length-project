# # import pandas as pd
# #
# # df = pd.read_csv('DCU - 2021.csv', header=4)
# # print(df.columns)
# # df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)
# # df.columns = df.columns.str.strip().str.lower()  # removes whitespace, lowercase
# #
# # if 'timestamp' in df.columns:
# #     print("✅ Found 'timestamp' column")
# # else:
# #     print("❌ 'timestamp' column still missing!")
# #
# # print(df.columns.tolist())
# #
# #
# # print("\n📌 First 10 rows of raw file (to check where real headers/data start):")
# # print(df.head(10))
# # # Print actual columns detected
# # print("\n📋 Actual columns in file:")
# # print(list(df.columns))
# #
# # # Your column mapping
# # column_mapping = {
# #     'timestamp': "timestamp",
# #     'KG/CM2': "Coil_Inlet_Pressure_Pass1",
# #     'Unnamed: 4': "Coil_Inlet_Pressure_Pass2",
# #     'Unnamed: 5': "Coil_Inlet_Pressure_Pass3",
# #     # ... (rest of your mapping)
# # }
# #
# # # Columns in CSV that are mapped
# # matched_columns = [col for col in column_mapping if col in df.columns]
# # missing_in_file = [col for col in column_mapping if col not in df.columns]
# # extra_in_file = [col for col in df.columns if col not in column_mapping]
# #
# # print("\n✅ Mapped columns found in file:")
# # print(matched_columns)
# #
# # print("\n❌ Missing columns (defined in mapping but not found in file):")
# # print(missing_in_file)
# #
# # print("\n⚠️ Extra columns present in file but not mapped:")
# # print(extra_in_file)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # Step 1: Load Data
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score
#
# # Step 1: Load Data
# df_raw = pd.read_csv('merged-csv-files.csv', header=4, skiprows=[0, 1, 2, 3])
# print("📄 Raw data loaded. Shape:", df_raw.shape)
#
# # Step 2: Rename Columns
# renamed_columns = {
#     df_raw.columns[2]: "timestamp",
#     df_raw.columns[3]: "Coil_Inlet_Pressure_Pass1",
#     df_raw.columns[4]: "Coil_Inlet_Pressure_Pass2",
#     df_raw.columns[5]: "Coil_Inlet_Pressure_Pass3",
#     df_raw.columns[6]: "Coil_Inlet_Pressure_Pass4",
#     df_raw.columns[7]: "FeedFlow_Pass1",
#     df_raw.columns[8]: "FeedFlow_Pass2",
#     df_raw.columns[9]: "FeedFlow_Pass3",
#     df_raw.columns[10]: "FeedFlow_Pass4",
#     df_raw.columns[11]: "COT_Pass1",
#     df_raw.columns[12]: "COT_Pass2",
#     df_raw.columns[13]: "COT_Pass3",
#     df_raw.columns[14]: "COT_Pass4",
#     df_raw.columns[15]: "COIL_dP_Pass1",
#     df_raw.columns[16]: "COIL_dP_Pass2",
#     df_raw.columns[17]: "COIL_dP_Pass3",
#     df_raw.columns[18]: "COIL_dP_Pass4",
#     df_raw.columns[19]: "BFW_Injection_Pass1",
#     df_raw.columns[20]: "BFW_Injection_Pass2",
#     df_raw.columns[21]: "BFW_Injection_Pass3",
#     df_raw.columns[22]: "BFW_Injection_Pass4",
#     df_raw.columns[23]: "MAX_SkinTemp_Pass1",
#     df_raw.columns[24]: "Max_SkinTemp_Pass2",
#     df_raw.columns[25]: "Max_SkinTemp_Pass3",
#     df_raw.columns[26]: "Max_SkinTemp_Pass4",
# }
# df_renamed = df_raw.rename(columns=renamed_columns)
#
# # Step 3: Clean the Data
# df_cleaned = df_renamed.drop(columns=[df_raw.columns[0], df_raw.columns[1]], errors='ignore')
# df_cleaned['timestamp'] = pd.to_datetime(df_cleaned['timestamp'], errors='coerce')
# df_cleaned = df_cleaned.dropna(subset=['timestamp']).reset_index(drop=True)
# print("✅ Cleaned data shape:", df_cleaned.shape)
#
# # Step 4: SAD + Feed Cut Dates
# sad_starts = [
#     '2021-05-03',
#     '2022-11-23',
#     '2023-10-04',
#     '2024-12-14'
# ]
# feed_cuts = [
#     '2021-05-30',
#     '2022-12-05',
#     '2023-11-18',
#     '2024-11-29'  # assumed correct
# ]
# sad_feed_blocks = [(pd.to_datetime(sad), pd.to_datetime(feed)) for sad, feed in zip(sad_starts, feed_cuts)]
#
# # Step 5: Initialize label
# df_cleaned['days_to_sad'] = np.nan
# rows_to_keep = pd.Series([False] * len(df_cleaned), index=df_cleaned.index)  # Default False
#
# # Step 6: Assign labels only in valid pre-SAD windows per cycle
# for sad_start, feed_cut in sad_feed_blocks:
#     pre_sad_mask = (df_cleaned['timestamp'] < sad_start) & (df_cleaned['timestamp'] > feed_cut - pd.Timedelta(days=365))
#     df_cleaned.loc[pre_sad_mask, 'days_to_sad'] = (sad_start - df_cleaned.loc[pre_sad_mask, 'timestamp']).dt.days
#     rows_to_keep |= pre_sad_mask  # Mark valid rows
#
# # Apply row filter (keep only rows in pre-SAD windows)
# df_final = df_cleaned[rows_to_keep].reset_index(drop=True)
#
# # Step 7: ML Model
# df_model = df_final.dropna(subset=['days_to_sad']).copy()
# X = df_model.drop(columns=['timestamp', 'days_to_sad'])
# y = df_model['days_to_sad']
# print("🎯 Trainable rows:", df_model.shape[0])
#
# # Step 7: ML Model
# X = df_model.drop(columns=['timestamp', 'days_to_sad'])
# y = df_model['days_to_sad']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# from xgboost import XGBRegressor
# model = XGBRegressor(n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42)
# model.fit(X_train, y_train)
#
# y_pred = model.predict(X_test)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# r2 = r2_score(y_test, y_pred)
#
# print(f"✅ RMSE: {rmse:.2f} days")
# print(f"📊 R² Score: {r2:.3f}")
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, y_pred, alpha=0.4)
# plt.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--')
# plt.xlabel("Actual Days to SAD")
# plt.ylabel("Predicted Days to SAD")
# plt.title("Predicted vs Actual")
# plt.grid(True)
# plt.show()
#
# # plt.figure(figsize=(8, 5))
# # plt.scatter(y_test, y_pred, alpha=0.6, color='dodgerblue')
# # plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
# # plt.xlabel("Actual dP (kg/cm²)")
# # plt.ylabel("Predicted dP (kg/cm²)")
# # plt.title("Regression: Actual vs Predicted Coil dP (Pass 3)")
# # plt.grid(True)
# # plt.tight_layout()
# # plt.show()
#
#
#
#
# import joblib
#
# # # Save
# joblib.dump(model, 'sad_days_predictor_all_years_merged.pkl')
# joblib.dump(list(X.columns), 'sad_feature_columns_all_years_merged.pkl')
# print("💾 Model and features saved successfully!")

# 🔁 Step 0: Load required libraries
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib  # for saving/loading model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# ✅ Step 1: Load merged dataset (2021–2025)
df_raw = pd.read_csv('merged-csv-files2.csv', header=4, skiprows=[0, 1, 2, 3])
print("📄 Raw data loaded. Shape:", df_raw.shape)

# ✅ Step 2: Rename columns
renamed_columns = {
    df_raw.columns[2]: "timestamp",
    df_raw.columns[3]: "Coil_Inlet_Pressure_Pass1",
    df_raw.columns[4]: "Coil_Inlet_Pressure_Pass2",
    df_raw.columns[5]: "Coil_Inlet_Pressure_Pass3",
    df_raw.columns[6]: "Coil_Inlet_Pressure_Pass4",
    df_raw.columns[7]: "FeedFlow_Pass1",
    df_raw.columns[8]: "FeedFlow_Pass2",
    df_raw.columns[9]: "FeedFlow_Pass3",
    df_raw.columns[10]: "FeedFlow_Pass4",
    df_raw.columns[11]: "COT_Pass1",
    df_raw.columns[12]: "COT_Pass2",
    df_raw.columns[13]: "COT_Pass3",
    df_raw.columns[14]: "COT_Pass4",
    df_raw.columns[15]: "COIL_dP_Pass1",
    df_raw.columns[16]: "COIL_dP_Pass2",
    df_raw.columns[17]: "COIL_dP_Pass3",
    df_raw.columns[18]: "COIL_dP_Pass4",
    df_raw.columns[19]: "BFW_Injection_Pass1",
    df_raw.columns[20]: "BFW_Injection_Pass2",
    df_raw.columns[21]: "BFW_Injection_Pass3",
    df_raw.columns[22]: "BFW_Injection_Pass4",
    df_raw.columns[23]: "MAX_SkinTemp_Pass1",
    df_raw.columns[24]: "Max_SkinTemp_Pass2",
    df_raw.columns[25]: "Max_SkinTemp_Pass3",
    df_raw.columns[26]: "Max_SkinTemp_Pass4",
}
df_renamed = df_raw.rename(columns=renamed_columns)

# ✅ Step 3: Clean
df_cleaned = df_renamed.drop(columns=[df_raw.columns[0], df_raw.columns[1]], errors='ignore')
df_cleaned['timestamp'] = pd.to_datetime(df_cleaned['timestamp'], errors='coerce')
df_cleaned = df_cleaned.dropna(subset=['timestamp']).reset_index(drop=True)

# ✅ Step 4: Mark 2025 and earlier datasets
df_cleaned['year'] = df_cleaned['timestamp'].dt.year
df_2025 = df_cleaned[(df_cleaned['timestamp'] >= '2025-01-01') & (df_cleaned['timestamp'] <= '2025-06-27')].copy()
df_2021_24 = df_cleaned[df_cleaned['year'] < 2025].copy()

# ✅ Step 5: Label days_to_sad on 2021–2024
sad_starts = ['2021-05-03', '2022-11-23', '2023-10-04', '2024-12-14']
feed_cuts = ['2021-05-30', '2022-12-05', '2023-11-18', '2024-11-29']
sad_feed_blocks = [(pd.to_datetime(s), pd.to_datetime(f)) for s, f in zip(sad_starts, feed_cuts)]

df_2021_24['days_to_sad'] = np.nan
rows_to_keep = pd.Series([False] * len(df_2021_24), index=df_2021_24.index)

for sad_start, feed_cut in sad_feed_blocks:
    mask = (df_2021_24['timestamp'] < sad_start) & (df_2021_24['timestamp'] > feed_cut - pd.Timedelta(days=365))
    df_2021_24.loc[mask, 'days_to_sad'] = (sad_start - df_2021_24.loc[mask, 'timestamp']).dt.days
    rows_to_keep |= mask

df_labeled = df_2021_24[rows_to_keep].copy()

# ✅ Step 6: Train initial model on true data
features = df_labeled.drop(columns=['timestamp', 'days_to_sad', 'year']).columns.tolist()
X = df_labeled[features]
y = df_labeled['days_to_sad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_base = XGBRegressor(n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42)
model_base.fit(X_train, y_train)

# ✅ Save if needed
joblib.dump(model_base, "model_base_2021_24.pkl")
joblib.dump(features, "features_base_2021_24.pkl")



# ✅ Step 7: Predict pseudo-labels for 2025
df_2025['pseudo_days_to_sad'] = model_base.predict(df_2025[features])

# ✅ Step 8: Retrain on 2025 data with pseudo-labels
X_2025 = df_2025[features]
y_2025 = df_2025['pseudo_days_to_sad']

X_train_pseudo, X_val_pseudo, y_train_pseudo, y_val_pseudo = train_test_split(X_2025, y_2025, test_size=0.2, random_state=42)

model_retrained = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
model_retrained.fit(X_train_pseudo, y_train_pseudo)

# ✅ Evaluate on pseudo labels (useful for drift analysis)
y_pred_pseudo = model_retrained.predict(X_val_pseudo)
rmse_pseudo = np.sqrt(mean_squared_error(y_val_pseudo, y_pred_pseudo))
r2_pseudo = r2_score(y_val_pseudo, y_pred_pseudo)

print(f"🔁 Retrained on 2025 (pseudo) | RMSE: {rmse_pseudo:.2f} days | R²: {r2_pseudo:.3f}")
joblib.dump(model_retrained, "model_retrained_2025.pkl")
joblib.dump(features, 'sad_feature_columns_retrained_2025.pkl')

import matplotlib.pyplot as plt

plt.figure(figsize=(14,5))
plt.plot(df_2025['timestamp'], df_2025['pseudo_days_to_sad'], label='Predicted Days to SAD (2025)', color='orange')
plt.ylabel('Predicted Days to SAD')
plt.xlabel('Date')
plt.title('SAD Proximity Forecast - 2025 (Pseudo Labels)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
import seaborn as sns

for col in features[:6]:  # Check 6 key features
    sns.kdeplot(df_labeled[col], label='2021–24', fill=True)
    sns.kdeplot(df_2025[col], label='2025', fill=True)
    plt.title(f'Distribution Shift: {col}')
    plt.legend()
    plt.show()
