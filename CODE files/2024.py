# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# df_raw = pd.read_csv('DCU - 2024.csv',header=4 , skiprows=[0, 1, 2, 3])
# df_raw.info(), df_raw.head()
#
# renamed_columns = {
#     df_raw.columns[2]: "timestamp",
#     df_raw.columns[3]: "Coil_Inlet_Pressure_Pass1",
#     df_raw.columns[4]: "Coil_Inlet_Pressure_Pass2",
#     df_raw.columns[5]: "Coil_Inlet_Pressure_Pass3",
#     df_raw.columns[6]: "Coil_Inlet_Pressure_Pass4",
#     df_raw.columns[7]: "FeedFlow_Pass1",
#     df_raw.columns[8]: "FeedFlow_Pass2",
#     df_raw.columns[9]: "FeedFlow _Pass3",
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
#
# }
#
#
# df_renamed = df_raw.rename(columns=renamed_columns)
# df_renamed.info(), df_renamed.head()
# df_cleaned = df_renamed.drop(columns=[df_raw.columns[0], df_raw.columns[1]])
# df_cleaned['timestamp'] = pd.to_datetime(df_cleaned['timestamp'], errors='coerce')
# df_cleaned = df_cleaned.dropna(subset=['timestamp'])
# df_cleaned = df_cleaned.reset_index(drop=True)
# df_cleaned.head()
#
# numerical_df = df_cleaned.select_dtypes(include='number')
#
# # Compute correlation matrix
# # correlation_matrix = numerical_df.corr()
# #
# # # Plot correlation heatmap
# # plt.figure(figsize=(18, 12))
# # sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt=".2f", linewidths=0.5)
# # plt.title("Correlation Matrix of Furnace Parameters", fontsize=19)
# # plt.tight_layout()
# # #plt.show()
# # plt.savefig('correlation_matrix_2024.png')
#
# correlations = df_cleaned.corr()['COIL_dP_Pass3'].drop('COIL_dP_Pass3')
#
# # Top 5 features (absolute correlation)
# top_5_features = correlations.abs().sort_values(ascending=False).head(5)
# print("Top 5 Correlated Features:\n", correlations.loc[top_5_features.index])
#
#
# top_feature = top_5_features.index[0]
# plt.plot(df_cleaned['timestamp'], df_cleaned['COIL_dP_Pass3'], label='COIL_dP_Pass3', color='red')
# plt.plot(df_cleaned['timestamp'], df_cleaned[top_feature], label=top_feature, color='blue', alpha=0.6)
# plt.axvspan(pd.to_datetime("12-14-2024"), pd.to_datetime("12-21-2024"), color='orange', alpha=0.3, label='SAD Period')
#
# plt.xlabel('Timestamp')
# plt.ylabel('Value')
# plt.title(f"Time Series: COIL_dP_Pass3 vs {top_feature}")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('timeseries_COIL_dP_Pass3_vs_top_feature.png')
# #plt.show()
# plt.savefig('timeseries 2024_plot.png')
#
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 🎯 Set your regression target
# target = 'COIL_dP_Pass3'
#
# # ❌ Drop features that leak or are irrelevant
# X = df_cleaned.drop(columns=[
#     'timestamp',
#     'COIL_dP_Pass2',
#     'COIL_dP_Pass4',
#     'COIL_dP_Pass3',
#     'BFW_Injection_Pass4',
#     'BFW_Injection_Pass2',
#     'BFW_Injection_Pass1',
# ])
#
# y = df_cleaned[target]
#
# # Split data into train and test sets
# # 🧪 Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 🌲 Train regression model
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)
#
# # 📈 Evaluate model performance
# y_pred = model.predict(X_test)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# r2 = r2_score(y_test, y_pred)
#
# print(f"✅ RMSE: {rmse:.2f} kg/cm²")
# print(f"📊 R² Score: {r2:.3f} (1.0 = perfect prediction)")
# plt.figure(figsize=(8, 5))
# plt.scatter(y_test, y_pred, alpha=0.6, color='dodgerblue')
# plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
# plt.xlabel("Actual dP (kg/cm²)")
# plt.ylabel("Predicted dP (kg/cm²)")
# plt.title("Regression: Actual vs Predicted Coil dP (Pass 3)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()




import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df_raw = pd.read_csv('DCU - 2023.csv',header=4 , skiprows=[0, 1, 2, 3])
df_raw.info(), df_raw.head()

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
df_renamed.info(), df_renamed.head()
df_cleaned = df_renamed.drop(columns=[df_raw.columns[0], df_raw.columns[1]])
df_cleaned['timestamp'] = pd.to_datetime(df_cleaned['timestamp'], errors='coerce')
df_cleaned = df_cleaned.dropna(subset=['timestamp'])
df_cleaned = df_cleaned.reset_index(drop=True)
df_cleaned.head()

df_cleaned.info()

numerical_df = df_cleaned.select_dtypes(include='number')
# Example SAD period (update as needed)
sad_start = pd.to_datetime("2024-12-14")
sad_end = pd.to_datetime("2024-12-21")

# Create a column for 'days_to_sad'
df_cleaned['days_to_sad'] = (sad_start - df_cleaned['timestamp']).dt.days

# Set values after SAD or far before to NaN (not useful for training)
df_cleaned.loc[df_cleaned['timestamp'] >= sad_start, 'days_to_sad'] = np.nan
df_cleaned.loc[df_cleaned['days_to_sad'] < 0, 'days_to_sad'] = np.nan
df_model = df_cleaned.dropna(subset=['days_to_sad'])



from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X = df_model.drop(columns=[
    'timestamp',
    'days_to_sad'  # <- target
])
y = df_model['days_to_sad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ RMSE: {rmse:.2f} days")
print(f"📊 R² Score: {r2:.3f}")

# 🔍 Optional: Visualize prediction vs actual
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.6, color='dodgerblue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
plt.xlabel("Actual dP (kg/cm²)")
plt.ylabel("Predicted dP (kg/cm²)")
plt.title("Regression: Actual vs Predicted Coil dP (Pass 3)")
plt.grid(True)
plt.tight_layout()
plt.show()

import joblib
joblib.dump(model, 'sad_days_predictor_2024.pkl')
joblib.dump(X.columns.tolist(), 'sad_feature_columns_2024.pkl')


# Compute correlation matrix
# correlation_matrix = numerical_df.corr()
#
# # Plot correlation heatmap
# plt.figure(figsize=(18, 12))
# sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt=".2f", linewidths=0.5)
# plt.title("Correlation Matrix of Furnace Parameters", fontsize=16)
# plt.tight_layout()
# plt.show()
# plt.savefig('correlation_matrix_2022.png')
#
# correlations = df_cleaned.corr()['COIL_dP_Pass3','COIL_dP_Pass4','COIL_dP_Pass1','COIL_dP_Pass2'].drop['COIL_dP_Pass3','COIL_dP_Pass4','COIL_dP_Pass1','COIL_dP_Pass2']
# #
# # Top 5 features (absolute correlation)
# top_5_features = correlations.abs().sort_values(ascending=False).head(5)
# print("Top 5 Correlated Features:\n", correlations.loc[top_5_features.index])
#
#
# top_feature = top_5_features.index[0]
# #
# plt.plot(df_cleaned['timestamp'], df_cleaned['COIL_dP_Pass3'], label='COIL_dP_Pass3', color='red')
# plt.plot(df_cleaned['timestamp'], df_cleaned[top_feature], label=top_feature, color='blue', alpha=0.6)
# plt.axvspan(pd.to_datetime("23-11-2022"), pd.to_datetime("28-11-2022"), color='orange', alpha=0.3, label='SAD Period')
#
# plt.xlabel('Timestamp')
# plt.ylabel('Value')
# plt.title(f"Time Series: COIL_dP_Pass3 vs {top_feature}")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('timeseries_COIL_dP_Pass3_vs_top_feature.png')
# plt.show()
# plt.savefig('timeseries 2022.png')