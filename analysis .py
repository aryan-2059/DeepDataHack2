import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Set your CSV file path here
csv_path = 'urban_pluvial_flood_risk_dataset.csv'  # Update this path as needed
df = pd.read_csv(csv_path)

# Outlier detection columns
numerical_col = df.select_dtypes(include=np.number).columns.tolist()

# Outlier detection and removal
plt.figure(figsize=(10,6))
sns.boxplot(data=df[numerical_col], orient="h", palette="Set3")
plt.title("Outlier Detection Across All Numeric Columns")
plt.show()

numeric_df = df.select_dtypes(include=np.number)
z_scores = np.abs(stats.zscore(numeric_df))
threshold = 3
df_no_outliers = df[(z_scores < threshold).all(axis=1)]
df = df_no_outliers

plt.figure(figsize=(10,6))
sns.boxplot(data=df[numerical_col], orient="h", palette="Set3")
plt.title("Outlier Detection After Removal")
plt.show()

# Feature Engineering
df['rainfall_intensity_per_year'] = df['historical_rainfall_intensity_mm_hr'] / df['return_period_years'].replace(0, np.nan)
df['rainfall_intensity_per_year'] = df['rainfall_intensity_per_year'].fillna(0)

# Clean and encode risk labels
df['risk_labels_no_event'] = df['risk_labels'].str.replace(r'\|?event_\d{4}-\d{2}-\d{2}', '', regex=True)
df['risk_labels_final'] = (
    df['risk_labels_no_event']
    .str.split('|')
    .apply(lambda x: sorted(set(x)))   # remove duplicates & sort
    .str.join('|')
)
df['risk_labels_list'] = df['risk_labels_final'].str.split('|')
df['risk_labels_list'] = df['risk_labels_list'].apply(lambda x: list(set(x)))
risk_dummies = df['risk_labels_list'].explode().str.get_dummies().groupby(level=0).max()
df_clean = pd.concat([df, risk_dummies], axis=1)
df_clean.drop(columns=['risk_labels_list','risk_labels_final','risk_labels_no_event','risk_labels'], inplace=True)
df_encoded = pd.get_dummies(df_clean, columns=['land_use','soil_group','country','storm_drain_type','rainfall_source'], dtype=int)

# Prepare features and labels
X = df_encoded.drop(columns=['city_name','city'])
y = df_encoded[['extreme_rain_history','low_lying','monitor','ponding_hotspot','sparse_drainage']]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
standardized_X_train = scaler.fit_transform(X_train)
standardized_X_test = scaler.transform(X_test)

# Build and train model
model = Sequential([
    Dense(128, activation='relu', input_shape=(standardized_X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(y_train.shape[1], activation='sigmoid')  # sigmoid for multi-label
])
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',   # multi-label classification
    metrics=['accuracy']
)
history = model.fit(
    standardized_X_train, y_train,
    validation_data=(standardized_X_test, y_test),
    epochs=30,
    batch_size=32
)
loss, acc = model.evaluate(standardized_X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")

# Plot accuracy and loss
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()