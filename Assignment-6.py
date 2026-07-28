import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

print("=" * 60)
print("TASK 1: DATA COLLECTION AND UNDERSTANDING")
print("=" * 60)

# We will try New Delhi first. If it's too hot/cold and doesn't have both classes,
# we will fall back to a temperate location (London) to get a mix of Cool & Warm classes.
locations = [
    {"name": "New Delhi", "lat": 28.6139, "lon": 77.2090},
    {"name": "London", "lat": 51.5074, "lon": -0.1278}
]

df = None
fetched_location = ""

for loc in locations:
    url = f"https://api.open-meteo.com/v1/forecast?latitude={loc['lat']}&longitude={loc['lon']}&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m&forecast_days=7"
    try:
        print(f"Attempting to fetch data for {loc['name']}...")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        hourly_data = data['hourly']
        temp_df = pd.DataFrame(hourly_data)
        
        # Test if this location provides both Warm (>=25) and Cool (<25) classes
        temps = temp_df['temperature_2m']
        n_warm = (temps >= 25.0).sum()
        n_cool = (temps < 25.0).sum()
        
        print(f"  Found {n_warm} Warm and {n_cool} Cool hours.")
        if n_warm > 5 and n_cool > 5:
            df = temp_df
            fetched_location = loc['name']
            print(f"Successfully selected {loc['name']} data with mixed classes.")
            break
        else:
            print(f"  Skipping {loc['name']} due to insufficient class diversity.")
    except Exception as e:
        print(f"  Error fetching {loc['name']}: {e}")

# If API calls fail or don't provide class diversity, use synthetic data
if df is None:
    print("\nLive data not available or lacked class diversity. Generating synthetic mixed weather data...")
    fetched_location = "Synthetic Weather Station"
    np.random.seed(42)
    time_range = pd.date_range(start="2026-07-28", periods=168, freq="h").strftime("%Y-%m-%dT%H:00").tolist()
    
    # Simulate a realistic weather pattern that crosses the 25°C boundary
    temp_base = 24 + 5 * np.sin(np.linspace(0, 7 * 2 * np.pi, 168)) # Oscillates between 19°C and 29°C
    temperature = temp_base + np.random.normal(0, 0.8, 168)
    
    # Relative humidity (inversely proportional to temperature)
    humidity = 80 - 15 * (temperature - 20) / 10 + np.random.normal(0, 3, 168)
    humidity = np.clip(humidity, 30, 98)
    
    # Surface pressure
    pressure = 1005 + 5 * np.cos(np.linspace(0, 7 * 2 * np.pi, 168)) + np.random.normal(0, 1, 168)
    
    # Wind speed
    wind_speed = np.abs(np.random.normal(8, 3, 168))
    
    df = pd.DataFrame({
        'time': time_range,
        'temperature_2m': temperature,
        'relative_humidity_2m': humidity,
        'surface_pressure': pressure,
        'wind_speed_10m': wind_speed
    })

# Rename features as suggested
df.rename(columns={
    'temperature_2m': 'Temperature',
    'relative_humidity_2m': 'Relative_Humidity',
    'surface_pressure': 'Surface_Pressure',
    'wind_speed_10m': 'Wind_Speed'
}, inplace=True)

# Display first five records
print(f"\nFirst 5 records from {fetched_location}:")
print(df.head())

# Identify Input Features & Target Variable
print("\nInput Features: Temperature, Relative_Humidity, Surface_Pressure, Wind_Speed")
print("Target Variable: Weather_Class (Created below)")

# Create a new column named Weather_Class: Warm (>= 25) vs Cool (< 25)
df['Weather_Class'] = df['Temperature'].apply(lambda t: 'Warm' if t >= 25.0 else 'Cool')

# Target distribution
print(f"\nClass Distribution:")
print(df['Weather_Class'].value_counts())

# Display dataset info and summary statistics
print("\nDataset Info:")
df.info()

print("\nSummary Statistics:")
print(df.describe())

print("\n" + "=" * 60)
print("TASK 2: DATA PREPROCESSING")
print("=" * 60)

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Remove unnecessary columns
# 'time' is not needed for the classification model
df_cleaned = df.drop(columns=['time'])
print("\nDropped time column.")

# Encode the target variable: Warm -> 1, Cool -> 0
df_cleaned['Weather_Class'] = df_cleaned['Weather_Class'].map({'Warm': 1, 'Cool': 0})
print("Encoded Target (Weather_Class): Warm = 1, Cool = 0")

# Separate features and target
X = df_cleaned.drop(columns=['Weather_Class'])
y = df_cleaned['Weather_Class']

# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\nDataset split into 80% training and 20% testing (Stratified).")

# Standardize the feature values using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features standardized using StandardScaler (fit on training data only).")

print("\n" + "=" * 60)
print("TASK 3: MODEL DEVELOPMENT")
print("=" * 60)

# Build an SVM Classifier using Kernel = RBF
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)
print("\nSVM Classifier (Kernel=RBF) trained successfully.")

# Predict the weather class for the test dataset
y_pred = svm_model.predict(X_test_scaled)
print("Predictions generated on test set.")

print("\n" + "=" * 60)
print("TASK 4: MODEL EVALUATION")
print("=" * 60)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"\nAccuracy Score: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision Score: {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall Score:    {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1-Score:        {f1:.4f} ({f1 * 100:.2f}%)")

# Generate Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Cool (0)', 'Warm (1)'],
            yticklabels=['Cool (0)', 'Warm (1)'])
plt.title(f'Confusion Matrix - SVM Weather Classification ({fetched_location})', fontsize=10, fontweight='bold')
plt.xlabel('Predicted Weather Class')
plt.ylabel('Actual Weather Class')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.close()
print("\nConfusion matrix plot saved as 'confusion_matrix.png'")

# Observations
print("\n" + "=" * 60)
print("OBSERVATIONS")
print("=" * 60)

obs_1 = f"1. The SVM model achieves an accuracy of {accuracy*100:.2f}% in predicting the weather class for data collected from {fetched_location}."
obs_2 = "2. Applying StandardScaler was essential because parameters like Surface Pressure (~1005 hPa) operate on a completely different scale compared to Wind Speed (~8 km/h) or Temperature (~26°C)."
obs_3 = "3. An RBF (Radial Basis Function) kernel is highly suited for weather classification because weather boundaries are often non-linear and involve complex multidimensional thresholds."

print(obs_1)
print(obs_2)
print(obs_3)

print("\n" + "=" * 60)
print("TASK 5: CONCLUSION")
print("=" * 60)

conclusion = f"""
Key Findings: The Support Vector Machine (SVM) model with an RBF kernel classifies the weather condition (Cool vs Warm) with an accuracy of {accuracy*100:.1f}%. The precision is {precision*100:.1f}% and recall is {recall*100:.1f}%, highlighting high robustness. The target variable is derived directly from Temperature (Warm if >= 25°C), making Temperature the dominant predictive feature.

Importance of Feature Scaling in SVM: SVM attempts to maximize the margin between classes by measuring geometric distances between data points (support vectors). Without feature scaling (via StandardScaler), features with large numeric ranges like Surface Pressure (around 1000 hPa) would completely dominate features with smaller scales like Temperature or Wind Speed, distorting the distance measurements and making the RBF kernel ineffective.

Advantage and Limitation of SVM: One key advantage of SVM is its ability to handle non-linear decision boundaries efficiently using the kernel trick (specifically the RBF kernel). A major limitation of SVM is its sensitivity to hyperparameter selection (like C and gamma) and its high computational cost when training on very large datasets.
"""

print(conclusion.strip())

# Save conclusion to file
with open('conclusion.txt', 'w', encoding='utf-8') as f:
    f.write(conclusion.strip())

print("\nConclusion text saved to 'conclusion.txt'")
print("=" * 60)
print("ASSIGNMENT 6 COMPLETE")
print("=" * 60)
