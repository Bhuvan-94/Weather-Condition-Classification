# Weather Condition Classification using SVM and Open-Meteo API

## Objective
Develop a Support Vector Machine (SVM) classifier with an RBF kernel to predict weather classes (`Cool` vs `Warm`) using meteorological observations fetched from the Open-Meteo API.

## API Documentation Link
- **Open-Meteo API**: [https://open-meteo.com/](https://open-meteo.com/)

## Libraries Used
- **Pandas**: For data loading, manipulation, and converting the JSON API response into a structured DataFrame.
- **NumPy**: For numerical computations and class metrics.
- **Scikit-Learn**: For dataset splitting (`train_test_split`), feature scaling (`StandardScaler`), building the machine learning model (`SVC`), and evaluating its performance (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`).
- **Matplotlib & Seaborn**: For creating confusion matrix heatmap visualizations.
- **Requests**: For fetching weather data from the API endpoint.

## Methodology
1. **Data Collection & Understanding**: Fetched weather forecast data for the next 7 days from the Open-Meteo API. To ensure robust training with a diverse dataset, the script checks for class diversity. If the initial location (New Delhi) has single-class summer temperatures (all Warm), the script automatically falls back to a temperate location (London) or a synthetic data generator.
2. **Feature Engineering**: Suggested features (`Temperature`, `Relative_Humidity`, `Surface_Pressure`, `Wind_Speed`) were extracted. The target variable `Weather_Class` was engineered: `Warm` (1) if Temperature &ge; 25&deg;C, else `Cool` (0).
3. **Data Preprocessing**: Checked for missing values. Removed the `time` index to prepare the features for classification. Split the data into an 80% training set and a 20% testing set using stratification to maintain class balance.
4. **Feature Scaling**: Applied `StandardScaler` (fit on training data and applied to both sets) to prevent features with larger numeric scales (e.g. Surface Pressure ~1015 hPa) from dominating features with smaller scales (e.g. Temperature ~22&deg;C).
5. **Model Development**: Trained a Support Vector Machine (SVM) Classifier with a Radial Basis Function (RBF) kernel using Scikit-Learn.
6. **Model Evaluation**: Evaluated the model on the test set using Accuracy, Precision, Recall, and F1-Score. Generated a heatmap of the confusion matrix.

## Results

### Metrics (on London Test Set)
- **Accuracy**: 91.18%
- **Precision**: 83.33%
- **Recall**: 71.43%
- **F1-Score**: 76.92%

### Confusion Matrix
```
                 Predicted
              Cool(0)  Warm(1)
Actual Cool      26       1
Actual Warm       2       5
```
- **True Negatives**: 26 (Cool weather correctly identified)
- **False Positives**: 1 (Cool weather predicted as Warm)
- **False Negatives**: 2 (Warm weather predicted as Cool)
- **True Positives**: 5 (Warm weather correctly identified)

### Key Observations
1. The SVM model achieves a high classification accuracy of **91.18%** on the test dataset.
2. Feature scaling was critical: without `StandardScaler`, Surface Pressure would have heavily dominated distance calculations.
3. The RBF kernel successfully maps the non-linear interactions of weather indicators to establish a highly predictive boundary.

## Conclusion
Key Findings: The Support Vector Machine (SVM) model with an RBF kernel classifies the weather condition (Cool vs Warm) with an accuracy of 91.2%. The precision is 83.3% and recall is 71.4%, highlighting high robustness. The target variable is derived directly from Temperature (Warm if >= 25°C), making Temperature the dominant predictive feature.

Importance of Feature Scaling in SVM: SVM attempts to maximize the margin between classes by measuring geometric distances between data points (support vectors). Without feature scaling (via StandardScaler), features with large numeric ranges like Surface Pressure (around 1000 hPa) would completely dominate features with smaller scales like Temperature or Wind Speed, distorting the distance measurements and making the RBF kernel ineffective.

Advantage and Limitation of SVM: One key advantage of SVM is its ability to handle non-linear decision boundaries efficiently using the kernel trick (specifically the RBF kernel). A major limitation of SVM is its sensitivity to hyperparameter selection (like C and gamma) and its high computational cost when training on very large datasets.
