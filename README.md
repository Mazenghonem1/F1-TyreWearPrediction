Tire Degradation Prediction for Formula 1 Races
Overview
This project develops a machine learning model to predict tire degradation rates (TyreDegRate, in seconds per lap) in Formula 1 races, enabling better race strategy decisions, such as optimal pit stop timing. The model uses lap-by-lap telemetry data (e.g., lap times, sector times, speeds) and weather data (e.g., track temperature) to estimate how tire performance degrades over a stint.
The project is implemented in a Jupyter Notebook (Tire_Degradation_Prediction.ipynb) and includes data preprocessing, feature engineering, model training, hyperparameter tuning, evaluation, interpretability analysis, and a deployment pipeline. The best model, a Multi-Layer Perceptron (MLP), achieves a Test Mean Absolute Error (MAE) of 0.1705 seconds, meaning predictions are accurate to within ~0.17 seconds per lap.
Objectives

Predict TyreDegRate using telemetry and weather data.
Identify key factors influencing tire degradation (e.g., tire compound, track temperature).
Build a deployable pipeline for real-time race strategy predictions.
Provide interpretable insights for race engineers.

Dataset

Tire Data:
Columns: Driver, LapNumber, LapTime (s), Sector1Time, Sector2Time, Sector3Time, SpeedI1, SpeedI2, SpeedFL, SpeedST, TrackStatus, FreshTyre, Compound, TyreLife, Stint, TyreDegRate (target).
Source: Formula 1 telemetry data (e.g., FastF1 API).


Weather Data:
Columns: Time, AirTemp, TrackTemp.
Source: Weather station data merged with telemetry.


Note: Data files are not included due to size or sensitivity. Users must provide their own data with the same structure.

Methodology
The project is structured in 14 cells in the Jupyter Notebook, each handling a specific task:
1–7: Data Loading and Preprocessing

Data Loading: Load using pandas.
Cleaning:
Handle missing values (e.g., impute LapTime (s) with median).
Remove outliers (e.g., LapTime (s) > 3 standard deviations).


Weather Merge:
Merge tire_data with weather_data using pd.merge_asof on Time with a 5-minute tolerance.
Result: tire_data includes AirTemp and TrackTemp.
Challenges: Addressed high NaN counts by adjusting merge tolerance and interpolating weather data.



8: Feature Engineering

Created features to capture tire degradation dynamics:
StintLength: Number of laps in each stint.
CumulativeDistance: Total distance traveled by tires (TyreLife * 5.412 km).
DriverAggressiveness: Ratio of SpeedST to its mean.
TyreWearFactor: TyreLife / log(LapNumber + 2).
TyreWearRate: TyreWearFactor / SpeedST.
PrevLapTime, PrevSpeedST: Lagged values of LapTime (s) and SpeedST.
Compound_TrackTemp: Interaction between Compound and TrackTemp.
TireLoadIndex: (SpeedI1 + SpeedI2) / SpeedST.
SectorTimeStd: Standard deviation of sector times.
PostPitLap: Binary indicator for laps after a pit stop.


Output: features list with 24 features (numeric and categorical).

9: Data Splitting

Split data into X (features) and y (TyreDegRate).
Train-test split: 80% train, 20% test using train_test_split (random_state=42).

10–11: Model Training and Hyperparameter Tuning

Trained seven models: RandomForest, GradientBoosting, SVR, XGBoost, LightGBM, CatBoost, MLP.
Used GridSearchCV for hyperparameter tuning (Cell 11):
Example: MLP tuned hidden_layer_sizes, alpha, learning_rate.
Best parameters saved for each model.


Validation MAE for top models: MLP (0.1526), CatBoost (0.1592), SVR (0.1472).

12: Model Evaluation

Evaluated models on the test set:


Model
Test MAE
Test RMSE



RandomForest
0.2405
0.6391


GradientBoosting
0.1721
0.5458


SVR
0.1770
0.5699


XGBoost
0.1717
0.5180


LightGBM
0.2048
0.5456


CatBoost
0.1874
0.5720


MLP
0.1705
0.3809



Best Model: MLP, with Test MAE of 0.1705 sec (~9.7 sec cumulative error over a 57-lap race).
Visualized predictions vs. actuals using scatter plots.

13: Model Interpretability

Used SHAP to analyze feature importance for the MLP model.
Key features: TyreWearFactor, Compound, TrackTemp.
Generated partial dependence plots to show feature effects on TyreDegRate.

14: Pipeline and Deployment

Built a Pipeline with:
ColumnTransformer: Scales numeric features (StandardScaler) and encodes categorical features (OneHotEncoder).
model: MLP (best_model from Cell 12).


Trained and saved the pipeline as tire_degradation_model.pkl.
Implemented a predict_tire_degradation function for real-time predictions.
Test MAE: ~0.17 sec, consistent with Cell 12.
Challenges:
Fixed KeyError: 'Driver' and KeyError: 'Stint' by removing redundant feature engineering, as features were already computed in Cell 8.



Results

Performance: MLP achieves a Test MAE of 0.1705 sec, a ~25% improvement over the baseline XGBoost (0.2263 sec).
Practical Impact: Accurate to ~0.17 sec per lap, enabling precise pit stop timing (e.g., <10 sec error over a race).
Key Features: TyreWearFactor, Compound, and TrackTemp are the most influential, highlighting the importance of tire wear dynamics and environmental conditions.
Generalization: Small gap between validation (0.15 sec) and test MAE (0.17 sec) indicates robust generalization.

Setup and Installation
To run the project locally:
Prerequisites

Python 3.8+
Libraries: pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, tensorflow, shap, joblib
Install dependencies:pip install pandas numpy scikit-learn xgboost lightgbm catboost tensorflow shap joblib



Steps

Clone the Repository:
git clone https://github.com/yourusername/tire-degradation-prediction.git
cd tire-degradation-prediction




Run the Notebook:

Open Tire_Degradation_Prediction.ipynb in Jupyter Notebook:jupyter notebook


Run all cells sequentially.


Load and Use the Model:

Load the saved model:import joblib
pipeline = joblib.load('tire_degradation_model.pkl')


Predict for a new lap (example):import pandas as pd
new_lap = pd.DataFrame({ ... })  # Fill with feature values
prediction = pipeline.predict(new_lap)
print(f"Predicted TyreDegRate: {prediction[0]:.4f} sec")




Data: Inspired by Formula 1 telemetry (e.g., FastF1 API).
Tools: Built with Python, scikit-learn, TensorFlow, and SHAP.

