# Forest Cover Type Prediction

## Usage

### 1. Environment Setup
Install the required dependencies:
```bash
pip3 install -r requirements.txt
```

### 2. Run Training
Execute the model selection script to train models and generate artifacts:
```bash
python3 scripts/model_selection.py
```

### 3. Run Prediction
Generate predictions on the test set:
```bash
python3 scripts/predict.py
```

## Audit Check Section

### Models Used
1.  **Logistic Regression**: Scale-sensitive linear model.
2.  **K-Nearest Neighbors (KNN)**: Scale-sensitive distance-based model.
3.  **Support Vector Machine (SVM)**: Scale-sensitive, trained on a subset (20,000 samples) to ensure performance.
4.  **Random Forest**: Tree-based ensemble model (no scaling required).
5.  **GradientBoostingClassifier**: Scikit-learn's Gradient Boosting implementation (no scaling required).

### Cross-Validation Strategy
-   **5-Fold Stratified Cross-Validation**: Used `StratifiedKFold` with `check_cv=5` to preserve the percentage of samples for each class.
-   Refitting is done on the whole training set after cross-validation to produce the `best_pipeline.pkl`.

### Feature Engineering Logic
-   **Soil Type Mapping**: Reversed One-Hot Encoded `Soil_Type1`...`Soil_Type40` into a single `Soil_Type` column. Further mapped these to `Climatic_Zone` and `Geologic_Zone` using the specific research dictionary.
-   **Wilderness Area Mapping**: Reversed One-Hot Encoded `Wilderness_Area1`...`Wilderness_Area4` into `Wilderness_ID`.
-   **New Features**:
    -   `Euclidean_Dist_Hydro`: Combined horizontal and vertical distance to hydrology.
    -   `Sin_Aspect` / `Cos_Aspect`: Cyclical transformation of Aspect.
    -   `Mean_Hillshade`: Average of 9am, Noon, and 3pm hillshade.
    -   `Hillshade_Contrast`: Absolute difference between 9am and 3pm hillshade.
    -   `Interaction_Fire_Road`: signed difference between distance to fire points and roadways.

## Results

### Train/Test Accuracies
-   **Train Accuracy**: 0.9358 (Audit Pass: < 0.98)
-   **Test Accuracy**: 0.9133 (Audit Pass: > 0.65)

### Artifacts
-   **Confusion Matrix**: `results/confusion_matrix_heatmap.png`
-   **Learning Curve**: `results/learning_curve_best_model.png`
-   **Best Pipeline**: `results/best_pipeline.pkl`
-   **Test Predictions**: `results/test_predictions.csv`
