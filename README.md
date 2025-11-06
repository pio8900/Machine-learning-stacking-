
***

# Airbnb Price Prediction — Ensemble ML + Flask App

## Project Overview

This project predicts Airbnb listing prices by training and stacking multiple machine learning regressors. It includes code for data exploration, robust preprocessing, model training, and a simple Flask web app for predictions.

**Pipeline Highlights:**
- Data preprocessing/analysis in Jupyter notebook
- Multiple regression models (CatBoost, LightGBM, GradientBoosting)
- Stacked ensemble (XGBoost meta-model)
- Joblib model exports
- Browser-based prediction interface (Flask app)

## Folder Structure

```
projects/ml/
├── pio_mll.ipynb           # Main ML notebook (EDA, training, model export)
├── app.py                  # Flask web app + prediction interface
├── catboost_model.joblib   # Trained CatBoost model
├── lgbm_model.joblib       # Trained LightGBM model
├── gbr_model.joblib        # Trained GradientBoosting model
├── model1.joblib           # Stacked meta-model (XGBoost)
```

## Quick Start (Dev Environment)

```bash
# 1. (Optional) Create a Python virtualenv
python -m venv .venv && source .venv/bin/activate

# 2. Install requirements
pip install -r requirements.txt
# If missing, install:
pip install flask pandas numpy scikit-learn joblib catboost lightgbm xgboost

# 3. Place model files in project folder:
#    - catboost_model.joblib, lgbm_model.joblib, gbr_model.joblib, model1.joblib
#    - OR: Open pio_mll.ipynb, retrain, and save models via joblib.dump()

# 4. Run Flask app locally
python app.py
# Open http://127.0.0.1:5000/ in your browser
```

## Notebook Usage: [`pio_mll.ipynb`](projects/ml/pio_mll.ipynb)

- Explore and preprocess Airbnb listing data
- Analyze features (see: `data_overview`, `univariateAnalysis_numeric`, `univariateAnalysis_category`, `bivariate_num_num`)
- Select features: `selected_features`
- Train regressors: `cat_model`, `lgbm_model`, `gbr_model`
- Build stacked meta-model: `meta_model`
- Export all models using `joblib.dump(...)`
- Optionally export feature and allowed value lists for alignment tests

## App Usage: [`app.py`](projects/ml/app.py)

- Loads models: `catboost_model.joblib`, `lgbm_model.joblib`, `gbr_model.joblib`, `model1.joblib`
- Accepts form input, applies one-hot encoding to categorical fields
- Ensures input features align with `expected_features` (see variable)
- Predicts log(price), returns the exponentiated value

### Inputs & Feature Alignment

- Categorical fields one-hot encoded; columns aligned to `expected_features`
- For "Feature shape mismatch" errors:
  - Confirm model files were saved with the same feature set as `selected_features`
  - All expected one-hot columns must exist in submitted data

## Testing & Debugging

- Print `expected_features` in `app.py` and compare with form’s encoded columns
- Re-save the feature list (from notebook) for verification
- Re-save model files in case of failed loading (mismatched features)
- Add unit tests for feature alignment and prediction pipeline as needed

## Improvements Suggested

- Add a `requirements.txt` and `Dockerfile` for deployment
- Persist and reuse scalers/encoders for consistent preprocessing
- Expand unit tests for input and prediction code

## License

No open license included. Please add a LICENSE file if you plan to share/distribute this project.

***

**References:**

Key symbols visible in [pio_mll.ipynb](projects/ml/pio_mll.ipynb):  
`data_overview`, `univariateAnalysis_numeric`, `univariateAnalysis_category`, `bivariate_num_num`, `selected_features`, `cat_model`, `lgbm_model`, `gbr_model`, `meta_model`  
Key symbols in [app.py](projects/ml/app.py):  
`index`, `clear`, `expected_features`, `cat_model`

***

**End of README**
