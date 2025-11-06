from flask import Flask, render_template, request, jsonify, redirect, url_for
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained models
cat_model = joblib.load("catboost_model.joblib")
lgbm_model = joblib.load("lgbm_model.joblib")
gbr_model = joblib.load("gbr_model.joblib")
meta_model = joblib.load("model1.joblib")  # Stacking model

# Expected feature names (from training)
expected_features = cat_model.feature_names_

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Collect input values from the form
            bathrooms = float(request.form["bathrooms"])
            number_of_reviews = int(request.form["number_of_reviews"])
            accommodates = int(request.form["accommodates"])
            review_scores_rating = float(request.form["review_scores_rating"])
            bedrooms = int(request.form["bedrooms"])
            host_response_rate = int(request.form["host_response_rate"])
            beds = int(request.form["beds"])
            cleaning_fee = float(request.form["cleaning_fee"])
            room_type = request.form["room_type"]
            city = request.form["city"]
            property_type = request.form["property_type"]
            cancellation_policy = request.form["cancellation_policy"]
            bed_type = request.form["bed_type"]
            host_identity_verified = request.form["host_identity_verified"]
            instant_bookable = request.form["instant_bookable"]

            # Create a dataframe with input data
            input_df = pd.DataFrame({
                "bathrooms": [bathrooms],
                "number_of_reviews": [number_of_reviews],
                "accommodates": [accommodates],
                "review_scores_rating": [review_scores_rating],
                "bedrooms": [bedrooms],
                "host_response_rate": [host_response_rate],
                "beds": [beds],
                "cleaning_fee": [cleaning_fee],
                "room_type": [room_type],
                "city": [city],
                "property_type": [property_type],
                "cancellation_policy": [cancellation_policy],
                "bed_type": [bed_type],
                "host_identity_verified": [host_identity_verified],
                "instant_bookable": [instant_bookable]
            })

            # One-hot encode categorical variables
            input_encoded = pd.get_dummies(input_df)

            # Ensure all expected features exist
            for feature in expected_features:
                if feature not in input_encoded.columns:
                    input_encoded[feature] = 0

            # Reorder columns to match training
            input_encoded = input_encoded[expected_features]

            # Convert to numpy array
            input_array = input_encoded.to_numpy()

            # Ensure correct shape before prediction
            if input_array.shape[1] != len(expected_features):
                return jsonify({"error": f"Feature shape mismatch: expected {len(expected_features)}, got {input_array.shape[1]}"})

            # Model Predictions
            y_pred_cat = cat_model.predict(input_array)
            y_pred_lgbm = lgbm_model.predict(input_array)
            y_pred_gbr = gbr_model.predict(input_array)

            # Stacking model prediction
            stacked_predictions = np.column_stack((y_pred_cat, y_pred_lgbm, y_pred_gbr))
            y_pred_stack = meta_model.predict(stacked_predictions)

            prediction = round(np.exp(y_pred_stack[0]), 2) 

            return render_template("index.html", prediction=prediction)

        except Exception as e:
            return jsonify({"error": str(e)})

    return render_template("index.html", prediction=None)

@app.route("/clear", methods=["GET"])
def clear():
    """ Clears the prediction result and resets the form """
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
