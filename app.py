

from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load models
daily_model = joblib.load("sparkathon-randomforest-daily.pkl")
monthly_model = joblib.load("sparkathon-randomforest-monthly.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    daily_result = None
    monthly_result = None

    if request.method == "POST":
        year = int(request.form["year"])
        month = int(request.form["month"])

        # Monthly prediction
        monthly_input = pd.DataFrame(np.array([[year, month]]), columns=["Year", "Month"])
        monthly_result = monthly_model.predict(monthly_input)[0]

        # Daily prediction (only if day is provided)
        day = request.form.get("day")
        if day:
            day = int(day)
            daily_input = pd.DataFrame(np.array([[year, month, day]]), columns=["Year", "Month", "Day"])
            daily_result = daily_model.predict(daily_input)[0]

    return render_template("index.html", daily_result=daily_result, monthly_result=monthly_result)

if __name__ == "__main__":
    app.run(debug=True)


