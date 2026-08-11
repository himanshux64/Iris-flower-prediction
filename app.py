import os
import sqlite3

import joblib
from flask import Flask, render_template, request, redirect, url_for


# =========================================================
# Flask Configuration
# =========================================================

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE_PATH = os.path.join(BASE_DIR, "IrisDatabase.db")
MODEL_PATH = os.path.join(BASE_DIR, "iris_model_new.pkl")


# =========================================================
# Load ML Model
# =========================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"ML model not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# =========================================================
# Database
# =========================================================

def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS iris_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sepal_length REAL NOT NULL,
            sepal_width REAL NOT NULL,
            petal_length REAL NOT NULL,
            petal_width REAL NOT NULL,
            variety TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


init_db()


# =========================================================
# Helper: Get Database Records
# =========================================================

def get_iris_data():
    conn = get_db_connection()

    data = conn.execute("""
        SELECT *
        FROM iris_data
        ORDER BY id DESC
    """).fetchall()

    conn.close()

    return data


# =========================================================
# Helper: Convert Model Prediction
# =========================================================

def get_species_name(prediction):
    """
    Handles models that return either:
        0, 1, 2
    or:
        'setosa', 'versicolor', 'virginica'
    """

    # Convert numpy values / other scalar values to Python value
    try:
        prediction = prediction.item()
    except AttributeError:
        pass

    # If model returns numeric class
    if isinstance(prediction, (int, float)):
        labels = {
            0: "Iris Setosa",
            1: "Iris Versicolor",
            2: "Iris Virginica"
        }

        prediction = int(prediction)

        if prediction in labels:
            return labels[prediction]

    # If model returns string class
    prediction_string = str(prediction).strip().lower()

    label_map = {
        "setosa": "Iris Setosa",
        "iris setosa": "Iris Setosa",

        "versicolor": "Iris Versicolor",
        "iris versicolor": "Iris Versicolor",

        "virginica": "Iris Virginica",
        "iris virginica": "Iris Virginica"
    }

    if prediction_string in label_map:
        return label_map[prediction_string]

    # Fallback
    return str(prediction)


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    iris_data = get_iris_data()

    return render_template(
        "index.html",
        iris_data=iris_data,
        prediction=None,
        error=None
    )


# =========================================================
# PREDICT
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ---------------------------------------------
        # Get form values
        # ---------------------------------------------

        sepal_length = float(
            request.form.get("sepal_length", "").strip()
        )

        sepal_width = float(
            request.form.get("sepal_width", "").strip()
        )

        petal_length = float(
            request.form.get("petal_length", "").strip()
        )

        petal_width = float(
            request.form.get("petal_width", "").strip()
        )

        # ---------------------------------------------
        # Basic validation
        # ---------------------------------------------

        values = [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]

        if any(value <= 0 for value in values):
            raise ValueError("Measurements must be positive.")

        # ---------------------------------------------
        # ML Prediction
        # ---------------------------------------------

        features = [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]]

        result = model.predict(features)[0]

        # ---------------------------------------------
        # Convert prediction to species
        # ---------------------------------------------

        output = get_species_name(result)

        # ---------------------------------------------
        # Database records
        # ---------------------------------------------

        iris_data = get_iris_data()

        # ---------------------------------------------
        # Render result
        # ---------------------------------------------

        return render_template(
            "index.html",
            iris_data=iris_data,
            prediction=output,
            error=None,

            # Send measurements to template
            prediction_measurements={
                "sepal_length": sepal_length,
                "sepal_width": sepal_width,
                "petal_length": petal_length,
                "petal_width": petal_width
            }
        )

    except (ValueError, TypeError, KeyError) as error:

        print("Prediction error:", error)

        iris_data = get_iris_data()

        return render_template(
            "index.html",
            iris_data=iris_data,
            prediction=None,
            error="Please enter valid flower measurements."
        )


    except Exception as error:

        print("ML prediction error:", error)

        iris_data = get_iris_data()

        return render_template(
            "index.html",
            iris_data=iris_data,
            prediction=None,
            error="Unable to make prediction. Please try again."
        )


# =========================================================
# ADD DATA
# =========================================================

@app.route("/add", methods=["POST"])
def add():

    try:

        sepal_length = float(
            request.form.get("sepal_length", "").strip()
        )

        sepal_width = float(
            request.form.get("sepal_width", "").strip()
        )

        petal_length = float(
            request.form.get("petal_length", "").strip()
        )

        petal_width = float(
            request.form.get("petal_width", "").strip()
        )

        variety = request.form.get("variety", "").strip()

        if not variety:
            return redirect(url_for("home"))

        conn = get_db_connection()

        conn.execute("""
            INSERT INTO iris_data (
                sepal_length,
                sepal_width,
                petal_length,
                petal_width,
                variety
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            sepal_length,
            sepal_width,
            petal_length,
            petal_width,
            variety
        ))

        conn.commit()
        conn.close()

        return redirect(url_for("home"))

    except (ValueError, TypeError):

        return redirect(url_for("home"))


# =========================================================
# DELETE DATA
# =========================================================

@app.route("/delete/<int:id>")
def delete(id):

    conn = get_db_connection()

    conn.execute(
        """
        DELETE FROM iris_data
        WHERE id = ?
        """,
        (id,)
    )

    conn.commit()
    conn.close()

    return redirect(url_for("home"))


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )