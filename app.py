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
        f"ML model not found at: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# =========================================================
# Database Helper
# =========================================================

def get_db_connection():
    """
    Create and return a SQLite database connection.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# =========================================================
# Initialize Database
# =========================================================

def init_db():
    conn = get_db_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS iris_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sepal_length REAL,
            sepal_width REAL,
            petal_length REAL,
            petal_width REAL,
            variety TEXT
        )
    """)

    conn.commit()
    conn.close()


init_db()


# =========================================================
# Home
# =========================================================

@app.route("/")
def home():
    conn = get_db_connection()

    iris_data = conn.execute("""
        SELECT *
        FROM iris_data
        ORDER BY id DESC
    """).fetchall()

    conn.close()

    return render_template(
        "index.html",
        iris_data=iris_data
    )


# =========================================================
# Add Data
# =========================================================

@app.route("/add", methods=["POST"])
def add():

    try:
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])
        variety = request.form["variety"]

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

    except (ValueError, KeyError):
        return redirect(url_for("home"))


# =========================================================
# Delete Data
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
# Predict Iris Species
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ---------------------------------------------
        # Get input values
        # ---------------------------------------------

        sepal_length = float(
            request.form["sepal_length"]
        )

        sepal_width = float(
            request.form["sepal_width"]
        )

        petal_length = float(
            request.form["petal_length"]
        )

        petal_width = float(
            request.form["petal_width"]
        )

        # ---------------------------------------------
        # ML Prediction
        # ---------------------------------------------

        result = model.predict([
            [
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ]
        ])[0]

        # ---------------------------------------------
        # Convert prediction to species name
        # ---------------------------------------------

        labels = [
            "Setosa",
            "Versicolor",
            "Virginica"
        ]

        output = labels[int(result)]

        # ---------------------------------------------
        # Fetch database records
        # ---------------------------------------------

        conn = get_db_connection()

        iris_data = conn.execute("""
            SELECT *
            FROM iris_data
            ORDER BY id DESC
        """).fetchall()

        conn.close()

        # ---------------------------------------------
        # Render result
        # ---------------------------------------------

        return render_template(
            "index.html",
            iris_data=iris_data,
            prediction=output
        )

    except (ValueError, KeyError, IndexError):

        conn = get_db_connection()

        iris_data = conn.execute("""
            SELECT *
            FROM iris_data
            ORDER BY id DESC
        """).fetchall()

        conn.close()

        return render_template(
            "index.html",
            iris_data=iris_data,
            error="Please enter valid flower measurements."
        )


# =========================================================
# Run Flask Application
# =========================================================

if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )