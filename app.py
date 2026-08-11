from flask import Flask, render_template, request, redirect
import sqlite3
import joblib
import os

app = Flask(__name__)

# Load ML Model safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'iris_model_new.pkl')
model = joblib.load(model_path)

# Initialize DB
def init_db():
    conn = sqlite3.connect('IrisDatabase.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS iris_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sepal_length REAL,
            sepal_width REAL,
            petal_length REAL,
            petal_width REAL,
            variety TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Home
@app.route('/')
def home():
    conn = sqlite3.connect('IrisDatabase.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM iris_data")
    iris_data = cursor.fetchall()
    conn.close()
    return render_template('index.html', iris_data=iris_data)

# Add Data
@app.route('/add', methods=['POST'])
def add():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    variety = request.form['variety']

    conn = sqlite3.connect('IrisDatabase.db')
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO iris_data (sepal_length, sepal_width, petal_length, petal_width, variety) VALUES (?, ?, ?, ?, ?)",
        (sepal_length, sepal_width, petal_length, petal_width, variety)
    )
    conn.commit()
    conn.close()

    return redirect('/')

# Delete Data
@app.route('/delete/<int:id>')
def delete(id):
    conn = sqlite3.connect('IrisDatabase.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM iris_data WHERE id=?", (id,))
    conn.commit()
    conn.close()

    return redirect('/')

# Predict
@app.route('/predict', methods=['POST'])
def predict():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]

    # Proper labels
    labels = ['Setosa', 'Versicolor', 'Virginica']
    output = labels[result]

    conn = sqlite3.connect('IrisDatabase.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM iris_data")
    iris_data = cursor.fetchall()
    conn.close()

    return render_template('index.html', iris_data=iris_data, prediction=output)

if __name__ == '__main__':
    app.run(debug=True)