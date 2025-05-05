from flask import Flask, render_template, request
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Initialize Flask app
app = Flask(__name__)

# Load model and encoder
model = joblib.load('salary_predictor_with_field.pkl')
encoder = joblib.load('field_label_encoder.pkl')

# Home page (GET) and form submission (POST)
@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_salary = None
    error = None
    fields = list(encoder.classes_)

    if request.method == 'POST':
        try:
            # Get form input
            years = float(request.form['years_experience'])
            field = request.form['field']

            # Encode the field
            field_encoded = encoder.transform([field])[0]
            input_data = np.array([[years, field_encoded]])

            prediction = model.predict(input_data)
            predicted_salary = max(0, min(2_00_00_000, prediction[0]))

            # Prepare values for display
            predicted_yearly = f"₹{int(predicted_salary):,}/year"
            predicted_monthly = f"≈ ₹{int(predicted_salary // 12):,}/month"

        except Exception as e:
            error = str(e)

    return render_template(
    'index.html',
    fields=fields,
    prediction=predicted_yearly,
    monthly=predicted_monthly,
    error=error,
    selected_years=years,
    selected_field=field
)


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
