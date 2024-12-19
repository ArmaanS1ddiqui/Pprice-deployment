import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("XGBoostModel2.pkl", "rb"))

# Route for the main page
@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

# Route for price prediction
@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    # Log user inputs
    print(location, bhk, bath, sqft)

    # Prepare input data for the model
    input_df = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = pipe.predict(input_df)[0] * 1e5

    # Format the prediction with commas
    formatted_prediction = "{:,.2f}".format(prediction)
    
    return formatted_prediction

# Route for EMI calculation
@app.route('/calculate_emi', methods=['POST'])
def calculate_emi():
    predicted_price = float(request.form.get('predicted_price'))
    interest_rate = float(request.form.get('interest_rate')) / 100 / 12  # Monthly interest rate
    years = int(request.form.get('years'))
    number_of_months = years * 12

    # EMI calculation formula
    emi = (predicted_price * interest_rate * (1 + interest_rate) ** number_of_months) / \
          ((1 + interest_rate) ** number_of_months - 1)

    # Format the EMI value
    formatted_emi = "{:,.2f}".format(emi)

    return formatted_emi

if __name__ == '__main__':
    app.run()
