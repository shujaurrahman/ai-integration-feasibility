import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)
CORS(app)

# Load the model and preprocessing objects
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
label_encoders = joblib.load('label_encoders.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        business_type = request.form['business_type']
        complexity_level = request.form['complexity_level']
        data_availability = request.form['data_availability']
        annual_revenue = float(request.form['annual_revenue'])
        employee_count = int(request.form['employee_count'])
        current_tech_investment = float(request.form['current_tech_investment'])
        customer_satisfaction = float(request.form['customer_satisfaction'])
        operational_efficiency = float(request.form['operational_efficiency'])
        workforce_skill_level = float(request.form['workforce_skill_level'])
        ai_integration_cost = float(request.form['ai_integration_cost'])
        potential_improvement = float(request.form['potential_improvement'])

        # Preprocess the input data
        features = np.array([
            business_type,
            complexity_level,
            data_availability,
            annual_revenue,
            employee_count,
            current_tech_investment,
            customer_satisfaction,
            operational_efficiency,
            workforce_skill_level,
            ai_integration_cost,
            potential_improvement
        ]).reshape(1, -1)

        # Label encoding for categorical features
        features[0, 0] = label_encoders['business_type'].transform([features[0, 0]])[0]
        features[0, 1] = label_encoders['complexity_level'].transform([features[0, 1]])[0]
        features[0, 2] = label_encoders['data_availability'].transform([features[0, 2]])[0]

        # Standardize the features using the scaler
        features_scaled = scaler.transform(features[:, 3:])
        features[:, 3:] = features_scaled

        # Make the prediction
        prediction = model.predict(features)

        # Set prediction result message
        if prediction[0] == 1:
            result = "AI integration is beneficial for this business."
        else:
            result = "AI integration is not beneficial for this business."

        # Return the prediction result along with the input data
        return render_template('result.html', 
                               result=result,
                               business_type=business_type,
                               annual_revenue=annual_revenue,
                               employee_count=employee_count,
                               current_tech_investment=current_tech_investment,
                               customer_satisfaction=customer_satisfaction,
                               operational_efficiency=operational_efficiency,
                               workforce_skill_level=workforce_skill_level,
                               ai_integration_cost=ai_integration_cost,
                               potential_improvement=potential_improvement,
                               complexity_level=complexity_level,
                               data_availability=data_availability)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
