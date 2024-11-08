import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)
n_samples = 1000

# Generate synthetic data
data = {
    'business_type': np.random.choice(['Retail', 'Manufacturing', 'Healthcare', 'Finance', 'Tech'], n_samples),
    'annual_revenue': np.random.uniform(1e6, 1e9, n_samples),
    'employee_count': np.random.randint(50, 5000, n_samples),
    'current_tech_investment': np.random.uniform(1e5, 1e7, n_samples),
    'customer_satisfaction': np.random.uniform(1, 10, n_samples),
    'operational_efficiency': np.random.uniform(0.5, 1.0, n_samples),
    'complexity_level': np.random.choice(['Low', 'Medium', 'High'], n_samples),
    'data_availability': np.random.choice(['Poor', 'Moderate', 'Rich'], n_samples),
    'workforce_skill_level': np.random.uniform(1, 10, n_samples),
    'ai_integration_cost': np.random.uniform(1e5, 5e6, n_samples),
}

# Calculating potential improvement score
data['potential_improvement'] = (
    (data['employee_count'] / 5000) *
    (data['customer_satisfaction'] / 10) *
    (data['operational_efficiency'])
)

# Determine if AI is beneficial based on the improvement threshold
data['ai_beneficial'] = (data['potential_improvement'] > 0.3).astype(int)

# Create DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv('business_data.csv', index=False)
print("Synthetic data generated and saved as 'business_data.csv'")
