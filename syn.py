import pandas as pd
import numpy as np

# Define the leadership styles
leadership_styles = [
    'Authoritarian', 'Participative', 'Delegative', 'Pacesetting',
    'Transactional', 'Transformational', 'Visionary', 'Coaching', 'Bureaucratic'
]

def generate_synthetic_dataset(n_samples_per_style=1000):
    np.random.seed(42)  # for reproducibility
    
    # Define base patterns (neutral responses)
    base_pattern = [3] * 26
    
    # Define characteristic response patterns for each leadership style
    style_patterns = {
        'Authoritarian': base_pattern.copy(),
        'Participative': base_pattern.copy(),
        'Delegative': base_pattern.copy(),
        'Pacesetting': base_pattern.copy(),
        'Transactional': base_pattern.copy(),
        'Transformational': base_pattern.copy(),
        'Visionary': base_pattern.copy(),
        'Coaching': base_pattern.copy(),
        'Bureaucratic': base_pattern.copy()
    }
    
    # Modify specific questions for each style
    for i in [0, 1, 2, 12]:  # Q1, Q2, Q3, Q13
        style_patterns['Authoritarian'][i] = 5
    
    for i in [3, 4, 5, 6]:  # Q4, Q5, Q6, Q7
        style_patterns['Participative'][i] = 5
    
    for i in [6, 7, 8]:  # Q7, Q8, Q9
        style_patterns['Delegative'][i] = 5
    
    for i in [9, 10, 11]:  # Q10, Q11, Q12
        style_patterns['Pacesetting'][i] = 5
    
    for i in [12, 13, 14]:  # Q13, Q14, Q15
        style_patterns['Transactional'][i] = 5
    
    for i in [15, 16, 17, 18]:  # Q16, Q17, Q18, Q19
        style_patterns['Transformational'][i] = 5
    
    for i in [17, 18, 19]:  # Q18, Q19, Q20
        style_patterns['Visionary'][i] = 5
    
    for i in [20, 21, 22]:  # Q21, Q22, Q23
        style_patterns['Coaching'][i] = 5
    
    for i in [24, 25]:  # Q25, Q26
        style_patterns['Bureaucratic'][i] = 5
    
    data = []
    for style, pattern in style_patterns.items():
        for _ in range(n_samples_per_style):
            # Add randomness to the pattern
            sample = [max(1, min(5, int(np.random.normal(p, 0.7)))) for p in pattern]
            data.append(sample + [style])
    
    # Use the specified headers
    columns = [f'Q{i}' for i in range(1, 27)] + ['leadership_style']
    df = pd.DataFrame(data, columns=columns)
    return df

# Generate the dataset
synthetic_data = generate_synthetic_dataset()

# Save the dataset to a CSV file
synthetic_data.to_csv('synthetic_leadership_data_1.csv', index=False)

print(f"Dataset generated and saved to 'synthetic_leadership_data.csv'")
print(f"Shape: {synthetic_data.shape}")
print("\nFirst few rows:")
print(synthetic_data.head())

print("\nLeadership style distribution:")
print(synthetic_data['leadership_style'].value_counts())

# Print mean values for each question by leadership style
print("\nMean values for each question by leadership style:")
print(synthetic_data.groupby('leadership_style').mean().round(2).T)