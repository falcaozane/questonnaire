import pandas as pd
import numpy as np

# Define the leadership styles
leadership_styles = [
    'Authoritarian', 'Participative', 'Delegative', 'Pacesetting',
    'Transactional', 'Transformational', 'Visionary', 'Coaching', 'Bureaucratic'
]

def generate_synthetic_dataset(n_samples_per_style=1000):
    np.random.seed(42)  # for reproducibility
    
    # Define characteristic response patterns for each leadership style
    style_patterns = {
        'Authoritarian': [4, 5, 4, 2, 2, 3, 2, 1, 2, 4, 4, 5, 4, 3, 4, 3, 3, 3, 3, 3, 2, 2, 2, 4, 4, 5],
        'Participative': [2, 2, 2, 5, 5, 5, 4, 3, 4, 4, 4, 4, 4, 2, 3, 4, 4, 4, 5, 4, 4, 5, 4, 3, 3, 3],
        'Delegative': [1, 1, 1, 3, 3, 4, 5, 5, 5, 3, 3, 3, 2, 1, 1, 3, 3, 3, 4, 3, 3, 3, 3, 1, 1, 1],
        'Pacesetting': [3, 4, 3, 3, 4, 4, 3, 2, 3, 5, 5, 5, 4, 3, 4, 4, 5, 4, 4, 4, 4, 4, 3, 4, 4, 4],
        'Transactional': [3, 4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 4, 5, 3, 4, 3, 3, 3, 3, 4, 3, 4, 4, 4],
        'Transformational': [2, 2, 2, 4, 5, 5, 4, 3, 4, 4, 5, 4, 4, 2, 3, 5, 5, 5, 5, 4, 5, 5, 5, 3, 3, 3],
        'Visionary': [2, 2, 2, 4, 4, 5, 4, 3, 4, 4, 4, 4, 4, 2, 3, 5, 5, 5, 5, 5, 4, 4, 4, 3, 3, 3],
        'Coaching': [2, 2, 2, 4, 5, 5, 4, 3, 4, 4, 4, 4, 4, 2, 3, 4, 4, 4, 5, 4, 5, 5, 5, 3, 3, 3],
        'Bureaucratic': [3, 4, 4, 3, 3, 3, 2, 2, 2, 4, 4, 4, 3, 3, 4, 3, 3, 3, 3, 2, 3, 3, 3, 5, 5, 5]
    }
    
    data = []
    for style, pattern in style_patterns.items():
        for _ in range(n_samples_per_style):
            # Add some randomness to the pattern
            sample = [max(1, min(5, int(np.random.normal(p, 0.5)))) for p in pattern]
            data.append(sample + [style])
    
    # Use the specified headers
    columns = [f'Q{i}' for i in range(1, 27)] + ['leadership_style']
    df = pd.DataFrame(data, columns=columns)
    return df

# Generate the dataset
synthetic_data = generate_synthetic_dataset()

# Save the dataset to a CSV file
synthetic_data.to_csv('synthetic_leadership_data.csv', index=False)

print(f"Dataset generated and saved to 'synthetic_leadership_data.csv'")
print(f"Shape: {synthetic_data.shape}")
print("\nFirst few rows:")
print(synthetic_data.head())

print("\nLeadership style distribution:")
print(synthetic_data['leadership_style'].value_counts())