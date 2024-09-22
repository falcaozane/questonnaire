import pandas as pd

# Define the leadership styles and their corresponding response patterns
leadership_styles = {
    'Authoritarian': [1, 2, 3, 4, 5, 4, 1, 2, 3, 4, 5, 4, 1, 2, 3, 4, 5, 4, 5, 1, 2, 3, 4, 5, 4, 5, 1],
    'Participative': [2, 3, 4, 5, 1, 5, 2, 3, 4, 5, 1, 5, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1],
    'Delegative': [3, 4, 5, 1, 2, 3, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3],
    'Pacesetting': [4, 5, 1, 2, 3, 4, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4],
    'Transactional': [5, 1, 2, 3, 4, 5, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    'Transformational': [1, 2, 3, 4, 5, 4, 1, 2, 3, 4, 5, 4, 1, 2, 3, 4, 5, 4, 5, 1, 2, 3, 4, 5, 4, 5, 1],
    'Visionary': [2, 3, 4, 5, 1, 5, 2, 3, 4, 5, 1, 5, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1],
    'Coaching': [3, 4, 5, 1, 2, 3, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3],
    'Bureaucratic': [4, 5, 1, 2, 3, 4, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4],
}

# Create a list of lists to store the data
data = []

# Iterate through each leadership style and add a row to the data
for style, responses in leadership_styles.items():
    data.append([style] + responses)

# Create a DataFrame
df = pd.DataFrame(data, columns=['leadership_style'] + [f'Q{i}' for i in range(1, 28)])

# Save the DataFrame to a CSV file
df.to_csv('leadership_style_data.csv', index=False)

print("CSV file generated successfully.")