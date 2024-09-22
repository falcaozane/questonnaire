import pandas as pd

# Define the questions and leadership styles
questions = [f"Q{i}" for i in range(1, 27)]
leadership_styles = ["Authoritarian", "Participative", "Delegative", "Pacesetting", "Transactional", "Transformational", "Visionary", "Coaching", "Bureaucratic"]

# Define the answers for each leadership style
answers = [
    [1, 2, 3, 4, 5, 4, 1, 2, 3, 4, 5, 4, 1, 2, 3, 4, 5, 4, 5, 1, 2, 3, 4, 5, 4, 5],  # Authoritarian
    [2, 3, 4, 5, 1, 5, 2, 3, 4, 5, 1, 5, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],  # Participative
    [3, 4, 5, 1, 2, 3, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2],  # Delegative
    [4, 5, 1, 2, 3, 4, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3],  # Pacesetting
    [5, 1, 2, 3, 4, 5, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4],  # Transactional
    [1, 2, 3, 4, 5, 4, 1, 2, 3, 4, 5, 4, 1, 2, 3, 4, 5, 4, 5, 1, 2, 3, 4, 5, 4, 5],  # Transformational
    [2, 3, 4, 5, 1, 5, 2, 3, 4, 5, 1, 5, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],  # Visionary
    [3, 4, 5, 1, 2, 3, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2],  # Coaching
    [4, 5, 1, 2, 3, 4, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3]   # Bureaucratic
]

# Create the dataset
data = {question: [] for question in questions}
data["leadership_style"] = []
for i, style in enumerate(leadership_styles):
    for j, answer in enumerate(answers[i]):
        data[questions[j]].append(answer)
    data["leadership_style"].append(style)

# Convert the data to a Pandas DataFrame
df = pd.DataFrame(data)

# Print the dataset
#print(df)
df.to_csv('leadership_style_data-1.csv', index=False)

print("CSV file generated successfully.")