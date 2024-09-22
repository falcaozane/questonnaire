import streamlit as st
import pandas as pd
import altair as alt
import plotly.graph_objects as go
import numpy as np

# Define the questionnaire
questions = [
    "Do you prefer to make decisions without much input from your team?",
    "Do you often expect your team members to follow your instructions without question?",
    "Do you believe that strict control is necessary for achieving goals?",
    "Do you frequently seek input from your team members when making decisions?",
    "Do you encourage open communication and collaboration within your team?",
    "Do you value everyone's opinions and ideas?",
    "Do you often let your team members decide how to complete tasks?",
    "Do you provide minimal guidance and expect your team to self-manage?",
    "Do you believe in giving your team members a lot of autonomy?",
    "Do you set high standards for performance and quality?",
    "Do you often lead by example, demonstrating high performance yourself?",
    "Do you expect your team members to meet these high standards?",
    "Do you use rewards and recognition to motivate your team?",
    "Do you use punishments or consequences to correct poor performance?",
    "Do you believe in a clear system of rewards and punishments to drive results?",
    "Do you inspire your team members to achieve more than they thought possible?",
    "Do you often challenge your team to reach higher goals?",
    "Do you inspire employees with a clear vision for the future?",
    "Do you earn trust by being transparent and honest?",
    "Are you particularly helpful for organizations that are growing quickly?",
    "Do you help your employees develop new skills and improve their performance?",
    "Do you provide regular feedback and support to your team members?",
    "Do you focus on long-term development rather than short-term results?",
    "Do you follow established processes and regulations strictly?",
    "Do you believe in adhering to policies and procedures without deviation?",
    "Do you enforce rules and regulations consistently?"
]

# Define the mapping of responses to numerical values
response_mapping = {
    'Never': 1,
    'Rarely': 2,
    'Sometimes': 3,
    'Often': 4,
    'Always': 5
}

# Define the mapping of questions to leadership styles
style_mapping = {
    'Authoritarian': [1, 2, 3],
    'Participative': [4, 5, 6],
    'Delegative': [7, 8, 9],
    'Pacesetting': [10, 11, 12],
    'Transactional': [13, 14, 15],
    'Transformational': [16, 17, 18],
    'Visionary': [19, 20, 21],
    'Coaching': [22, 23, 24],
    'Bureaucratic': [25, 26, 27]
}

# Function to calculate leadership style scores based on responses
def calculate_style_scores(responses):
    style_scores = {style: 0 for style in style_mapping.keys()}
    
    for i, response in enumerate(responses, 1):
        for style, questions in style_mapping.items():
            if i in questions:
                style_scores[style] += response

    best_style = max(style_scores, key=style_scores.get)
    return best_style, style_scores

# Function to process the sample dataset
def process_sample_dataset(file):
    df = pd.read_csv(file)
    actual_styles = df['leadership_style'].tolist()
    predicted_styles = []

    for _, row in df.iterrows():
        responses = row.iloc[:26].tolist()  # Exclude the last column (actual leadership style)
        best_style, _ = calculate_style_scores(responses)
        predicted_styles.append(best_style)

    accuracy = sum(a == p for a, p in zip(actual_styles, predicted_styles)) / len(actual_styles)
    return actual_styles, predicted_styles, accuracy

# Create the Streamlit app
st.title("Leadership Style Questionnaire")

# Sidebar for dataset validation
st.sidebar.title("Dataset Validation")
uploaded_file = st.sidebar.file_uploader("Upload sample dataset (CSV)", type="csv")

if uploaded_file is not None:
    actual_styles, predicted_styles, accuracy = process_sample_dataset(uploaded_file)
    
    st.sidebar.write(f"Model Accuracy on Sample Dataset: {accuracy:.2%}")
    
    comparison_df = pd.DataFrame({
        'Actual Style': actual_styles,
        'Predicted Style': predicted_styles
    })
    
    st.sidebar.write("Comparison of Actual vs Predicted Styles:")
    st.sidebar.dataframe(comparison_df)

    confusion_matrix = pd.crosstab(comparison_df['Actual Style'], comparison_df['Predicted Style'])
    st.sidebar.write("Confusion Matrix:")
    st.sidebar.dataframe(confusion_matrix)

# Main questionnaire
st.write("Please answer the following questions about your leadership style:")
responses = []
for question in questions:
    response = st.radio(question, ('Never', 'Rarely', 'Sometimes', 'Often', 'Always'))
    responses.append(response_mapping[response])

# Calculate the leadership style
best_style, style_scores = calculate_style_scores(responses)

# Display the result
st.markdown(f"Your predicted leadership style is: **:orange[{best_style}]**")

# Create a DataFrame for the style scores
df = pd.DataFrame(list(style_scores.items()), columns=['Style', 'Score'])

# Create a bar chart using Altair
chart = alt.Chart(df).mark_bar().encode(
    x='Style',
    y='Score',
    color=alt.condition(
        alt.datum.Style == best_style,
        alt.value('yellow'),  # The bar for the best style will be orange
        alt.value('steelblue')  # Other bars will be steel blue
    )
).properties(
    title='Leadership Style Scores'
)

# Display the chart
st.altair_chart(chart, use_container_width=True)

# Display the full style score breakdown
st.write("Here is the score breakdown for each leadership style:")
for style, score in style_scores.items():
    st.write(f"{style}: {score}")

# Prepare data for radar chart
radar_data = pd.DataFrame(style_scores, index=['Score']).transpose()
radar_data['angle'] = np.linspace(0, 2*np.pi, len(radar_data), endpoint=False)

# Create radar chart
fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=radar_data['Score'],
    theta=radar_data.index,
    fill='toself',
    name='Leadership Style'
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, max(style_scores.values())]
        )),
    showlegend=False,
    title='Leadership Style Radar Chart'
)

# Display the radar chart
st.plotly_chart(fig, use_container_width=True)