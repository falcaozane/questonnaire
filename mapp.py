import streamlit as st
import pandas as pd
import altair as alt
import plotly.graph_objects as go
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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
    "Do you help your employees / teammates develop new skills and improve their performance?",
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

# Define the leadership styles
leadership_styles = [
    'Authoritarian', 'Participative', 'Delegative', 'Pacesetting',
    'Transactional', 'Transformational', 'Visionary', 'Coaching', 'Bureaucratic'
]

# Initialize the Multi-Layer Perceptron model
mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

# Function to train the MLP model
def train_model(data):
    X = data.drop('leadership_style', axis=1)
    y = data['leadership_style']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mlp_model.fit(X_train, y_train)
    y_pred = mlp_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, X_test, y_test, y_pred

# Function to predict leadership style
def predict_style(responses):
    return mlp_model.predict([responses])[0]

# Create the Streamlit app
st.title("Leadership Style Questionnaire")

# Sidebar for dataset validation
st.sidebar.title("Dataset Validation")
uploaded_file = st.sidebar.file_uploader("Upload sample dataset (CSV)", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    accuracy, X_test, y_test, y_pred = train_model(data)
    
    st.sidebar.write(f"Model Accuracy on Test Set: {accuracy:.2%}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=leadership_styles, yticklabels=leadership_styles)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.sidebar.pyplot(plt)

# Main questionnaire
st.write("Please answer the following questions about your leadership style:")
responses = []
for question in questions:
    response = st.radio(question, ('Never', 'Rarely', 'Sometimes', 'Often', 'Always'))
    responses.append(response_mapping[response])

if st.button("Submit"):
    # Predict the leadership style
    predicted_style = predict_style(responses)

    # Display the result
    st.markdown(f"Your predicted leadership style is: **{predicted_style}**")

    # Get probabilities for each style
    probabilities = mlp_model.predict_proba([responses])[0]
    style_probs = dict(zip(leadership_styles, probabilities))

    # Create a DataFrame for the style probabilities
    df = pd.DataFrame(list(style_probs.items()), columns=['Style', 'Probability'])

    # Create a bar chart using Altair
    chart = alt.Chart(df).mark_bar().encode(
        x='Style',
        y='Probability',
        color=alt.condition(
            alt.datum.Style == predicted_style,
            alt.value('yellow'),
            alt.value('steelblue')
        )
    ).properties(
        title='Leadership Style Probabilities'
    )

    # Display the chart
    st.altair_chart(chart, use_container_width=True)

    # Prepare data for radar chart
    radar_data = pd.DataFrame(style_probs, index=['Probability']).transpose()
    radar_data['angle'] = np.linspace(0, 2*np.pi, len(radar_data), endpoint=False)

    # Create radar chart
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=radar_data['Probability'],
        theta=radar_data.index,
        fill='toself',
        name='Leadership Style',
        marker=dict(color='rgba(255, 190, 250, 0.8)')
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(style_probs.values())]
            )),
        showlegend=False,
        title='Leadership Style Radar Chart'
    )

    # Display the radar chart
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""summary_line
    
    Authoritarian: Questions 1, 2, 3, and 13 (emphasizing control and external motivation).
    Participative: Questions 4, 5, 6, and 7 (emphasizing collaboration, input, and autonomy).
    Delegative: Questions 7, 8, and 9 (emphasizing autonomy and minimal guidance).
    Pacesetting: Questions 10, 11, and 12 (emphasizing high standards and performance).
    Transactional: Questions 13, 14, and 15 (emphasizing rewards, punishments, and clear systems).
    Transformational: Questions 16, 17, 18, and 19 (emphasizing inspiration, vision, and trust).
    Visionary: Questions 18, 19, and 20 (emphasizing vision, trust, and growth).
    Coaching: Questions 21, 22, and 23 (emphasizing development, feedback, and support).
    Bureaucratic: Questions 25, 26, and 27 (emphasizing rules, regulations, and consistency).
    """
    )
