import streamlit as st
import pandas as pd
import altair as alt
import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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

# Define the number of clusters (same as the number of leadership styles)
n_clusters = len(leadership_styles)
kmeans = KMeans(n_clusters=n_clusters, random_state=40)

# Function to perform clustering
def cluster_respondents(data):
    X = data.drop('leadership_style', axis=1)
    
    # Standardize the data (K-Means is sensitive to scale)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit the KMeans clustering algorithm
    kmeans = KMeans(n_clusters=n_clusters, random_state=40)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add the cluster labels to the data
    data['cluster'] = clusters
    
    return data, kmeans

# Function to predict the cluster for a new respondent
def predict_cluster(responses, kmeans, scaler):
    responses_scaled = scaler.transform([responses])
    return kmeans.predict(responses_scaled)[0]

# Create the Streamlit app
st.title("Leadership Style Questionnaire - Clustering")

# Sidebar for dataset validation
st.sidebar.title("Dataset Validation")
uploaded_file = st.sidebar.file_uploader("Upload sample dataset (CSV)", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Perform clustering on the dataset
    clustered_data, kmeans = cluster_respondents(data)
    
    st.sidebar.write(f"Cluster Centers: {kmeans.cluster_centers_}")
    
    # Visualize clusters (using the first two dimensions for simplicity)
    plt.figure(figsize=(10,8))
    sns.scatterplot(x=clustered_data.iloc[:, 0], y=clustered_data.iloc[:, 1], hue=clustered_data['cluster'], palette='tab10')
    plt.title('Respondents Cluster Visualization')
    st.sidebar.pyplot(plt)
    
    # Show the number of respondents in each cluster
    cluster_counts = clustered_data['cluster'].value_counts()
    st.sidebar.write("Number of respondents in each cluster:")
    st.sidebar.dataframe(cluster_counts)

# Main questionnaire
st.write("Please answer the following questions about your leadership style:")
responses = []
for question in questions:
    response = st.radio(question, ('Never', 'Rarely', 'Sometimes', 'Often', 'Always'))
    responses.append(response_mapping[response])

if st.button("Submit"):
    # Predict the cluster for the new respondent
    predicted_cluster = predict_cluster(responses, kmeans, scaler)
    
    # Map the cluster to a leadership style
    predicted_style = leadership_styles[predicted_cluster]

    # Display the result
    st.markdown(f"Based on your responses, your leadership style is most likely: **:orange[{predicted_style}]**")

    # Display the distribution of leadership styles (clusters)
    style_counts = clustered_data['cluster'].value_counts().sort_index()
    style_distribution = pd.DataFrame({
        'Style': leadership_styles,
        'Count': style_counts
    })

    # Create a bar chart using Altair
    chart = alt.Chart(style_distribution).mark_bar().encode(
        x='Style',
        y='Count',
        color='Style'
    ).properties(
        title='Leadership Style Distribution (Clusters)'
    )

    # Display the chart
    st.altair_chart(chart, use_container_width=True)

    # Create radar chart
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=style_counts,
        theta=leadership_styles,
        fill='toself',
        name='Leadership Style',
        marker=dict(color='rgba(255, 190, 250, 0.8)')
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(style_counts)]
            )),
        showlegend=False,
        title='Leadership Style Radar Chart'
    )

    # Display the radar chart
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    Leadership styles were clustered based on your responses. Each cluster represents a dominant style.
    """)

