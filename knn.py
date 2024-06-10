import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st

# Load the dataset
dataset = pd.read_csv('fruit.csv')

# Verify if 'Color' column exists in the dataset
if 'Color' in dataset.columns and 'Texture' in dataset.columns:
    # Prepare data for training
    X = dataset[['Weight (grams)', 'Color', 'Texture']]
    y = dataset['Fruit Type']

    # Convert categorical variables to one-hot encoding
    X = pd.get_dummies(X, columns=['Color', 'Texture'])

    # Train the KNN model
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)

    # Streamlit UI
    st.title('Fruit Type Prediction')

    st.write('Enter fruit attributes to predict the fruit type.')

    weight = st.text_input('Weight (grams)')
    color = st.selectbox('Color', dataset['Color'].unique())
    texture = st.selectbox('Texture', dataset['Texture'].unique())

    # Create a predict button
    if st.button('Predict'):
        # Check if all inputs are provided
        if weight and color and texture:
            # Convert weight to float
            weight = float(weight)

            # Convert color and texture to one-hot encoded features
            color_values = [1 if c == color else 0 for c in dataset['Color'].unique()]
            texture_values = [1 if t == texture else 0 for t in dataset['Texture'].unique()]

            # Combine all features into one array
            input_features = [weight] + color_values + texture_values

            # Make prediction
            prediction = model.predict([input_features])[0]

            # Display the predicted fruit type
            st.write('Prediction:')
            st.write(prediction)
else:
    st.write("Error: 'Color' or 'Texture' column not found in the dataset.")










