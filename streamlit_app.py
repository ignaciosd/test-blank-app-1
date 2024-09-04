#import streamlit as st

#st.title("🎈 My new app")
#st.write(
#    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
#)


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Set the title
st.title("Streamlit App with Sklearn Dataset and Random Forest")

# Load the Iris dataset from sklearn
st.subheader("Iris Dataset")
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Show the dataset
st.write("Dataset Preview:")
st.dataframe(df.head())

# Show some basic statistics
st.write("Basic Statistics:")
st.write(df.describe())

# Plotting the data
st.subheader("Data Visualization")
st.write("Scatter plot of the Iris dataset:")

# Pairplot using seaborn
sns.pairplot(df, hue="target", markers=["o", "s", "D"])
st.pyplot(plt)

# Feature importance using Random Forest
st.subheader("Random Forest Example")

# Splitting data into train and test
X = df[iris.feature_names]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Display accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy:.2f}")

st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Feature importance plot
st.write("Feature Importance:")
feature_importances = pd.DataFrame(model.feature_importances_,
                                   index=iris.feature_names,
                                   columns=['importance']).sort_values('importance', ascending=False)

st.bar_chart(feature_importances)



