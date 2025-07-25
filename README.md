🏡 Machine Learning Model Deployment: Boston House Price Prediction
This repository contains a complete machine learning pipeline, from data exploration and model training to an interactive web application deployed using Streamlit. The project focuses on predicting median house prices in Boston using the classic Boston Housing Prices dataset.

🎯 Project Overview
The main objective of this assignment was to build an end-to-end machine learning solution:

Data Analysis & Preprocessing: Explore, clean, and prepare the Boston Housing dataset.

Model Training: Train and evaluate multiple regression models to predict house prices.

Streamlit Web Application: Create an interactive, user-friendly web interface for data exploration, visualization, and real-time predictions.

Cloud Deployment: Deploy the Streamlit application to Streamlit Cloud for public access.

✨ Features of the Streamlit App
The Streamlit application (app.py) provides the following functionalities:

Data Exploration:

Display dataset overview (shape, columns, data types).

Show sample data and descriptive statistics.

Interactive filtering options to view specific rows or columns.

Visualizations:

Interactive plots including the distribution of house prices, feature-vs-price scatter plots, and a correlation heatmap.

Allows users to select features for dynamic visualization.

Model Prediction:

Input widgets for users to enter various house features.

Real-time display of the predicted median house value based on user inputs.

Includes basic error handling for inputs.

Model Performance:

Displays key regression evaluation metrics (Mean Squared Error, R-squared).

Visualizes actual vs. predicted values and residuals plots to assess model fit.

📊 Dataset
This project utilizes the Boston Housing Prices Dataset, a classic dataset for regression tasks. It contains 13 features describing various aspects of housing and neighborhoods in Boston, and the target variable is the median value of owner-occupied homes (in $1000s).

Source: Commonly available through Kaggle or scikit-learn (though load_boston is deprecated).

Target Variable: medv (median value of owner-occupied homes in $1000s).

🧠 Model
Two regression algorithms were trained and compared:

Linear Regression

Random Forest Regressor

The Random Forest Regressor was selected as the best-performing model based on its evaluation metrics (e.g., lower Mean Squared Error and higher R-squared) on the test dataset. This trained model, along with the data scaler and feature names, is saved as .pkl files for use in the Streamlit application.

📁 Project Structure
The repository is organized as follows:

your-project/<br>
├── app.py                      # Streamlit web application code<br>
├── requirements.txt            # List of Python dependencies<br>
├── model.pkl                   # Saved trained machine learning model<br>
├── scaler.pkl                  # Saved StandardScaler object for feature scaling<br>
├── features.pkl                # Saved list of feature names (for consistent input order)<br>
├── data/<br>
│   └── BostonHousing.csv       # The dataset file<br>
└── notebooks/<br>
    └── model_training.ipynb    # Jupyter Notebook for EDA, preprocessing, and model training<br>
└── README.md                   # This README file<br>

⚙️ Installation
To set up and run this project locally, follow these steps:

Clone the repository:

git clone https://github.com/AkilaShashimantha/Machine-Learning-Model-Boston-Housing-datasetDeployment-with-Streamlit.git

cd your-project-name

Create a virtual environment (recommended):

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

Install the required libraries:

pip install -r requirements.txt

🚀 How to Run Locally
Follow these two main steps to get the application running on your machine:

Step 1: Train the Model and Generate Artifacts
You need to run the Jupyter Notebook to preprocess the data, train the model, and save the model.pkl, scaler.pkl, and features.pkl files.

Navigate to the notebooks directory:

cd notebooks

Launch Jupyter Notebook:

jupyter notebook

This will open Jupyter in your web browser.

Open model_training.ipynb and run all cells (Kernel > Restart & Run All).

Verify: Ensure that model.pkl, scaler.pkl, and features.pkl are created in your project's root directory (one level up from the notebooks folder) and are not empty (check their file size).

Step 2: Run the Streamlit Application
Once the .pkl files are generated:

Navigate back to the project's root directory:

cd ..

Run the Streamlit app:

streamlit run app.py

This command will launch the Streamlit application and provide a local URL (e.g., http://localhost:8501) which will automatically open in your browser.

☁️ Deployment
This application is designed for deployment on Streamlit Cloud.<br>

Deployed Application URL: https://machine-learning-model-boston-housing-datasetdeployment-with-a.streamlit.app/ <br>

🛠️ Technologies Used
Python

Streamlit: For building the interactive web application.

Pandas: For data manipulation and analysis.

NumPy: For numerical operations.

Scikit-learn: For machine learning model training and preprocessing (Linear Regression, Random Forest Regressor, StandardScaler).

Matplotlib & Seaborn: For data visualization.

Pickle: For serializing and deserializing Python objects (saving/loading models).

💡 Learning Outcomes & Reflection
Completing this assignment provided valuable hands-on experience with:

Building a complete machine learning pipeline from raw data to deployment.

Performing comprehensive Exploratory Data Analysis (EDA) and data preprocessing.

Training and evaluating multiple regression models.

Developing interactive web applications using Streamlit, including handling user inputs and displaying dynamic content.

Understanding the process of deploying ML applications to cloud platforms like Streamlit Cloud.

Practicing version control with Git and GitHub for collaborative development and deployment.

✍️ Author
[Akila Shashimantha]

