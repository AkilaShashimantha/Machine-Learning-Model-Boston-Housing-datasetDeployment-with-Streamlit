import streamlit as st [cite: 106]
import pandas as pd [cite: 107]
import numpy as np [cite: 108]
import pickle [cite: 113]
import matplotlib.pyplot as plt [cite: 109]
import seaborn as sns [cite: 110]
from sklearn.metrics import mean_squared_error, r2_score # For model performance metrics

# Helper function to load model, scaler, and feature names
@st.cache_resource # Cache the model loading for efficiency
def load_model_and_scaler():
    try:
        # Load from the root directory of the deployed app
        with open('model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        with open('features.pkl', 'rb') as features_file:
            features = pickle.load(features_file)
        return model, scaler, features
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e}. Please ensure 'model.pkl', 'scaler.pkl', and 'features.pkl' are in the root directory.")
        st.stop() # Stop the app if crucial files are missing
    except Exception as e:
        st.error(f"Error loading model, scaler, or features: {e}")
        st.stop()

model, scaler, feature_names = load_model_and_scaler()

# Load the dataset (for data exploration and visualization)
@st.cache_data # Cache the data loading for efficiency
def load_data():
    try:
        # Load from the 'data' directory
        data = pd.read_csv('data/boston_housing.csv')
        return data
    except FileNotFoundError:
        st.error("Dataset file (boston_housing.csv) not found in the 'data' directory. Please ensure it's uploaded.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

df = load_data()

# --- Part 2: Streamlit Application Development ---

# Title and Description [cite: 33, 34]
st.title("🏡 Boston House Price Prediction App") [cite: 33, 34]
st.write("This application predicts the median house price in Boston using a trained Machine Learning model. Explore the data, visualize relationships, and get real-time predictions!") [cite: 33, 34]

# Sidebar Navigation [cite: 34]
st.sidebar.title("Navigation") [cite: 34]
page = st.sidebar.radio("Go to", ["Data Exploration", "Visualizations", "Model Prediction", "Model Performance"]) [cite: 34]

if page == "Data Exploration":
    st.header("🔍 Data Exploration Section") [cite: 34]
    st.write("Understand the structure and raw data of the Boston Housing dataset.") [cite: 34]

    # Display dataset overview (shape, columns, data types) [cite: 35]
    st.subheader("Dataset Overview")
    st.write(f"**Shape of the dataset:** {df.shape[0]} rows, {df.shape[1]} columns")
    st.write("**Columns and Data Types:**")
    st.dataframe(df.dtypes.rename('Data Type'))

    # Show sample data [cite: 35]
    st.subheader("Sample Data")
    st.dataframe(df.head())

    # Interactive data filtering options [cite: 37]
    st.subheader("Interactive Data Filtering")
    num_rows = st.slider("Number of rows to display", min_value=5, max_value=len(df), value=10)
    st.dataframe(df.sample(num_rows, random_state=42))

    selected_columns = st.multiselect("Select columns to view", df.columns.tolist(), default=df.columns.tolist()[:5])
    st.dataframe(df[selected_columns].head())

    # Basic descriptive statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe())

elif page == "Visualizations":
    st.header("📊 Visualizations Section") [cite: 38]
    st.write("Explore relationships between features using interactive charts.") [cite: 38]

    # At least 3 different charts/plots [cite: 40]
    st.subheader("1. Distribution of Median House Value (MEDV)")
    fig, ax = plt.subplots()
    sns.histplot(df['MEDV'], kde=True, ax=ax)
    ax.set_title('Distribution of MEDV')
    ax.set_xlabel('MEDV ($1000s)')
    ax.set_ylabel('Frequency')
    st.pyplot(fig) # Display plot

    st.subheader("2. Feature Relationship with MEDV (Scatter Plot)")
    # Interactive visualisations using Streamlit widgets [cite: 41]
    # Ensure selected feature is not 'MEDV' itself
    feature_options = df.columns.drop('MEDV').tolist()
    # Provide a default if the list is not empty
    default_feature = feature_options[0] if feature_options else None
    if 'RM' in feature_options:
        default_feature = 'RM' # Prioritize 'RM' if available as a common feature
    
    feature_x = st.selectbox("Select X-axis feature", feature_options, index=feature_options.index(default_feature) if default_feature else 0)
    
    fig_scatter, ax_scatter = plt.subplots()
    sns.scatterplot(x=df[feature_x], y=df['MEDV'], ax=ax_scatter)
    ax_scatter.set_title(f'{feature_x} vs MEDV')
    ax_scatter.set_xlabel(feature_x)
    ax_scatter.set_ylabel('MEDV ($1000s)')
    st.pyplot(fig_scatter)

    st.subheader("3. Correlation Heatmap")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
    ax_corr.set_title('Correlation Matrix')
    st.pyplot(fig_corr)

    st.subheader("4. Box Plot for CHAS (Charles River dummy variable)")
    # Using 'CHAS' as a categorical example
    if 'CHAS' in df.columns:
        fig_box, ax_box = plt.subplots()
        sns.boxplot(x='CHAS', y='MEDV', data=df, ax=ax_box)
        ax_box.set_title('MEDV by Charles River Proximity (CHAS)')
        ax_box.set_xlabel('Proximity to Charles River (0 = No, 1 = Yes)')
        ax_box.set_ylabel('MEDV ($1000s)')
        st.pyplot(fig_box)
    else:
        st.info("CHAS column not found for Box Plot example.")


elif page == "Model Prediction":
    st.header("🔮 Model Prediction Section") [cite: 42]
    st.write("Enter the features of a house to predict its median value.") [cite: 42]

    # Input widgets for users to enter feature values [cite: 43]
    st.subheader("Enter House Features:")

    inputs = {}
    for feature in feature_names:
        min_val = float(df[feature].min()) if not df[feature].isnull().all() else 0.0
        max_val = float(df[feature].max()) if not df[feature].isnull().all() else 100.0
        mean_val = float(df[feature].mean()) if not df[feature].isnull().all() else 50.0

        # Use appropriate Streamlit widgets [cite: 51]
        # Implement error handling for user inputs [cite: 52]
        try:
            if feature == 'CHAS': # Binary feature
                inputs[feature] = st.selectbox(
                    f"**{feature}** (Charles River dummy variable):",
                    options=[0, 1],
                    format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
                )
            elif feature == 'RAD' or feature == 'ZN': # Often discrete or integer-like
                 inputs[feature] = st.number_input(
                    f"**{feature}** (Range: {int(min_val)} - {int(max_val)}):",
                    min_value=int(min_val),
                    max_value=int(max_val),
                    value=int(mean_val),
                    step=1
                 )
            else: # Other numerical features (float)
                inputs[feature] = st.number_input(
                    f"**{feature}** (Range: {min_val:.2f} - {max_val:.2f}):",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=(max_val - min_val) / 100.0 if (max_val - min_val) > 0 else 0.01,
                    format="%.2f"
                )
        except Exception as e:
            st.warning(f"Could not create input for '{feature}'. Defaulting to 0.0. Error: {e}")
            inputs[feature] = 0.0 # Fallback default

    if st.button("Predict House Price"):
        # ⏳ Include loading states for long operations [cite: 53]
        with st.spinner('Making prediction...'):
            try:
                # Create a DataFrame from inputs ensuring column order matches training
                input_df = pd.DataFrame([inputs])
                input_df = input_df[feature_names] # Reorder columns to match training features

                # Apply the same scaling as used during training
                scaled_input = scaler.transform(input_df)
                prediction = model.predict(scaled_input)[0]

                # Real-time prediction display [cite: 44]
                st.subheader("Prediction Result:")
                st.success(f"The predicted median house value is: **${prediction:,.2f}** (in thousands)")
                st.info("This is an estimated value based on the trained model.")

                # Prediction confidence/probability (if applicable) [cite: 45]
                st.markdown("---")
                st.write("Note: For regression tasks, 'confidence' typically relates to prediction intervals or model error. This model's general performance can be seen in the 'Model Performance' section.")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.info("Please verify your input values and try again. Ensure all fields are valid numbers.")


elif page == "Model Performance":
    st.header("📈 Model Performance Section") [cite: 46]
    st.write("Review the evaluation metrics of the trained model.") [cite: 46]

    # Display model evaluation metrics [cite: 47]
    st.subheader("Model Evaluation Metrics:")

    # Re-split data and make predictions for evaluation consistency
    X_full = df[feature_names] # Use only the features the model was trained on
    y_full = df['MEDV']
    X_train_eval, X_test_eval, y_train_eval, y_test_eval = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
    X_test_scaled_eval = scaler.transform(X_test_eval) # Apply the loaded scaler

    y_pred_eval = model.predict(X_test_scaled_eval)

    mse = mean_squared_error(y_test_eval, y_pred_eval)
    rmse = np.sqrt(mse) # Root Mean Squared Error
    r2 = r2_score(y_test_eval, y_pred_eval)

    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
    st.write(f"**R-squared (R²):** {r2:.2f}")
    st.info("Higher R-squared and lower MSE/RMSE indicate better model performance.")

    # Relevant performance charts [cite: 48]
    st.subheader("Actual vs. Predicted Values Plot")
    fig_actual_pred, ax_actual_pred = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y_test_eval, y=y_pred_eval, ax=ax_actual_pred, alpha=0.6)
    # Add a perfect prediction line
    min_val = min(y_test_eval.min(), y_pred_eval.min())
    max_val = max(y_test_eval.max(), y_pred_eval.max())
    ax_actual_pred.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax_actual_pred.set_xlabel("Actual MEDV")
    ax_actual_pred.set_ylabel("Predicted MEDV")
    ax_actual_pred.set_title("Actual vs. Predicted Median House Values")
    ax_actual_pred.legend()
    st.pyplot(fig_actual_pred)
    st.markdown("A perfect model would have all points lying on the red dashed line.")

    st.subheader("Residuals Plot")
    residuals = y_test_eval - y_pred_eval
    fig_residuals, ax_residuals = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y_pred_eval, y=residuals, ax=ax_residuals, alpha=0.6)
    ax_residuals.axhline(y=0, color='r', linestyle='--', lw=2)
    ax_residuals.set_xlabel("Predicted MEDV")
    ax_residuals.set_ylabel("Residuals (Actual - Predicted)")
    ax_residuals.set_title("Residuals Plot")
    st.pyplot(fig_residuals)
    st.markdown("Ideally, residuals should be randomly scattered around zero, with no clear pattern.")

    # Model comparison results [cite: 49]
    st.subheader("Model Comparison (Context)")
    st.write("During the training phase, multiple algorithms were compared (e.g., Linear Regression, Random Forest Regressor). The currently deployed model was selected based on its superior performance (e.g., lower Mean Squared Error and higher R-squared) on the held-out test dataset.")
    st.write("For a detailed comparison and the training process, please refer to the `model_training.ipynb` Jupyter Notebook in the project's GitHub repository.")

# Add documentation/help text for users [cite: 55]
st.sidebar.markdown("---")
st.sidebar.info(
    "This application is a demonstration for the 'Machine Learning Model Deployment with Streamlit' assignment[cite: 1, 2, 3]. "
    "Developed as part of the course requirements. "
)

# Apply consistent styling and layout [cite: 54]
# Streamlit widgets and layout functions (st.title, st.header, st.subheader, st.write, st.columns)
# inherently help in applying consistent styling.