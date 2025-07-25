import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split # Needed for model performance section (to re-split data)

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
        st.error(f"Required file not found: {e}. Please ensure 'model.pkl', 'scaler.pkl', and 'features.pkl' are in the root directory and correctly named.")
        st.stop() # Stop the app if crucial files are missing
    except Exception as e:
        st.error(f"Error loading model, scaler, or features: {e}. Ensure they were saved correctly and are not empty/corrupted.")
        st.stop()

model, scaler, feature_names = load_model_and_scaler()

# Load the dataset (for data exploration and visualization)
@st.cache_data # Cache the data loading for efficiency
def load_data():
    try:
        # Load from the 'data' directory. Ensure correct filename: 'BostonHousing.csv'
        data = pd.read_csv('data/BostonHousing.csv')
        return data
    except FileNotFoundError:
        st.error("Dataset file (BostonHousing.csv) not found in the 'data' directory. Please ensure it's uploaded with the correct name.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading dataset: {e}. Please check the file's integrity and path.")
        st.stop()

df = load_data()

# Set the name of your target column consistently
target_column_name = 'medv' # Make sure this matches the name in your CSV and notebook

# --- Part 2: Streamlit Application Development ---

# Title and Description
st.title("🏡 Boston House Price Prediction App")
st.write("This application predicts the median house price in Boston using a trained Machine Learning model. Explore the data, visualize relationships, and get real-time predictions!")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Exploration", "Visualizations", "Model Prediction", "Model Performance"])

if page == "Data Exploration":
    st.header("🔍 Data Exploration Section")
    st.write("Understand the structure and raw data of the Boston Housing dataset.")

    # Display dataset overview (shape, columns, data types)
    st.subheader("Dataset Overview")
    st.write(f"**Shape of the dataset:** {df.shape[0]} rows, {df.shape[1]} columns")
    st.write("**Columns and Data Types:**")
    st.dataframe(df.dtypes.rename('Data Type'))

    # Show sample data
    st.subheader("Sample Data")
    st.dataframe(df.head())

    # Interactive data filtering options
    st.subheader("Interactive Data Filtering")
    num_rows = st.slider("Number of rows to display", min_value=5, max_value=len(df), value=10)
    st.dataframe(df.sample(num_rows, random_state=42))

    selected_columns = st.multiselect("Select columns to view", df.columns.tolist(), default=df.columns.tolist()[:5])
    st.dataframe(df[selected_columns].head())

    # Basic descriptive statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe())

elif page == "Visualizations":
    st.header("📊 Visualizations Section")
    st.write("Explore relationships between features using interactive charts.")

    # At least 3 different charts/plots
    st.subheader(f"1. Distribution of Median House Value ({target_column_name})")
    fig, ax = plt.subplots()
    sns.histplot(df[target_column_name], kde=True, ax=ax)
    ax.set_title(f'Distribution of {target_column_name}')
    ax.set_xlabel(f'{target_column_name} ($1000s)')
    ax.set_ylabel('Frequency')
    st.pyplot(fig) # Display plot

    st.subheader(f"2. Feature Relationship with {target_column_name} (Scatter Plot)")
    # Interactive visualisations using Streamlit widgets
    # Ensure selected feature is not the target column itself
    feature_options = [col for col in df.columns if col != target_column_name]
    default_feature = 'RM' if 'RM' in feature_options else (feature_options[0] if feature_options else None) # Prioritize 'RM' or first feature
    
    if feature_options:
        feature_x = st.selectbox("Select X-axis feature", feature_options, index=feature_options.index(default_feature))
        fig_scatter, ax_scatter = plt.subplots()
        sns.scatterplot(x=df[feature_x], y=df[target_column_name], ax=ax_scatter)
        ax_scatter.set_title(f'{feature_x} vs {target_column_name}')
        ax_scatter.set_xlabel(feature_x)
        ax_scatter.set_ylabel(f'{target_column_name} ($1000s)')
        st.pyplot(fig_scatter)
    else:
        st.info("No features available for scatter plot after excluding target column.")

    st.subheader("3. Correlation Heatmap")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
    ax_corr.set_title('Correlation Matrix')
    st.pyplot(fig_corr)

    st.subheader("4. Box Plot for CHAS (Charles River dummy variable)")
    # Using 'CHAS' as a categorical example
    if 'CHAS' in df.columns:
        fig_box, ax_box = plt.subplots()
        sns.boxplot(x='CHAS', y=df[target_column_name], data=df, ax=ax_box)
        ax_box.set_title(f'{target_column_name} by Charles River Proximity (CHAS)')
        ax_box.set_xlabel('Proximity to Charles River (0 = No, 1 = Yes)')
        ax_box.set_ylabel(f'{target_column_name} ($1000s)')
        st.pyplot(fig_box)
    else:
        st.info("CHAS column not found for Box Plot example.")


elif page == "Model Prediction":
    st.header("🔮 Model Prediction Section")
    st.write("Enter the features of a house to predict its median value.")

    # Input widgets for users to enter feature values
    st.subheader("Enter House Features:")

    inputs = {}
    for feature in feature_names: # Iterate through feature_names loaded from features.pkl
        # Ensure feature exists in df and handle potential NaNs in min/max/mean calculation
        if feature in df.columns and not df.loc[:,feature].isnull().all():
            min_val = float(df.loc[:,feature].min())
            max_val = float(df.loc[:,feature].max())
            mean_val = float(df.loc[:,feature].mean())
        else: # Fallback if feature not in df or all NaNs
            min_val = 0.0
            max_val = 100.0
            mean_val = 50.0
            st.warning(f"Feature '{feature}' not found in dataset or contains all NaNs. Using default ranges.")

        # Use appropriate Streamlit widgets
        # Implement error handling for user inputs
        try:
            if feature == 'CHAS': # Binary feature
                inputs[feature] = st.selectbox(
                    f"**{feature}** (Charles River dummy variable):",
                    options=[0, 1],
                    format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
                )
            elif feature == 'RAD' or feature == 'ZN' or df[feature].dtype == 'int64': # Often discrete or integer-like
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
            st.warning(f"Could not create input widget for '{feature}'. Defaulting to 0.0. Error: {e}")
            inputs[feature] = 0.0 # Fallback default

    if st.button("Predict House Price"):
        # Include loading states for long operations
        with st.spinner('Making prediction...'):
            try:
                # Create a DataFrame from inputs ensuring column order matches training
                input_df = pd.DataFrame([inputs])
                # Ensure all features expected by the model are present, even if their input was skipped (will use default 0.0)
                input_df = input_df.reindex(columns=feature_names, fill_value=0.0)

                # Apply the same scaling as used during training
                scaled_input = scaler.transform(input_df)
                prediction = model.predict(scaled_input)[0]

                # Real-time prediction display
                st.subheader("Prediction Result:")
                st.success(f"The predicted median house value is: **${prediction:,.2f}** (in thousands)")
                st.info("This is an estimated value based on the trained model.")

                # Prediction confidence/probability (if applicable)
                st.markdown("---")
                st.write("Note: For regression tasks, 'confidence' typically relates to prediction intervals or model error. This model's general performance can be seen in the 'Model Performance' section.")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.info("Please verify your input values and try again. Ensure all fields are valid numbers and the model/scaler loaded correctly.")


elif page == "Model Performance":
    st.header("📈 Model Performance Section")
    st.write("Review the evaluation metrics of the trained model.")

    # Display model evaluation metrics
    st.subheader("Model Evaluation Metrics:")

    # Re-split data and make predictions for evaluation consistency
    # Use only the features the model was trained on
    # Ensure df has all feature_names and target_column_name before splitting
    if all(f in df.columns for f in feature_names) and target_column_name in df.columns:
        X_full = df[feature_names] 
        y_full = df[target_column_name]
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

        # Relevant performance charts
        st.subheader("Actual vs. Predicted Values Plot")
        fig_actual_pred, ax_actual_pred = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=y_test_eval, y=y_pred_eval, ax=ax_actual_pred, alpha=0.6)
        # Add a perfect prediction line
        min_val = min(y_test_eval.min(), y_pred_eval.min())
        max_val = max(y_test_eval.max(), y_pred_eval.max())
        ax_actual_pred.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax_actual_pred.set_xlabel(f"Actual {target_column_name}")
        ax_actual_pred.set_ylabel(f"Predicted {target_column_name}")
        ax_actual_pred.set_title(f"Actual vs. Predicted Median House Values ({target_column_name})")
        ax_actual_pred.legend()
        st.pyplot(fig_actual_pred)
        st.markdown("A perfect model would have all points lying on the red dashed line.")

        st.subheader("Residuals Plot")
        residuals = y_test_eval - y_pred_eval
        fig_residuals, ax_residuals = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=y_pred_eval, y=residuals, ax=ax_residuals, alpha=0.6)
        ax_residuals.axhline(y=0, color='r', linestyle='--', lw=2)
        ax_residuals.set_xlabel(f"Predicted {target_column_name}")
        ax_residuals.set_ylabel("Residuals (Actual - Predicted)")
        ax_residuals.set_title("Residuals Plot")
        st.pyplot(fig_residuals)
        st.markdown("Ideally, residuals should be randomly scattered around zero, with no clear pattern.")

        # Model comparison results
        st.subheader("Model Comparison (Context)")
        st.write("During the training phase, multiple algorithms were compared (e.g., Linear Regression, Random Forest Regressor). The currently deployed model was selected based on its superior performance (e.g., lower Mean Squared Error and higher R-squared) on the held-out test dataset.")
        st.write("For a detailed comparison and the training process, please refer to the `model_training.ipynb` Jupyter Notebook in the project's GitHub repository.")
    else:
        st.warning("Could not perform model performance evaluation. Ensure the dataset contains all expected features and the target column.")

# Add documentation/help text for users
st.sidebar.markdown("---")
st.sidebar.info(
    "This application is a demonstration for the 'Machine Learning Model Deployment with Streamlit' assignment. "
    "Developed as part of the course requirements. "
)

# Apply consistent styling and layout
# Streamlit widgets and layout functions (st.title, st.header, st.subheader, st.write, st.columns)
# inherently help in applying consistent styling.