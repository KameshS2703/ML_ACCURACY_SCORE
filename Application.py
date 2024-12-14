import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, VotingRegressor
from sklearn.svm import SVR, SVC
from xgboost import XGBRegressor, XGBClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import io

# Set page config
st.set_page_config(page_title="Model Trainer & Evaluator", layout="wide", page_icon=":guardsman:", initial_sidebar_state="expanded")

# Header
st.title("Model Training and Evaluation")
st.sidebar.header("Navigation")

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type=["csv"])

# Variables
combined_model = None
X_train, X_test, y_train, y_test = None, None, None, None
task = None
regression_algorithms = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Elastic Net": ElasticNet(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "SVR": SVR(),
    "Gradient Boosting Regressor": GradientBoostingRegressor(),
    "K-Neighbors Regressor": KNeighborsRegressor()
}
classification_algorithms = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest Classifier': RandomForestClassifier(),
    'Gradient Boosting Classifier': GradientBoostingClassifier(),
    'Support Vector Classifier': SVC(probability=True),
    'Decision Tree Classifier': DecisionTreeClassifier(),
    'XGBoost Classifier': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'K-Neighbors Classifier': KNeighborsClassifier(),
    'AdaBoost Classifier': AdaBoostClassifier()
}

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    target_column = st.selectbox("Select Target Column", data.columns)

    if target_column:
        X = data.drop(columns=[target_column])
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
        categorical_features = X.select_dtypes(exclude=['float64', 'int64']).columns

        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

        # Sidebar Task Selection
        task = st.sidebar.radio("Select Task", ["Regression", "Classification"])

        # --- Model Training & Evaluation ---
        if task == "Regression":
            with st.expander("Regression Model Training and Evaluation"):
                st.write("### Regression Metrics")
                results = []

                for name, model in regression_algorithms.items():
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', model)
                    ])

                    try:
                        pipeline.fit(X_train, y_train)
                        y_pred = pipeline.predict(X_test)

                        mae = mean_absolute_error(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)

                        results.append({"Model": name, "MAE": mae, "MSE": mse, "R²": r2})
                    except Exception as e:
                        st.warning(f"Model {name} failed: {e}")

                # Displaying the results in a table format
                results_df = pd.DataFrame(results)
                st.write("### Individual Model Performance")
                st.dataframe(results_df)

                # Combine Selected Models
                # Combine Selected Models
                st.write("### Combine Selected Models")
                selected_models = st.multiselect("Select Models for Combination", list(regression_algorithms.keys()))

                if selected_models and len(selected_models) > 1:
                    estimators = [(name, regression_algorithms[name]) for name in selected_models]
                    combined_model = VotingRegressor(estimators=estimators)
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', combined_model)
                    ])

                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)

                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    st.write("### Combined Model Performance")
                    combined_results = pd.DataFrame({
                        "MAE": [mae], "MSE": [mse], "R²": [r2]
                    })
                    st.dataframe(combined_results)

                # --- Regression Model Metrics Visualization ---
                st.write("### Regression Metrics Visualization")
                plot_type = st.selectbox(
                    "Select Visualization Type",
                    ["Bar Chart", "Line Chart", "Box Plot"]
                )

                mae_values, mse_values, r2_values = [], [], []
                for name, model in regression_algorithms.items():
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', model)
                    ])
                    try:
                        pipeline.fit(X_train, y_train)
                        y_pred = pipeline.predict(X_test)

                        mae = mean_absolute_error(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)

                        mae_values.append(mae)
                        mse_values.append(mse)
                        r2_values.append(r2)
                    except Exception as e:
                        st.warning(f"Model {name} failed: {e}")

                # Create subplots for stacked visualization
                fig, axs = plt.subplots(3, 1, figsize=(10, 15), constrained_layout=True)

                # Bar Chart Visualization
                if plot_type == "Bar Chart":
                    sns.barplot(ax=axs[0], x=mae_values, y=list(regression_algorithms.keys()), palette="Blues_d")
                    axs[0].set_title("Mean Absolute Error (MAE)")
                    axs[0].set_xlabel("MAE")

                    sns.barplot(ax=axs[1], x=mse_values, y=list(regression_algorithms.keys()), palette="Reds_d")
                    axs[1].set_title("Mean Squared Error (MSE)")
                    axs[1].set_xlabel("MSE")

                    sns.barplot(ax=axs[2], x=r2_values, y=list(regression_algorithms.keys()), palette="Greens_d")
                    axs[2].set_title("R² Score")
                    axs[2].set_xlabel("R²")

                # Line Chart Visualization
                elif plot_type == "Line Chart":
                    axs[0].plot(list(regression_algorithms.keys()), mae_values, marker='o', color="blue")
                    axs[0].set_title("Mean Absolute Error (MAE)")
                    axs[0].set_ylabel("MAE")
                    axs[0].set_xticklabels(list(regression_algorithms.keys()), rotation=45)

                    axs[1].plot(list(regression_algorithms.keys()), mse_values, marker='o', color="red")
                    axs[1].set_title("Mean Squared Error (MSE)")
                    axs[1].set_ylabel("MSE")
                    axs[1].set_xticklabels(list(regression_algorithms.keys()), rotation=45)

                    axs[2].plot(list(regression_algorithms.keys()), r2_values, marker='o', color="green")
                    axs[2].set_title("R² Score")
                    axs[2].set_ylabel("R²")
                    axs[2].set_xticklabels(list(regression_algorithms.keys()), rotation=45)

                # Box Plot Visualization
                elif plot_type == "Box Plot":
                    sns.boxplot(ax=axs[0], x=mae_values, color="blue")
                    axs[0].set_title("Mean Absolute Error (MAE)")
                    axs[0].set_xlabel("MAE")

                    sns.boxplot(ax=axs[1], x=mse_values, color="red")
                    axs[1].set_title("Mean Squared Error (MSE)")
                    axs[1].set_xlabel("MSE")

                    sns.boxplot(ax=axs[2], x=r2_values, color="green")
                    axs[2].set_title("R² Score")
                    axs[2].set_xlabel("R²")

                st.pyplot(fig)
        elif task == "Classification":
            with st.expander("Classification Model Training and Evaluation"):
                st.write("### Classification Metrics")
                results = []

                for name, model in classification_algorithms.items():
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', model)
                    ])

                    try:
                        pipeline.fit(X_train, y_train)
                        y_pred = pipeline.predict(X_test)

                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, average='weighted')
                        recall = recall_score(y_test, y_pred, average='weighted')
                        f1 = f1_score(y_test, y_pred, average='weighted')

                        results.append({
                            "Model": name, "Accuracy": accuracy,
                            "Precision": precision, "Recall": recall, "F1": f1
                        })
                    except Exception as e:
                        st.warning(f"Model {name} failed: {e}")

                # Displaying the results in a table format
                results_df = pd.DataFrame(results)
                st.write("### Individual Model Performance")
                st.dataframe(results_df)

                # Combine Selected Models
                st.write("### Combine Selected Models")
                selected_models = st.multiselect("Select Models for Combination", list(classification_algorithms.keys()))

                if selected_models and len(selected_models) > 1:
                    estimators = [(name, classification_algorithms[name]) for name in selected_models]
                    combined_model = VotingClassifier(estimators=estimators)
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', combined_model)
                    ])

                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)

                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')

                    st.write("### Combined Model Performance")
                    combined_results = pd.DataFrame({
                        "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1 Score": [f1]
                    })
                    st.dataframe(combined_results)

                # --- Classification Model Metrics Visualization ---
                st.write("### Classification Metrics Visualization")
                plot_type = st.selectbox(
                    "Select Visualization Type",
                    ["Bar Chart", "Line Chart", "Box Plot"]
                )

                accuracy_values, precision_values, recall_values, f1_values = [], [], [], []
                for name, model in classification_algorithms.items():
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', model)
                    ])
                    try:
                        pipeline.fit(X_train, y_train)
                        y_pred = pipeline.predict(X_test)

                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, average='weighted')
                        recall = recall_score(y_test, y_pred, average='weighted')
                        f1 = f1_score(y_test, y_pred, average='weighted')

                        accuracy_values.append(accuracy)
                        precision_values.append(precision)
                        recall_values.append(recall)
                        f1_values.append(f1)
                    except Exception as e:
                        st.warning(f"Model {name} failed: {e}")

                # Create subplots for stacked visualization
                fig, axs = plt.subplots(4, 1, figsize=(10, 20), constrained_layout=True)

                # Bar Chart Visualization
                if plot_type == "Bar Chart":
                    sns.barplot(ax=axs[0], x=accuracy_values, y=list(classification_algorithms.keys()), palette="Blues_d")
                    axs[0].set_title("Accuracy")
                    axs[0].set_xlabel("Accuracy")

                    sns.barplot(ax=axs[1], x=precision_values, y=list(classification_algorithms.keys()), palette="Reds_d")
                    axs[1].set_title("Precision")
                    axs[1].set_xlabel("Precision")

                    sns.barplot(ax=axs[2], x=recall_values, y=list(classification_algorithms.keys()), palette="Greens_d")
                    axs[2].set_title("Recall")
                    axs[2].set_xlabel("Recall")

                    sns.barplot(ax=axs[3], x=f1_values, y=list(classification_algorithms.keys()), palette="Purples_d")
                    axs[3].set_title("F1 Score")
                    axs[3].set_xlabel("F1 Score")

                # Line Chart Visualization
                elif plot_type == "Line Chart":
                    axs[0].plot(list(classification_algorithms.keys()), accuracy_values, marker='o', color="blue")
                    axs[0].set_title("Accuracy")
                    axs[0].set_ylabel("Accuracy")
                    axs[0].set_xticklabels(list(classification_algorithms.keys()), rotation=45)

                    axs[1].plot(list(classification_algorithms.keys()), precision_values, marker='o', color="red")
                    axs[1].set_title("Precision")
                    axs[1].set_ylabel("Precision")
                    axs[1].set_xticklabels(list(classification_algorithms.keys()), rotation=45)

                    axs[2].plot(list(classification_algorithms.keys()), recall_values, marker='o', color="green")
                    axs[2].set_title("Recall")
                    axs[2].set_ylabel("Recall")
                    axs[2].set_xticklabels(list(classification_algorithms.keys()), rotation=45)

                    axs[3].plot(list(classification_algorithms.keys()), f1_values, marker='o', color="purple")
                    axs[3].set_title("F1 Score")
                    axs[3].set_ylabel("F1 Score")
                    axs[3].set_xticklabels(list(classification_algorithms.keys()), rotation=45)

                # Box Plot Visualization
                elif plot_type == "Box Plot":
                    sns.boxplot(ax=axs[0], x=accuracy_values, color="blue")
                    axs[0].set_title("Accuracy")
                    axs[0].set_xlabel("Accuracy")

                    sns.boxplot(ax=axs[1], x=precision_values, color="red")
                    axs[1].set_title("Precision")
                    axs[1].set_xlabel("Precision")

                    sns.boxplot(ax=axs[2], x=recall_values, color="green")
                    axs[2].set_title("Recall")
                    axs[2].set_xlabel("Recall")

                    sns.boxplot(ax=axs[3], x=f1_values, color="purple")
                    axs[3].set_title("F1 Score")
                    axs[3].set_xlabel("F1 Score")

                st.pyplot(fig)


        # --- Predict on New Data ---
        with st.expander("Predict on New Data"):
            st.write("### Predict on New Data")

            # Filter models based on selected task
            available_models = regression_algorithms if task == "Regression" else classification_algorithms
            if combined_model:
                available_models["Combined Model"] = combined_model

            selected_model_name = st.selectbox("Select Model or Combined Model for Prediction", list(available_models.keys()))

            # Layout cards horizontally
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Model**: {selected_model_name}")
            with col2:
                st.write("**New Data Prediction Form**")
                new_data = {}
                for feature in X.columns:
                    if feature != target_column:
                        if feature in numeric_features:
                            new_data[feature] = st.number_input(f"Enter {feature} value")
                        elif feature in categorical_features:
                            categories = data[feature].unique().tolist()
                            new_data[feature] = st.selectbox(f"Select {feature}", categories)

                if st.button("Make Prediction"):
                    if selected_model_name == "Combined Model" and combined_model:
                        model = combined_model
                    else:
                        model = available_models[selected_model_name]

                    prediction_pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', model)
                    ])

                    prediction = prediction_pipeline.predict(pd.DataFrame([new_data]))

                    st.write(f"Prediction Result: {prediction[0]}")

                    # Saving the model as a pickle file
                    pickle_filename = f"{selected_model_name}_model.pkl"
                    with open(pickle_filename, 'wb') as file:
                        pickle.dump(model, file)

                    # Download Button for Pickle File
                    with open(pickle_filename, "rb") as file:
                        st.download_button(label="Download Model as Pickle", data=file, file_name=pickle_filename, mime="application/octet-stream")
