# **Machine Learning Model Trainer and Evaluator**  

An interactive, user-friendly application built with **Streamlit** that enables users to upload datasets, train machine learning models, evaluate their performance, and make predictions—all with minimal coding! This project is designed to simplify the machine learning workflow, making it accessible for beginners and useful for experienced practitioners.

---

## **Features**  

### 🔍 **Dataset Integration**  
- Upload your dataset in CSV format.  
- Preview the dataset directly within the app.  

### 🛠️ **Data Preprocessing**  
- Handles missing values automatically.  
- Scales numeric features and encodes categorical variables.  

### 🤖 **Model Training and Evaluation**  
#### Regression Algorithms:  
- Linear Regression  
- Ridge Regression  
- Lasso Regression  
- Elastic Net  
- Decision Tree Regressor  
- Random Forest Regressor  
- Gradient Boosting Regressor  
- XGBoost Regressor  
- K-Neighbors Regressor  

#### Classification Algorithms:  
- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  
- Gradient Boosting Classifier  
- XGBoost Classifier  
- Support Vector Classifier (SVC)  
- K-Neighbors Classifier  
- AdaBoost Classifier  

### 📊 **Performance Metrics**  
- **Regression**:  
  - Mean Absolute Error (MAE)  
  - Mean Squared Error (MSE)  
  - R² Score  
- **Classification**:  
  - Accuracy  
  - Precision  
  - Recall  
  - F1 Score  

### ⚡ **Ensemble Models**  
- Combine multiple models using Voting Regressor or Voting Classifier for enhanced accuracy and performance.  

### 🌟 **Visualization**  
- Compare model performance through Bar Charts, Line Plots, and Box Plots for deeper insights.  

### 🎯 **Real-Time Predictions**  
- Make predictions using new data by filling out an intuitive form.  
- Supports both individual models and ensemble models.  

### 💾 **Model Export**  
- Save trained models as `.pkl` files.  
- Download models directly from the app.  

---

## **Getting Started**  

### **Prerequisites**  
Ensure you have the following installed:  
- Python 3.7 or higher  
- pip  

### **Installation**  
1. Clone this repository:  
   ```bash
   git clone https://github.com/your-username/ml-model-trainer.git
   cd ml-model-trainer
   ```  
2. Install the required dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  

### **Run the App**  
Start the Streamlit app with:  
```bash
streamlit run app.py
```  
The app will be accessible at `http://localhost:8501` in your browser.  

---

## **How to Use**  

1. **Upload Dataset**:  
   - Navigate to the sidebar and upload your CSV file.  

2. **Select Target Column**:  
   - Choose the column you want to predict.  

3. **Choose Task**:  
   - Select "Regression" or "Classification" based on your problem.  

4. **Train Models**:  
   - Evaluate individual models and view their performance metrics.  

5. **Combine Models**:  
   - Create ensemble models for better accuracy.  

6. **Visualize Metrics**:  
   - Use interactive charts to understand model performance.  

7. **Make Predictions**:  
   - Input new data and predict results in real time.  

8. **Download Models**:  
   - Export trained models as `.pkl` files for future use.  

---

## **Tech Stack**  

- **Frontend**: Streamlit  
- **Backend**: Scikit-learn, XGBoost  
- **Visualization**: Seaborn, Matplotlib  
- **Data Manipulation**: Pandas  
- **Model Export**: Pickle  

---

## **Project Structure**  

```plaintext
📂 ml-model-trainer  
├── app.py                     # Main application script  
├── requirements.txt           # Dependencies  
├── README.md                  # Project documentation  
└── sample_data.csv            # Sample dataset (optional)  
```  

---

## **Future Enhancements**  
- Add support for unsupervised learning (e.g., clustering).  
- Integrate hyperparameter tuning options.  
- Enable deployment to the cloud for wider accessibility.  

---

## **Contributing**  
Contributions are welcome! Feel free to submit a pull request or raise an issue.  

---

## **License**  
This project is licensed under the MIT License. See the `LICENSE` file for more details.  

---

## **Acknowledgments**  
Thanks to the open-source community for tools like Streamlit and Scikit-learn, which make projects like this possible.  

---

