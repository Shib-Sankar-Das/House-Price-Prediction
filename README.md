# House Price Prediction - Machine Learning Project

## 🏠 Project Overview

This comprehensive machine learning project implements multiple algorithms to predict house prices using the Ames Housing Dataset. The project includes both regression and classification models, complete with exploratory data analysis, model training, evaluation, and deployment through a Streamlit web application.

## 📊 Dataset

- **Source**: Ames Housing Dataset
- **Training samples**: 1,460 houses
- **Test samples**: 1,459 houses
- **Features**: 79 features describing various aspects of residential properties
- **Target**: SalePrice (house sale price in dollars)

## 🤖 Implemented Algorithms

### Regression Models
1. **Simple Linear Regression** - Single feature prediction
2. **Multiple Linear Regression** - Multiple features prediction
3. **Polynomial Regression** - Non-linear relationships
4. **Ridge Regression** - L2 regularization
5. **Lasso Regression** - L1 regularization (feature selection)
6. **ElasticNet Regression** - Combined L1 and L2 regularization

### Classification Models
1. **Logistic Regression** - Linear classification
2. **Naive Bayes** - Probabilistic classifier
3. **k-Nearest Neighbors (k-NN)** - Instance-based learning
4. **Decision Trees** - Rule-based classification
5. **Random Forest** - Ensemble of decision trees
6. **Support Vector Machines (SVM)** - Margin-based classification

## 📁 Project Structure

```
House Price Prediction/
├── Data/
│   ├── train.csv                    # Training dataset
│   ├── test.csv                     # Test dataset
│   ├── data_description.txt         # Feature descriptions
│   └── sample_submission.csv        # Submission format
├── models/                          # Saved model files (generated after training)
│   ├── best_regression_model.pkl
│   ├── best_classification_model.pkl
│   ├── label_encoders.pkl
│   ├── feature_columns.pkl
│   ├── scaler.pkl
│   └── model_metadata.pkl
├── house_price_prediction_ml.ipynb  # Main analysis notebook
├── house_price_app.py              # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── submission.csv                  # Predictions (generated after training)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project**
   ```bash
   cd "Week 5 - House Price Prediction"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify data files**
   Ensure all data files are in the `Data/` directory:
   - `train.csv`
   - `test.csv`
   - `data_description.txt`
   - `sample_submission.csv`

## 📓 Running the Jupyter Notebook

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Open the analysis notebook**
   - Navigate to `house_price_prediction_ml.ipynb`
   - Run all cells to perform the complete analysis

3. **What the notebook does:**
   - Loads and explores the dataset
   - Performs comprehensive EDA with visualizations
   - Handles missing values and feature engineering
   - Trains all 12 machine learning models
   - Evaluates and compares model performance
   - Saves the best models as PKL files
   - Generates predictions for the test set

## 🌐 Running the Streamlit App

1. **After running the notebook** (to generate model files):
   ```bash
   streamlit run house_price_app.py
   ```

2. **Access the web application**
   - Open your browser and go to: `http://localhost:8501`

3. **App features:**
   - **🏠 Home**: Project overview and best model performance
   - **📊 Data Explorer**: Interactive data visualization and analysis
   - **🔮 Price Prediction**: Real-time house price prediction
   - **📈 Model Performance**: Model comparison and metrics
   - **📋 About**: Detailed project information

## 🔄 Workflow

### Step 1: Data Analysis (Jupyter Notebook)
1. Run the notebook to perform EDA
2. Train and evaluate all models
3. Generate model files and predictions

### Step 2: Web Application (Streamlit)
1. Use the saved models for real-time predictions
2. Explore data interactively
3. Compare model performances

## 📈 Model Performance

The project evaluates models using multiple metrics:

### Regression Models
- **R² Score**: Proportion of variance explained
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error

### Classification Models
- **Accuracy**: Correct predictions percentage
- **Classification Report**: Precision, recall, F1-score

## 🎯 Key Features

### Exploratory Data Analysis
- Missing value analysis and visualization
- Distribution analysis of target variable
- Correlation heatmaps
- Outlier detection
- Feature importance analysis

### Data Preprocessing
- Automated missing value imputation
- Categorical variable encoding
- Feature scaling and normalization
- Feature engineering (creating new features)

### Model Training & Evaluation
- Cross-validation for robust evaluation
- Hyperparameter considerations
- Model comparison and selection
- Performance visualization

### Web Application
- Interactive prediction interface
- Data exploration dashboards
- Model performance comparison
- Responsive design with custom styling

## 🛠️ Technology Stack

- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib & Seaborn**: Static visualizations
- **Plotly**: Interactive visualizations
- **Streamlit**: Web application framework
- **Joblib**: Model persistence
- **Jupyter**: Interactive development environment

## 📊 Expected Results

After running the complete pipeline, you should see:

1. **Model Performance Comparison**: Clear ranking of all 12 algorithms
2. **Best Model Selection**: Automatic identification of top performers
3. **Feature Importance**: Understanding which features matter most
4. **Predictions**: Price predictions for the test dataset
5. **Interactive App**: Fully functional web application

## 🔧 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Install missing packages using pip
2. **File not found errors**: Ensure data files are in the correct directory
3. **Memory errors**: Use a system with sufficient RAM (8GB+ recommended)
4. **Model files not found**: Run the Jupyter notebook first to generate model files

### Performance Tips

- The notebook may take 10-30 minutes to complete depending on your system
- For faster execution, consider reducing the number of models or using smaller datasets
- Ensure sufficient memory is available for large dataset operations

## 📚 Learning Outcomes

This project demonstrates:

- Complete machine learning pipeline implementation
- Multiple algorithm comparison and evaluation
- Feature engineering and selection techniques
- Model deployment and web application development
- Best practices in data science project organization
- Interactive data visualization techniques

## 🚀 Future Enhancements

- Advanced feature engineering techniques
- Ensemble methods and model stacking
- Hyperparameter optimization with Grid/Random Search
- Real-time data integration
- Advanced visualization dashboards
- Model interpretability with SHAP values

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Ensure all dependencies are installed correctly
3. Verify data files are in the correct locations
4. Review the notebook output for error messages

## 📄 License

This project is for educational purposes as part of the CSI Internship Certification Program.

---

**Happy Predicting! 🏠💰**
