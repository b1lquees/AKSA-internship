# AI Hallucination Cases Analysis
I used the Charlotin AI Hallucination Cases dataset to perform data loading, cleaning, manipulation, visualization, feature engineering, and machine learning.

## What I Did
- Loaded the AI Hallucination Cases dataset using Pandas.
- Explored the dataset, missing values, summary statistics.
- Cleaned the data by normalizing text casing, parsing dates, filling missing values, and simplifying multi-valued columns.
- Basic data analysis using NumPy and Pandas.
- Graphs were created using Matplotlib, Seaborn, and Plotly to visualize the data.
- Feature Engineering was applied by extracting counts from free-text hallucination data and converting categorical data into numerical values.
- Split dataset into training and testing sets.
- Scaled features using StandardScaler.
- Trained a Logistic Regression model and a Random Forest model.
- Evaluated the models using Accuracy, Precision, Recall, F1-score, and Confusion Matrix.

## Libraries Used
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn

## Machine Learning Models
- Logistic Regression
- Random Forest

## Dataset
Dataset: Charlotin AI Hallucination Cases

Features include:
- Date, Court, State(s)
- Legal Field Primary
- AI Tool
- Party(ies)
- Outcome
- Hallucination Items
- Alleged
- Monetary Penalty

Target:
- Professional Sanction

## Results
Model Accuracy:
- Logistic Regression: **90.8%**
- Random Forest: **89.9%**

## Installation
Clone the repository
```bash
git clone <repository_link>
```
Install dependencies
```bash
pip install numpy pandas matplotlib seaborn plotly scikit-learn
```
Run the project
```bash
cd src
python AI_Hallucination_Cases_ML_Task.py
```

## Author
Bilquees Ashfaq Jumani

