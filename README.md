# HR Analytics & Employee Attrition Analysis Dashboard

## 📌 Project Overview

The **HR Analytics & Employee Attrition Analysis Dashboard** is a data analytics project designed to explore employee workforce data, identify attrition patterns, and generate meaningful insights that can support Human Resource (HR) decision-making.

Employee attrition can affect workforce stability, recruitment costs, and organizational performance. This project uses data cleaning, exploratory data analysis, interactive Power BI visualizations, and machine learning techniques to understand employee attrition.

The project combines **Power BI, Python, Excel, and Machine Learning** to transform raw employee data into actionable insights.

## 🎯 Project Objectives

* Analyze employee attrition patterns across departments and job roles.
* Calculate important workforce KPIs.
* Identify employee groups with different attrition patterns.
* Understand the relationship between overtime and attrition.
* Explore salary and employee tenure trends.
* Build an interactive HR analytics dashboard.
* Experiment with machine learning models to predict employee attrition.
* Present insights through clear visualizations and analytical results.

## 🛠️ Tools and Technologies

| Tool / Technology | Purpose                                     |
| ----------------- | ------------------------------------------- |
| Power BI          | Interactive dashboard and KPI visualization |
| Python            | Data analysis and machine learning          |
| Pandas            | Data cleaning and manipulation              |
| NumPy             | Numerical operations                        |
| Matplotlib        | Data visualization                          |
| Scikit-learn      | Machine learning and model evaluation       |
| Excel             | Dataset inspection and cleaned data storage |
| GitHub            | Project documentation and version control   |

## 📂 Dataset Description

This project uses the IBM HR Analytics Employee Attrition dataset, which contains employee-related information such as demographics, department, job role, monthly income, job satisfaction, overtime, years at company, and attrition status.

The dataset contains **1,470 employee records and 39 columns** in the cleaned project file.

The analysis focuses on understanding workforce characteristics and examining how employee attributes relate to attrition.

## 🧹 Data Cleaning and Preparation

The dataset was prepared for analysis through the following steps:

* Inspected the dataset structure and available columns.
* Checked for missing values and duplicate records.
* Prepared a cleaned dataset for dashboard development.
* Reviewed relevant employee and job-related attributes.
* Created additional categorical features to support analysis.

### Feature Engineering

Two additional features were created:

* **AgeGroup:** Groups employees into age categories.
* **TenureGroup:** Groups employees according to their years at the company.

These features make it easier to compare employee groups in the dashboard.

## 📊 Power BI Dashboard

The interactive Power BI dashboard provides a consolidated view of employee data and attrition patterns.

### Key Performance Indicators (KPIs)

* **Total Employees:** 1,470
* **Attrition Rate:** 16.12%
* **Average Monthly Income:** Approximately 6,502
* **Average Years at Company:** Approximately 7.01

### Dashboard Visualizations

1. **Attrition Distribution:** Shows the distribution of employees by attrition status.
2. **Employees by Department:** Compares workforce distribution across departments.
3. **Job Role Analysis:** Displays employees across different job roles.
4. **Attrition by Overtime:** Examines attrition patterns among employees who work overtime and those who do not.
5. **Employees by Tenure Group:** Shows workforce distribution by years at the company.
6. **Average Monthly Income by Department:** Compares average income across departments.

### Interactive Filters

The dashboard includes slicers for:

* Department
* Job Role
* Gender
* Overtime

These filters allow users to explore different segments of the workforce.

## 🤖 Machine Learning Analysis

Two classification algorithms were evaluated to explore employee attrition prediction.

### Models Used

**1. Logistic Regression**

A classification model used to estimate the relationship between employee attributes and attrition outcomes.

**2. Random Forest Classifier**

An ensemble learning model that combines multiple decision trees to classify employee attrition outcomes.

### Model Evaluation

The models were evaluated using an 80/20 stratified train-test split with a random state of 42.

| Metric    | Logistic Regression | Random Forest |
| --------- | ------------------: | ------------: |
| Accuracy  |               78.9% |         85.7% |
| Precision |               40.7% |         63.2% |
| Recall    |               70.2% |         25.5% |
| F1-Score  |               51.6% |         36.4% |
| ROC-AUC   |               83.8% |         79.9% |

### Interpretation

Logistic Regression achieved a higher recall and ROC-AUC in this evaluation, while Random Forest achieved higher accuracy and precision.

Accuracy alone does not determine the best attrition model. Recall, precision, F1-score, class imbalance, and the business context should also be considered.

These results are specific to the current experiment and test split; further validation and model tuning would be needed before considering practical use.

## 💡 Key Analytical Insights

The dashboard and model evaluation support the following areas of investigation:

* Comparing attrition across departments and job roles.
* Examining differences in attrition between overtime groups.
* Understanding workforce distribution across tenure categories.
* Comparing average monthly income across departments.
* Evaluating how classification models perform when identifying employees who leave.

The dashboard enables users to explore these patterns interactively rather than relying only on static summaries.

File names may differ slightly depending on the files uploaded to the repository.

## ▶️ How to Run the Python Analysis

### 1. Clone the Repository

```bash
git clone https://github.com/nileshmahale103/HR-Analytics-Employee-Attrition.git
```

### 2. Navigate to the Project Folder

```bash
cd HR-Analytics-Employee-Attrition
```

### 3. Install Dependencies

```bash
pip install pandas numpy matplotlib scikit-learn openpyxl
```

### 4. Run the Python Script

```bash
python hr_analytics_task3.py
```

Make sure the cleaned dataset is available at the file path expected by the Python script. Update the path if necessary.

## 📈 Business Applications

This project demonstrates how data analytics can support HR teams in:

* Monitoring workforce attrition.
* Comparing employee groups.
* Exploring potential attrition-related patterns.
* Supporting workforce planning discussions.
* Communicating HR metrics through dashboards.
* Evaluating predictive analytics approaches.

The results are exploratory and should be interpreted alongside organizational context and additional evidence.

## ⚠️ Limitations and Ethical Considerations

* The dataset is a sample dataset and may not represent every organization.
* Correlation or association does not establish causation.
* Model results depend on the available features, data preparation, and evaluation method.
* Predictive models may contain bias and can produce incorrect predictions.
* This project is intended for learning and portfolio demonstration.
* Predictions should not be used as the sole basis for hiring, promotion, termination, or other employment decisions.

## 🚀 Future Improvements

Potential future enhancements include:

* Additional exploratory data analysis.
* Hyperparameter tuning and cross-validation.
* More detailed model comparison and explainability.
* Improved dashboard navigation and drill-through analysis.
* Automated reporting and periodic KPI tracking.
* Further validation on appropriate additional datasets.

## 👨‍💻 Author

**Nilesh Vasant Mahale**

Computer Engineering Student | Data Analytics

* **GitHub:** [nileshmahale103](https://github.com/nileshmahale103)

## ⭐ Acknowledgements

Dataset: IBM HR Analytics Employee Attrition dataset.

This project was developed for educational purposes to practice data analytics, dashboard development, and machine learning.

---

If you find this project useful, feel free to explore the repository and its files.

