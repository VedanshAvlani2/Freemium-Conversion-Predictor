# Freemium Conversion Prediction Simulator

## Overview
This project simulates and models user conversion from a freemium to paid product model using synthetic behavioral data. It applies logistic regression to predict user conversions based on key engagement metrics and visualizes the insights with a variety of charts.

## Objective
- Predict whether a user will convert to a paid product using logistic regression.
- Identify key features impacting conversion like session time, pricing clicks, and engagement.
- Visualize patterns and evaluate model performance through metrics and ROC curves.

## Dataset & Features
A synthetic dataset was generated with the following features:
- **Session_Time**: Time spent on platform per session
- **Feature_Usage_Count**: Number of features used in a session
- **Support_Tickets**: Number of support tickets raised
- **Days_Active**: Number of days active since signup
- **Clicks_on_Pricing**: Whether user clicked pricing page
- **Referral_Flag**: Whether user came via referral
- **Converted**: Target variable (0 = not converted, 1 = converted)

## Technologies Used
- Python
- NumPy, pandas
- Matplotlib, seaborn
- scikit-learn

## How to Run
1. Install required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
2. Run the script in Jupyter Notebook or any Python environment.

## Outputs
- Conversion classification report
- ROC AUC score and curve
- Confusion matrix heatmap
- Insightful charts showing behavior patterns

## Key Insights
- Session time and pricing click are strong indicators of conversion.
- Users clicking pricing pages are significantly more likely to convert.
- The model provides an AUC score to evaluate prediction reliability.

## Future Enhancements
- Add SHAP for interpretability
- Train on real product analytics data
- Test other classification models like XGBoost, RandomForest
