import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# -------------------------------
# Step 1: Synthetic Dataset
# -------------------------------
np.random.seed(42)
n_samples = 1000

df = pd.DataFrame({
    'Session_Time': np.random.gamma(2.0, 10.0, n_samples),  # minutes spent
    'Feature_Usage_Count': np.random.poisson(5, n_samples),  # features used per session
    'Support_Tickets': np.random.binomial(2, 0.2, n_samples),
    'Days_Active': np.random.randint(1, 60, n_samples),
    'Clicks_on_Pricing': np.random.binomial(1, 0.35, n_samples),
    'Referral_Flag': np.random.binomial(1, 0.15, n_samples)
})

# Logic for conversion (ground truth generation)
df['Converted'] = (
    (df['Session_Time'] > 15) &
    (df['Feature_Usage_Count'] > 4) &
    (df['Clicks_on_Pricing'] == 1)
).astype(int)

# -------------------------------
# Step 2: KPIs in Terminal
# -------------------------------
total_users = len(df)
converted_users = df['Converted'].sum()
conversion_rate = converted_users / total_users * 100

print("📊 FREEMIUM CONVERSION ANALYSIS")
print("-" * 40)
print(f"Total Users       : {total_users}")
print(f"Converted Users   : {converted_users}")
print(f"Conversion Rate   : {conversion_rate:.2f}%\n")

# -------------------------------
# Step 3: Modeling
# -------------------------------
X = df.drop('Converted', axis=1)
y = df['Converted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# -------------------------------
# Step 4: Terminal Evaluation
# -------------------------------
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("📈 MODEL PERFORMANCE")
print("-" * 40)
print(f"Accuracy           : {acc:.3f}")
print(f"ROC AUC Score      : {auc:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------
# Step 5: Visualizations
# -------------------------------

# 1. Conversion Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Converted', data=df, palette='coolwarm')
plt.title("Conversion Distribution")
plt.xlabel("Converted (0 = No, 1 = Yes)")
plt.tight_layout()
plt.show()

# 2. Feature Correlation Heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(), annot=True, cmap='YlGnBu')
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# 3. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}", color='green')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='rocket')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 5. Conversion by Pricing Clicks
plt.figure(figsize=(6, 4))
sns.barplot(x='Clicks_on_Pricing', y='Converted', data=df, palette='Set2')
plt.title("Conversion Rate by Pricing Click")
plt.xticks([0, 1], ['No Click', 'Clicked'])
plt.ylabel("Conversion Rate")
plt.tight_layout()
plt.show()

# 6. Conversion vs. Session Time
plt.figure(figsize=(7, 4))
sns.boxplot(x='Converted', y='Session_Time', data=df)
plt.title("Session Time by Conversion")
plt.xlabel("Converted")
plt.tight_layout()
plt.show()
