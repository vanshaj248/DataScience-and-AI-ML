import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ── 1. Load Data ──────────────────────────────────────────
housing = fetch_california_housing(as_frame=True)
df = housing.frame
print(df.head())
print(df.describe())

# ── 2. Exploratory Data Analysis ─────────────────────────
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()

plt.figure(figsize=(6,4))
df['MedHouseVal'].hist(bins=50, color='steelblue', edgecolor='white')
plt.title('Distribution of House Prices')
plt.xlabel('Median House Value ($100k)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('price_distribution.png')
plt.show()

# ── 3. Feature & Target Split ─────────────────────────────
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# ── 4. Train/Test Split ───────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ── 5. Feature Scaling ────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 6. Model 1: Linear Regression ────────────────────────
lr = LinearRegression()
lr.fit(X_train_sc, y_train)
y_pred_lr = lr.predict(X_test_sc)
print("Linear Regression:")
print(f"  RMSE : {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.4f}")
print(f"  R²   : {r2_score(y_test, y_pred_lr):.4f}")

# ── 7. Model 2: Random Forest ─────────────────────────────
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\nRandom Forest:")
print(f"  RMSE : {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.4f}")
print(f"  R²   : {r2_score(y_test, y_pred_rf):.4f}")

# ── 8. Feature Importance (RF) ────────────────────────────
feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
feat_imp.sort_values().plot(kind='barh', color='teal', figsize=(7,4))
plt.title('Feature Importances (Random Forest)')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# ── 9. Actual vs Predicted Plot ───────────────────────────
plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred_rf, alpha=0.3, color='royalblue', s=10)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted (Random Forest)')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.show()

# ── 10. Residual Analysis ─────────────────────────────────
residuals = y_test - y_pred_rf
plt.figure(figsize=(6,4))
plt.hist(residuals, bins=50, color='salmon', edgecolor='white')
plt.title('Residual Distribution')
plt.xlabel('Residual')
plt.tight_layout()
plt.savefig('residuals.png')
plt.show()