import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline

# Load dataset
df = pd.read_csv("delivery_data.csv")  # Replace with your file path

# Encode categorical columns
label_cols = ['Supplier Type', 'Order Priority', 'Transport Mode', 'In Stock', 'Order Time', 'Region', 'Weather']
for col in label_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# X and Y
X = df.drop("Delivery Time", axis=1)
Y = df["Delivery Time"]

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 1. Linear Regression
print("----- Linear Regression -----")
lr = LinearRegression()
lr.fit(X_train, Y_train)
print("Score:", lr.score(X_test, Y_test))
print("Predictions:", lr.predict(X_test[:5]), "\n")

# 2. Polynomial Regression
print("----- Polynomial Regression -----")
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(X_train, Y_train)
print("Score:", poly_model.score(X_test, Y_test))
print("Predictions:", poly_model.predict(X_test[:5]), "\n")

# 3. Ridge Regression
print("----- Ridge Regression -----")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, Y_train)
print("Score:", ridge.score(X_test, Y_test))
print("Predictions:", ridge.predict(X_test[:5]), "\n")

# 4. Support Vector Regression
print("----- Support Vector Regression -----")
svr = make_pipeline(StandardScaler(), SVR(kernel='rbf'))
svr.fit(X_train, Y_train)
print("Score:", svr.score(X_test, Y_test))
print("Predictions:", svr.predict(X_test[:5]), "\n")

# 5. Lasso Regression
print("----- Lasso Regression -----")
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, Y_train)
print("Score:", lasso.score(X_test, Y_test))
print("Predictions:", lasso.predict(X_test[:5]), "\n")

# 6. Elastic Net Regression
print("----- Elastic Net Regression -----")
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train, Y_train)
print("Score:", elastic.score(X_test, Y_test))
print("Predictions:", elastic.predict(X_test[:5]), "\n")