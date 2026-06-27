
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("cardekho.csv")

#  clean data
df = df.dropna()
df = df.drop("name", axis=1)

#  encoding
df = pd.get_dummies(df, drop_first=True)

#  input & output
X = df.drop("selling_price", axis=1)
y = df["selling_price"]

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)
#train test
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#  model train
model = LinearRegression()
model.fit(X_train, y_train)

#  prediction
y_pred = model.predict(X_test)

# print some predictions
print("Some Predictions:")
print(y_pred[:5])

#  graph 

# plt.figure(figsize=(6,5))

# plt.scatter(y_test, y_pred, color="blue")

# plt.xlabel("Actual Price")
# plt.ylabel("Predicted Price")
# plt.title("Actual vs Predicted Car Price")
# plt.show()
sns.lineplot(x=y_test, y=y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Price")
plt.show()