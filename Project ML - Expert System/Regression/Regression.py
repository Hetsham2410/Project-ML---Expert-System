import pandas as pd

dataset = pd.read_csv('insurance.csv')
# Factorize Data
Labels = ['sex', 'smoker', 'region']
for i in range(3):
    dataset[Labels[i]] = pd.factorize(dataset[Labels[i]])[0]
# print(data.head())
X = dataset.iloc[:, 0:6].values
# print(type(X))
# df = pd.DataFrame(X, columns = ['age','sex','bmi','children','smoker','region'])
# print(df)

y = dataset.iloc[:, 6].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state = 0)
from sklearn.preprocessing import StandardScaler

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
#Train Data
lin_reg.fit(X_train, y_train)
#Prediction.
linear_y_pred = lin_reg.predict(X_test)
# result = np.array([y_test,linear_y_pred])
real = pd.DataFrame(y_test, columns=['Real Charges'])
expected = pd.DataFrame(linear_y_pred, columns=['Expected Charges'])
result = real.join(expected)
result = result.iloc[0:10, :]
print(result)




