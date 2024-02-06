import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# loading train data
df = pd.read_csv('./train.csv')
print(f'Train data dimensions: {df.shape}')

# turning categorical features into numeric
for c in df.select_dtypes(include='object').columns:
    df[c] = df[c].astype('category').cat.codes

# checking the correlation with the target, leaving only highly correlated features
ct = df.corr(method='spearman',  numeric_only=False)['SalePrice'].sort_values(ascending=False)
ct = ct[ct.abs() > 0.5]
df = df.loc[:, ct.index]
print(f'New dimensions: {df.shape}')

# checking correlation between the features
cm = df.corr().round(1)
sns.heatmap(cm, annot=True,)

# dropping highly correlated cols (>=0.8)
correlated_cols_to_drop = [
    'TotalBsmtSF', 'GarageYrBlt', 'GarageArea', 'TotRmsAbvGrd'
]

df.drop(columns=correlated_cols_to_drop, inplace=True)
print(f'New dimensions: {df.shape}')

# no more missing values
assert df.isnull().sum().sum() == 0

# split to features and target
X = df.copy()
y = X.pop('SalePrice')

y = y.values.reshape(-1, 1)

# scaling
# scaler = StandardScaler()
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
# y_scaled = scaler.fit_transform(y)

# train-validation split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=.3, random_state=42)

# fitting the model
model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_val)

# # inverse scaling
# preds = scaler.inverse_transform(preds)
# y_val = scaler.inverse_transform(y_val)

# logs

preds_log = np.log(preds)
y_val_log = np.log(y_val)

# assessment
mse = mean_squared_error(y_val_log, preds_log)
r2 = r2_score(y_val, preds)
print(f'MSE: {mse :,.2f} / RMSE: {np.sqrt(mse) :,.2f} / R2: {r2 :,.2f}')

# 10-fold validation
model = LinearRegression()
mse_across_folds = []
r2_across_folds = []
kf = KFold(n_splits=10, random_state=None, shuffle=False)
for i, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
    X_train, y_train = X_scaled[train_idx], y[train_idx]
    X_val, y_val = X_scaled[test_idx], y[test_idx]
    y_val_log = np.log(y_val)
    model.fit(X_train, y_train)
    fold_preds = model.predict(X_val)
    fold_preds_log = np.log(fold_preds)
    fold_mse = mean_squared_error(y_val_log, fold_preds_log)
    fold_r2 = r2_score(y_val_log, fold_preds_log)
    print(f'- Fold {i}. MSE: {fold_mse :,.2f} / RMSE: {np.sqrt(fold_mse) :,.2f} / R2: {fold_r2 :,.2f}')
    mse_across_folds.append(fold_mse)
    r2_across_folds.append(fold_r2)

print(f'Mean MSE: {np.mean(mse_across_folds) :,.2f}')
print(f'Mean RMSE: {np.sqrt(mse_across_folds).mean() :,.2f}')
print(f'Mean R2: {np.mean(r2_across_folds) :,.2f}')

# checking predictions against test submission
test_df = pd.read_csv('./test.csv')
print(f'Test dimensions: {test_df.shape}')

# only relevant cols
test_ids = test_df['Id']
test_df = test_df.loc[:, ct.index[1:]]
test_df.drop(columns=correlated_cols_to_drop, inplace=True)
print(f'New dimensions: {test_df.shape}')

# filling missing values and converting columns to categorical
test_df['KitchenQual'] = test_df['KitchenQual'].fillna(test_df['KitchenQual'].mode()[0])
test_df['GarageCars'] = test_df['GarageCars'].fillna(test_df['GarageCars'].mode()[0])
for c in test_df.select_dtypes(include='object').columns:
    test_df[c] = test_df[c].astype('category').cat.codes

# checking for missing values
assert test_df.isnull().sum().sum() == 0

# scaling
X_test = scaler.fit_transform(test_df)

# importing test submission
test_sub = pd.read_csv('./sample_submission.csv')
assert test_sub['Id'].equals(test_ids)
y_sample = test_sub['SalePrice'].values.reshape(-1, 1)

# testing baseline model and comparing with sample submission
model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
mse = mean_squared_error(y_sample, preds)

print(f'Testing against sample submission:\n'
      f'- MSE: {mse :,.2f}\n'
      f'-RMSE: {np.sqrt(mse) :,.2f}')


plt.figure(figsize=(10, 5))
sns.lineplot(x=range(len(preds)), y=preds.flatten(), label='Our prediction', color='indianred')
sns.lineplot(x=range(len(y_sample)), y=y_sample.flatten(), label='Sample prediction', color='dimgrey')
plt.legend()
plt.title('Prediction vs Sample Submission', fontsize=20)
plt.show()




