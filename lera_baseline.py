import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# loading train data
df = pd.read_csv('./train.csv')

# filling in missing values
# LotFrontage depends on configuration (inside, corner etc)
df['LotFrontage'] = (df['LotFrontage']
                     .fillna(df.groupby('LotConfig')['LotFrontage'].transform('mean').round())
                    )
# creating a new category: not applicable ('NA')
not_appl_cols = [
    'Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
    'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'PoolQC', 'Fence',
    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'
]
df[not_appl_cols] = df[not_appl_cols].fillna('NA')

# 0 for NA
zero_cols = [
    'MasVnrArea', 'GarageYrBlt'
]
df[zero_cols] = df[zero_cols].fillna(0)

# filling with most common
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

# MiscFeature & MiscVal dropping (only have values for 3.7% dataset)
df.drop(columns=['MiscFeature', 'MiscVal'], inplace=True)

# making sure there are no more missing values
assert df.isnull().sum().sum() == 0

# dropping highly correlated cols

cols_to_drop = [
   'TotalBsmtSF', 'GarageYrBlt', 'GarageArea'
]
df.drop(columns=cols_to_drop, inplace=True)

# turning categorical features into numeric
for c in df.select_dtypes(include='object').columns:
    df[c] = df[c].astype('category').cat.codes

# split to features and target
X = df.copy()
y = X.pop('SalePrice')

y = y.values.reshape(-1, 1)

# scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

# train-validation split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=.3, random_state=42)

# fitting the model
model = LinearRegression()
model.fit(X_train, y_train)
preds_scaled = model.predict(X_val)

# inverse scaling
preds = scaler.inverse_transform(preds_scaled)
true_y = scaler.inverse_transform(y_val)

# assessment
mse = mean_squared_error(true_y, preds)
r2 = r2_score(true_y, preds)
print(f'MSE: {mse :,.2f} / RMSE: {np.sqrt(mse) :,.2f} / R2: {r2 :,.2f}')

# 10-fold validation
model = LinearRegression()

mse_across_folds = []
r2_across_folds = []

kf = KFold(n_splits=10, random_state=None, shuffle=False)
for i, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
    X_train, y_train = X_scaled[train_idx], y_scaled[train_idx]
    X_val, y_val = X_scaled[test_idx], y_scaled[test_idx]
    model.fit(X_train, y_train)
    fold_preds = model.predict(X_val)
    fold_mse = mean_squared_error(y_val, fold_preds)
    fold_r2 = r2_score(y_val, fold_preds)
    print(f'- Fold {i}. MSE: {fold_mse :,.2f} / RMSE: {np.sqrt(fold_mse) :,.2f} / R2: {fold_r2 :,.2f}')
    mse_across_folds.append(fold_mse)
    r2_across_folds.append(fold_r2)

print(f'Mean MSE: {np.mean(mse_across_folds) :,.2f}')
print(f'Mean RMSE: {np.sqrt(mse_across_folds).mean() :,.2f}')
print(f'Mean R2: {np.mean(r2_across_folds) :,.2f}')
