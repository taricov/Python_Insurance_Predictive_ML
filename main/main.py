# %% [markdown]
# # Import Libraries:

# %%
#cleaning data and EDA 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px
from sklearn.impute import SimpleImputer

#splitting data into training and testing sets
from sklearn.model_selection import train_test_split, GridSearchCV

#normalizing data
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#dummifying data
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

#models that we will use
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#evaluating results
from sklearn.metrics import r2_score

#save models
import joblib

# %%
original_copy = pd.read_csv('./data/insurance.csv')
data = original_copy.copy()
data.head()

# %%
data.isna().sum()

# %%
imputer = SimpleImputer(strategy='mean')
imputer.fit(data['bmi'].values.reshape(-1, 1))
data['bmi'] = imputer.transform(data['bmi'].values.reshape(-1, 1))

data.isna().sum()

# %% [markdown]
# # EDA:

# %%
sns.pairplot(data, hue='sex')

# %%
sns.jointplot(x='age', y='charges', data=data, kind='reg', height=10)

# %%
figure, ax = plt.subplots(4,2, figsize=(20,24))

sns.distplot(data['charges'], ax=ax[0,0])
sns.distplot(data.bmi, ax=ax[0,1])
sns.distplot(data.children, ax=ax[1,0])
sns.distplot(data.age, ax=ax[1,1])

sns.countplot(data.smoker, ax=ax[2,0])
sns.countplot(data.sex, ax=ax[2,1])
sns.countplot(data.region, ax=ax[3,0])
sns.distplot(data.charges, bins=40, ax=ax[3,1])

# %%
sns.lmplot(x="age", y="charges", hue="smoker", data=data, palette = 'muted', height= 7)

# %%
corr = data.corr()
plt.figure(figsize=(15,15))
sns.heatmap(corr, cmap='Wistia', annot=True)
plt.show()

# %%
px.bar(data, 'region', 'charges', title='Interactive bar plot')

# %% [markdown]
# ## Dummfying cartegorical variables:

# %% [markdown]
# ## Sex

# %%
sex = data.iloc[:,1:2].values

le = LabelEncoder()

sex = le.fit_transform(sex[:,0])
sex = pd.DataFrame(sex, columns=['sex'])
le_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

print(le.classes_)

# %% [markdown]
# ## Smoker

# %%
smoker = data.iloc[:,4]

smoker = le.fit_transform(smoker)
smoker = pd.DataFrame(smoker, columns=['smoker'])

smoker.head()

# %%
smoker = data["smoker"] # series 
s_encoded, s_categories = pd.factorize(smoker)
factor_s_mapping = dict(zip(s_categories, s_encoded))
factor_s_mapping
smoker = smoker.map(factor_s_mapping)
factor_s_mapping

# %% [markdown]
# ## Region

# %%
ohe = OneHotEncoder()

region = data.iloc[:,5:6].values
region = ohe.fit_transform(region).toarray()

# %%
cols = []
for i in data.region.unique():
    cols.insert(0,i)

region = pd.DataFrame(region, columns=cols)
region

# %% [markdown]
# ## Splitting data

# %%
X_num = data[['age', 'bmi', 'children']].copy()

features = pd.concat([X_num, region, sex, smoker], axis = 1)
labels = data[['charges']].copy()

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.33, random_state = 0 )

# %%
cleaned_data = pd.concat([features, labels], axis=1)
#backup copy
# cleaned_data.to_csv('./data/cleaned_insurance.csv')
cleaned_data.head()

# %% [markdown]
# ## Exporting files:

# %%
# X_train.to_csv('./data/split_X_train.csv', index = False)
# X_test.to_csv('./data/split_X_test.csv', index = False)
# y_train.to_csv('./data/split_y_train.csv', index = False)
# y_test.to_csv('./data/split_y_test.csv', index = False)

# %% [markdown]
# ## Feature scaling

# %%
sc = StandardScaler()
X_train = sc.fit_transform(X_train.astype(np.float))
X_test = sc.transform(X_test.astype(np.float))

# %% [markdown]
# # Making Predictions:

# %% [markdown]
# ## Base Algo:

# %%
def make_pred(model, name, xtrain, ytrain, xtest, ytest):
    mdl = model.fit(xtrain,ytrain.values.ravel())
    y_train_pred = model.predict(xtrain)
    y_test_pred = model.predict(xtest)
        
    try:
        print(f'{model}.coef_: {model.coef_}')
        print(f'{model}.intercept_: {model.intercept_}')
    except AttributeError:
        print(f'{model} has not coef_ or intercept_')

    print(' %s train score %.3f, %s test score: %.3f' % (name, model.score(xtrain, ytrain), name, model.score(xtest, ytest)))

# %% [markdown]
# # Models:

# %% [markdown]
# ## LinearRegression:

# %%
lr = LinearRegression()

make_pred(lr, 'LR', X_train, y_train, X_test, y_test)

# %% [markdown]
# ## SupportVectorRegressor:

# %%
svr = SVR(kernel='linear', C = 300)

make_pred(svr,'SVR', X_train, y_train, X_test, y_test)

# %% [markdown]
# ## DecisionTree:

# %%
dt = DecisionTreeRegressor(random_state=0)

make_pred(dt, 'DT', X_train, y_train, X_test, y_test)

# %% [markdown]
# ## RandomForest:

# %%
forest = RandomForestRegressor(n_estimators = 100, criterion = 'mse', random_state = 1, n_jobs = -1)

make_pred(forest, "RF", X_train, y_train, X_test, y_test)

# %% [markdown]
# ## PolyNomialRegression:

# %%
#preparing for PolyNomial model:
poly = PolynomialFeatures (degree = 3)

X_poly = poly.fit_transform(features) #feautures (X_train + y_train)

X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, labels, test_size = 0.33, random_state = 0)

#standardize after the new split:
sc = StandardScaler()
X_train_poly = sc.fit_transform(X_train_poly.astype(np.float))
X_test_poly = sc.transform(X_test_poly.astype(np.float))

#PolyNomial LR model:
poly_lr = LinearRegression().fit(X_train_poly, y_train_poly)

make_pred(poly_lr, 'PolyLR', X_train_poly, y_train_poly, X_test_poly, y_test_poly)

# %% [markdown]
# ## Using GridSearchCV:
# #### Tweaking HyperParmaters for electing the best performing model

# %% [markdown]
# ## Base

# %%
def best_params(model, name):
    param_dict = model.best_estimator_.get_params()
    model_str = str(model.estimator).split('(')[0]
    print("\n*** {} Best Parameters ***".format(model_str))
    for k in param_dict:
        print("{}: {}".format(k, param_dict[k]))

    print('\n %s train score %.3f, %s test score: %.3f' % (model_str, model.score(X_train,y_train), model_str, model.score(X_test, y_test)))

    joblib.dump(model.best_estimator_, f'./data/{name}.pkl')


# %% [markdown]
# ## SupportVectorRegressor models:

# %%
svr = SVR()

parameters = dict(kernel=[ 'linear', 'poly'],
                     degree=[2],
                     C=[600, 700, 800, 900],
                     epsilon=[0.0001, 0.00001, 0.000001]
)

model_svr = GridSearchCV(svr, parameters, cv=5, verbose=3)
model_svr = model_svr.fit(X_train,y_train.values.ravel())

best_params(model_svr, "model_SVR")

# %% [markdown]
# ## DecisionTree models:

# %%
dt = DecisionTreeRegressor(random_state=0)

parameters = dict(min_samples_leaf=np.arange(9, 13, 1, int), 
                  max_depth = np.arange(4,7,1, int),
                  min_impurity_decrease = [0, 1, 2],
)

model_dt = GridSearchCV(dt, parameters, cv=5, verbose=3)
model_dt = model_dt.fit(X_train,y_train.values.ravel())

best_params(model_dt, 'model_DT')

# %% [markdown]
# ## RandomForest models:

# %%
rf = RandomForestRegressor(random_state=0)

parameters = dict(n_estimators=[20],
                     max_depth=np.arange(1, 13, 2),
                     min_samples_split=[2],
                     min_samples_leaf= np.arange(1, 15, 2, int),
                     bootstrap=[True, False],
                     oob_score=[False, ]
)


model_rf = GridSearchCV(rf, parameters, cv=5, verbose=3)
model_rf = model_rf.fit(X_train,y_train.values.ravel())

best_params(model_rf, 'model_RF')


