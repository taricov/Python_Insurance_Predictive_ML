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


#reading the data
original_copy = pd.read_csv('./data/insurance.csv')
data = original_copy.copy()
data.head()

#getting to meet the data
data.isna().sum()

#wrangling the na/empty vals
imputer = SimpleImputer(strategy='mean')
imputer.fit(data['bmi'].values.reshape(-1, 1))
data['bmi'] = imputer.transform(data['bmi'].values.reshape(-1, 1))

data.isna().sum()

#visualization
sns.pairplot(data, hue='sex')
sns.jointplot(x='age', y='charges', data=data, kind='reg', height=10)

figure, ax = plt.subplots(4,2, figsize=(20,24))

sns.distplot(data['charges'], ax=ax[0,0])
sns.distplot(data.bmi, ax=ax[0,1])
sns.distplot(data.children, ax=ax[1,0])
sns.distplot(data.age, ax=ax[1,1])

sns.countplot(data.smoker, ax=ax[2,0])
sns.countplot(data.sex, ax=ax[2,1])
sns.countplot(data.region, ax=ax[3,0])
sns.distplot(data.charges, bins=40, ax=ax[3,1])


sns.lmplot(x="age", y="charges", hue="smoker", data=data, palette = 'muted', height= 7)


corr = data.corr()
plt.figure(figsize=(15,15))
sns.heatmap(corr, cmap='Wistia', annot=True)
plt.show()


px.bar(data, 'region', 'charges', title='Interactive bar plot')
Dummfying cartegorical variables:
Sex
sex = data.iloc[:,1:2].values

le = LabelEncoder()

sex = le.fit_transform(sex[:,0])
sex = pd.DataFrame(sex, columns=['sex'])
le_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

print(le.classes_)
['female' 'male']
Smoker
smoker = data.iloc[:,4]

smoker = le.fit_transform(smoker)
smoker = pd.DataFrame(smoker, columns=['smoker'])

smoker.head()
smoker
0	1
1	0
2	0
3	0
4	0
smoker = data["smoker"] # series 
s_encoded, s_categories = pd.factorize(smoker)
factor_s_mapping = dict(zip(s_categories, s_encoded))
factor_s_mapping
smoker = smoker.map(factor_s_mapping)
factor_s_mapping
{'yes': 0, 'no': 1}
Region
ohe = OneHotEncoder()

region = data.iloc[:,5:6].values
region = ohe.fit_transform(region).toarray()
cols = []
for i in data.region.unique():
    cols.insert(0,i)

region = pd.DataFrame(region, columns=cols)
region
northeast	northwest	southeast	southwest
0	0.0	0.0	0.0	1.0
1	0.0	0.0	1.0	0.0
2	0.0	0.0	1.0	0.0
3	0.0	1.0	0.0	0.0
4	0.0	1.0	0.0	0.0
...	...	...	...	...
1333	0.0	1.0	0.0	0.0
1334	1.0	0.0	0.0	0.0
1335	0.0	0.0	1.0	0.0
1336	0.0	0.0	0.0	1.0
1337	0.0	1.0	0.0	0.0
1338 rows × 4 columns

Splitting data
X_num = data[['age', 'bmi', 'children']].copy()

features = pd.concat([X_num, region, sex, smoker], axis = 1)
labels = data[['charges']].copy()

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.33, random_state = 0 )
cleaned_data = pd.concat([features, labels], axis=1)
#backup copy
# cleaned_data.to_csv('./data/cleaned_insurance.csv')
cleaned_data.head()
age	bmi	children	northeast	northwest	southeast	southwest	sex	smoker	charges
0	19	27.900	0	0.0	0.0	0.0	1.0	0	0	16884.92400
1	18	33.770	1	0.0	0.0	1.0	0.0	1	1	1725.55230
2	28	33.000	3	0.0	0.0	1.0	0.0	1	1	4449.46200
3	33	22.705	0	0.0	1.0	0.0	0.0	1	1	21984.47061
4	32	28.880	0	0.0	1.0	0.0	0.0	1	1	3866.85520
Exporting files:
# X_train.to_csv('./data/split_X_train.csv', index = False)
# X_test.to_csv('./data/split_X_test.csv', index = False)
# y_train.to_csv('./data/split_y_train.csv', index = False)
# y_test.to_csv('./data/split_y_test.csv', index = False)
Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train.astype(np.float))
X_test = sc.transform(X_test.astype(np.float))
Making Predictions:
Base Algo:
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
Models:
LinearRegression:
lr = LinearRegression()

make_pred(lr, 'LR', X_train, y_train, X_test, y_test)
LinearRegression().coef_: [ 3624.36356197  1966.90473927   661.35603447   242.57758422
   -29.49212715  -104.19142495   -99.14488063   -44.54996175
 -9310.54961689]
LinearRegression().intercept_: 13141.350831640624
 LR train score 0.728, LR test score: 0.786
SupportVectorRegressor:
svr = SVR(kernel='linear', C = 300)

make_pred(svr,'SVR', X_train, y_train, X_test, y_test)
SVR(C=300, kernel='linear').coef_: [[ 3652.88084977   202.46754224   475.97904631   179.67244105
     32.53987879  -126.02330398   -76.96879816  -184.83769075
  -6571.51888242]]
SVR(C=300, kernel='linear').intercept_: [10370.82641928]
 SVR train score 0.598, SVR test score: 0.628
DecisionTree:
dt = DecisionTreeRegressor(random_state=0)

make_pred(dt, 'DT', X_train, y_train, X_test, y_test)
DecisionTreeRegressor(random_state=0) has not coef_ or intercept_
 DT train score 0.999, DT test score: 0.706
RandomForest:
forest = RandomForestRegressor(n_estimators = 100, criterion = 'mse', random_state = 1, n_jobs = -1)

make_pred(forest, "RF", X_train, y_train, X_test, y_test)
RandomForestRegressor(n_jobs=-1, random_state=1) has not coef_ or intercept_
 RF train score 0.973, RF test score: 0.860
PolyNomialRegression:
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
LinearRegression().coef_: [ 1.12972993e-11  1.85641383e+16 -1.48642234e+16  1.97016627e+16
 -5.12296309e+15 -5.37644849e+15 -3.57080268e+15  2.59222606e+16
  5.24600197e+16  1.26616401e+16 -5.35130776e+15 -2.07369741e+16
  1.10452658e+16 -2.38592437e+15 -2.66411429e+16 -2.57544319e+16
 -3.31905799e+16  4.45120956e+16 -2.44396287e+16 -1.50253870e+16
 -4.26356291e+16  3.83043816e+16  1.88084260e+16  4.76771878e+16
  2.35733900e+15  4.10703563e+15  1.06182432e+16 -4.53888998e+16
 -2.47702056e+16 -2.78463493e+16 -6.33714970e+16 -7.46611583e+16
  1.45734658e+16  9.89745072e+15  5.08215452e+15  8.39990079e+15
 -1.39425629e+16 -6.24891773e+15 -2.17767524e+16 -8.89944289e+15
  2.26045509e+15  9.26109428e+15 -1.12923925e+15 -1.06753782e+16
  1.73926155e+16  9.40398283e+14  2.22948657e+15 -2.97531954e+16
  1.96722556e+15 -1.54219487e+16 -1.75073739e+16 -6.62890716e+15
  4.10510273e+15  1.68330615e+16 -9.25241337e+15  1.10260000e+04
 -3.05400000e+03  7.54800000e+03  4.43396092e+15  4.53797803e+15
  4.60715777e+15  4.43963787e+15 -1.26200000e+03  5.66950000e+03
  2.42100000e+03 -6.87500000e+02  2.20732108e+16  2.21070475e+16
  2.56988739e+16  2.34348079e+16 -7.25700000e+03 -4.32562500e+03
  2.89700000e+03 -6.33824692e+15 -6.79017009e+15 -6.98082453e+15
 -7.27775802e+15 -2.02200000e+03  8.08000000e+02 -2.16313839e+16
 -1.59077406e+14 -3.06470901e+14 -5.97660569e+14 -7.90705725e+15
  5.51527459e+14  2.08040788e+15 -2.44122815e+15 -1.94218762e+15
 -8.16681176e+15  5.62989248e+14  7.00506338e+14 -1.40599475e+15
 -8.69766125e+15  5.52781859e+14  8.81192694e+15 -8.01893502e+15
  5.83825708e+14 -3.14640273e+16 -6.18000000e+02  2.37723657e+16
 -1.99410000e+04 -3.46400000e+03  1.57687176e+16  1.53911251e+16
  2.16313950e+16  1.75651778e+16  1.07100000e+03 -2.09750000e+03
 -1.63000000e+03  2.38905616e+16  2.49808930e+16  2.86361222e+16
  2.92784615e+16  3.17000000e+03  7.68600000e+03 -7.79917117e+15
  0.00000000e+00  0.00000000e+00  0.00000000e+00  7.08996296e+15
  6.25965777e+15  1.18675768e+16  0.00000000e+00  0.00000000e+00
  7.00008943e+15  6.39386232e+15 -1.08206830e+16  0.00000000e+00
  8.97892623e+15  7.44376948e+15  3.05781808e+16  7.77660017e+15
  6.91810084e+15 -1.61040209e+16  1.07000000e+03 -1.78588878e+16
  1.94600000e+03  2.40204804e+16  2.24798902e+16  2.39926568e+16
  3.12990712e+16 -6.50000000e+01 -2.12200000e+03  1.28095313e+16
  0.00000000e+00  0.00000000e+00  0.00000000e+00 -9.26061437e+15
 -3.47227209e+15  1.58555562e+16  0.00000000e+00  0.00000000e+00
 -8.35317641e+15 -3.48939963e+15  5.10627400e+16  0.00000000e+00
 -8.91086398e+15 -3.51347094e+15  6.08648209e+16 -9.91748688e+15
 -4.14199832e+15  1.97321516e+15  1.02400000e+03 -3.72808168e+15
 -4.09772371e+15  0.00000000e+00  0.00000000e+00  0.00000000e+00
 -7.19114014e+15  2.34449018e+15  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00 -7.77506799e+15 -5.26964823e+15
  1.30349582e+16 -1.08384805e+15  0.00000000e+00  0.00000000e+00
 -1.29408135e+16 -5.36932296e+15  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00 -1.29744912e+16 -5.50289135e+15 -5.33553281e+15
 -1.73486741e+15  0.00000000e+00 -4.84675550e+15  2.39535764e+15
  0.00000000e+00  0.00000000e+00  0.00000000e+00 -4.84675550e+15
 -5.47455454e+15  2.39535764e+15 -1.47892778e+16 -9.91706777e+15
  6.78003339e+15 -9.91706777e+15 -5.55889949e+15  6.78003339e+15
  5.47265074e+14 -3.25351522e+15 -4.55592767e+15 -1.01253040e+16]
LinearRegression().intercept_: 13340.68156621422
 PolyLR train score 0.809, PolyLR test score: 0.829
Using GridSearchCV:
Tweaking HyperParmaters for electing the best performing model
Base
def best_params(model, name):
    param_dict = model.best_estimator_.get_params()
    model_str = str(model.estimator).split('(')[0]
    print("\n*** {} Best Parameters ***".format(model_str))
    for k in param_dict:
        print("{}: {}".format(k, param_dict[k]))

    print('\n %s train score %.3f, %s test score: %.3f' % (model_str, model.score(X_train,y_train), model_str, model.score(X_test, y_test)))

    joblib.dump(model.best_estimator_, f'./data/{name}.pkl')
SupportVectorRegressor models:
svr = SVR()

parameters = dict(kernel=[ 'linear', 'poly'],
                     degree=[2],
                     C=[600, 700, 800, 900],
                     epsilon=[0.0001, 0.00001, 0.000001]
)

model_svr = GridSearchCV(svr, parameters, cv=5, verbose=3)
model_svr = model_svr.fit(X_train,y_train.values.ravel())

best_params(model_svr, "model_SVR")
Fitting 5 folds for each of 24 candidates, totalling 120 fits
[CV 1/5] END C=600, degree=2, epsilon=0.0001, kernel=linear;, score=0.671 total time=   0.0s
[CV 2/5] END C=600, degree=2, epsilon=0.0001, kernel=linear;, score=0.663 total time=   0.0s
[CV 3/5] END C=600, degree=2, epsilon=0.0001, kernel=linear;, score=0.571 total time=   0.0s
[CV 4/5] END C=600, degree=2, epsilon=0.0001, kernel=linear;, score=0.636 total time=   0.0s
[CV 5/5] END C=600, degree=2, epsilon=0.0001, kernel=linear;, score=0.555 total time=   0.0s
[CV 1/5] END C=600, degree=2, epsilon=0.0001, kernel=poly;, score=0.485 total time=   0.0s
[CV 2/5] END C=600, degree=2, epsilon=0.0001, kernel=poly;, score=0.336 total time=   0.0s
[CV 3/5] END C=600, degree=2, epsilon=0.0001, kernel=poly;, score=0.278 total time=   0.0s
[CV 4/5] END C=600, degree=2, epsilon=0.0001, kernel=poly;, score=0.372 total time=   0.0s
[CV 5/5] END C=600, degree=2, epsilon=0.0001, kernel=poly;, score=0.222 total time=   0.0s
[CV 1/5] END C=600, degree=2, epsilon=1e-05, kernel=linear;, score=0.671 total time=   0.1s
[CV 2/5] END C=600, degree=2, epsilon=1e-05, kernel=linear;, score=0.663 total time=   0.0s
[CV 3/5] END C=600, degree=2, epsilon=1e-05, kernel=linear;, score=0.571 total time=   0.0s
[CV 4/5] END C=600, degree=2, epsilon=1e-05, kernel=linear;, score=0.636 total time=   0.0s
[CV 5/5] END C=600, degree=2, epsilon=1e-05, kernel=linear;, score=0.555 total time=   0.0s
[CV 1/5] END C=600, degree=2, epsilon=1e-05, kernel=poly;, score=0.485 total time=   0.0s
[CV 2/5] END C=600, degree=2, epsilon=1e-05, kernel=poly;, score=0.336 total time=   0.0s
[CV 3/5] END C=600, degree=2, epsilon=1e-05, kernel=poly;, score=0.278 total time=   0.0s
[CV 4/5] END C=600, degree=2, epsilon=1e-05, kernel=poly;, score=0.372 total time=   0.0s
[CV 5/5] END C=600, degree=2, epsilon=1e-05, kernel=poly;, score=0.222 total time=   0.0s
[CV 1/5] END C=600, degree=2, epsilon=1e-06, kernel=linear;, score=0.671 total time=   0.0s
[CV 2/5] END C=600, degree=2, epsilon=1e-06, kernel=linear;, score=0.663 total time=   0.0s
[CV 3/5] END C=600, degree=2, epsilon=1e-06, kernel=linear;, score=0.571 total time=   0.0s
[CV 4/5] END C=600, degree=2, epsilon=1e-06, kernel=linear;, score=0.636 total time=   0.0s
[CV 5/5] END C=600, degree=2, epsilon=1e-06, kernel=linear;, score=0.555 total time=   0.0s
[CV 1/5] END C=600, degree=2, epsilon=1e-06, kernel=poly;, score=0.485 total time=   0.0s
[CV 2/5] END C=600, degree=2, epsilon=1e-06, kernel=poly;, score=0.336 total time=   0.0s
[CV 3/5] END C=600, degree=2, epsilon=1e-06, kernel=poly;, score=0.278 total time=   0.0s
[CV 4/5] END C=600, degree=2, epsilon=1e-06, kernel=poly;, score=0.372 total time=   0.0s
[CV 5/5] END C=600, degree=2, epsilon=1e-06, kernel=poly;, score=0.222 total time=   0.0s
[CV 1/5] END C=700, degree=2, epsilon=0.0001, kernel=linear;, score=0.671 total time=   0.0s
[CV 2/5] END C=700, degree=2, epsilon=0.0001, kernel=linear;, score=0.662 total time=   0.0s
[CV 3/5] END C=700, degree=2, epsilon=0.0001, kernel=linear;, score=0.572 total time=   0.0s
[CV 4/5] END C=700, degree=2, epsilon=0.0001, kernel=linear;, score=0.636 total time=   0.0s
[CV 5/5] END C=700, degree=2, epsilon=0.0001, kernel=linear;, score=0.555 total time=   0.0s
[CV 1/5] END C=700, degree=2, epsilon=0.0001, kernel=poly;, score=0.530 total time=   0.0s
[CV 2/5] END C=700, degree=2, epsilon=0.0001, kernel=poly;, score=0.361 total time=   0.0s
[CV 3/5] END C=700, degree=2, epsilon=0.0001, kernel=poly;, score=0.311 total time=   0.0s
[CV 4/5] END C=700, degree=2, epsilon=0.0001, kernel=poly;, score=0.405 total time=   0.0s
[CV 5/5] END C=700, degree=2, epsilon=0.0001, kernel=poly;, score=0.249 total time=   0.0s
[CV 1/5] END C=700, degree=2, epsilon=1e-05, kernel=linear;, score=0.671 total time=   0.0s
[CV 2/5] END C=700, degree=2, epsilon=1e-05, kernel=linear;, score=0.662 total time=   0.0s
[CV 3/5] END C=700, degree=2, epsilon=1e-05, kernel=linear;, score=0.572 total time=   0.0s
[CV 4/5] END C=700, degree=2, epsilon=1e-05, kernel=linear;, score=0.636 total time=   0.0s
[CV 5/5] END C=700, degree=2, epsilon=1e-05, kernel=linear;, score=0.555 total time=   0.0s
[CV 1/5] END C=700, degree=2, epsilon=1e-05, kernel=poly;, score=0.530 total time=   0.0s
[CV 2/5] END C=700, degree=2, epsilon=1e-05, kernel=poly;, score=0.361 total time=   0.0s
[CV 3/5] END C=700, degree=2, epsilon=1e-05, kernel=poly;, score=0.311 total time=   0.0s
[CV 4/5] END C=700, degree=2, epsilon=1e-05, kernel=poly;, score=0.405 total time=   0.0s
[CV 5/5] END C=700, degree=2, epsilon=1e-05, kernel=poly;, score=0.249 total time=   0.0s
[CV 1/5] END C=700, degree=2, epsilon=1e-06, kernel=linear;, score=0.671 total time=   0.0s
[CV 2/5] END C=700, degree=2, epsilon=1e-06, kernel=linear;, score=0.662 total time=   0.0s
[CV 3/5] END C=700, degree=2, epsilon=1e-06, kernel=linear;, score=0.572 total time=   0.0s
[CV 4/5] END C=700, degree=2, epsilon=1e-06, kernel=linear;, score=0.636 total time=   0.0s
[CV 5/5] END C=700, degree=2, epsilon=1e-06, kernel=linear;, score=0.555 total time=   0.0s
[CV 1/5] END C=700, degree=2, epsilon=1e-06, kernel=poly;, score=0.530 total time=   0.0s
[CV 2/5] END C=700, degree=2, epsilon=1e-06, kernel=poly;, score=0.361 total time=   0.0s
[CV 3/5] END C=700, degree=2, epsilon=1e-06, kernel=poly;, score=0.311 total time=   0.0s
[CV 4/5] END C=700, degree=2, epsilon=1e-06, kernel=poly;, score=0.405 total time=   0.0s
[CV 5/5] END C=700, degree=2, epsilon=1e-06, kernel=poly;, score=0.249 total time=   0.0s
[CV 1/5] END C=800, degree=2, epsilon=0.0001, kernel=linear;, score=0.686 total time=   0.0s
[CV 2/5] END C=800, degree=2, epsilon=0.0001, kernel=linear;, score=0.621 total time=   0.0s
[CV 3/5] END C=800, degree=2, epsilon=0.0001, kernel=linear;, score=0.572 total time=   0.0s
[CV 4/5] END C=800, degree=2, epsilon=0.0001, kernel=linear;, score=0.588 total time=   0.0s
[CV 5/5] END C=800, degree=2, epsilon=0.0001, kernel=linear;, score=0.555 total time=   0.0s
[CV 1/5] END C=800, degree=2, epsilon=0.0001, kernel=poly;, score=0.569 total time=   0.0s
[CV 2/5] END C=800, degree=2, epsilon=0.0001, kernel=poly;, score=0.378 total time=   0.0s
[CV 3/5] END C=800, degree=2, epsilon=0.0001, kernel=poly;, score=0.345 total time=   0.0s
[CV 4/5] END C=800, degree=2, epsilon=0.0001, kernel=poly;, score=0.442 total time=   0.0s
[CV 5/5] END C=800, degree=2, epsilon=0.0001, kernel=poly;, score=0.268 total time=   0.0s
[CV 1/5] END C=800, degree=2, epsilon=1e-05, kernel=linear;, score=0.686 total time=   0.0s
[CV 2/5] END C=800, degree=2, epsilon=1e-05, kernel=linear;, score=0.621 total time=   0.0s
[CV 3/5] END C=800, degree=2, epsilon=1e-05, kernel=linear;, score=0.572 total time=   0.0s
[CV 4/5] END C=800, degree=2, epsilon=1e-05, kernel=linear;, score=0.588 total time=   0.0s
[CV 5/5] END C=800, degree=2, epsilon=1e-05, kernel=linear;, score=0.555 total time=   0.0s
[CV 1/5] END C=800, degree=2, epsilon=1e-05, kernel=poly;, score=0.569 total time=   0.0s
[CV 2/5] END C=800, degree=2, epsilon=1e-05, kernel=poly;, score=0.378 total time=   0.0s
[CV 3/5] END C=800, degree=2, epsilon=1e-05, kernel=poly;, score=0.345 total time=   0.0s
[CV 4/5] END C=800, degree=2, epsilon=1e-05, kernel=poly;, score=0.442 total time=   0.0s
[CV 5/5] END C=800, degree=2, epsilon=1e-05, kernel=poly;, score=0.268 total time=   0.0s
[CV 1/5] END C=800, degree=2, epsilon=1e-06, kernel=linear;, score=0.686 total time=   0.0s
[CV 2/5] END C=800, degree=2, epsilon=1e-06, kernel=linear;, score=0.621 total time=   0.0s
[CV 3/5] END C=800, degree=2, epsilon=1e-06, kernel=linear;, score=0.572 total time=   0.0s
[CV 4/5] END C=800, degree=2, epsilon=1e-06, kernel=linear;, score=0.588 total time=   0.0s
[CV 5/5] END C=800, degree=2, epsilon=1e-06, kernel=linear;, score=0.555 total time=   0.0s
[CV 1/5] END C=800, degree=2, epsilon=1e-06, kernel=poly;, score=0.569 total time=   0.0s
[CV 2/5] END C=800, degree=2, epsilon=1e-06, kernel=poly;, score=0.378 total time=   0.0s
[CV 3/5] END C=800, degree=2, epsilon=1e-06, kernel=poly;, score=0.345 total time=   0.0s
[CV 4/5] END C=800, degree=2, epsilon=1e-06, kernel=poly;, score=0.442 total time=   0.0s
[CV 5/5] END C=800, degree=2, epsilon=1e-06, kernel=poly;, score=0.268 total time=   0.0s
[CV 1/5] END C=900, degree=2, epsilon=0.0001, kernel=linear;, score=0.709 total time=   0.0s
[CV 2/5] END C=900, degree=2, epsilon=0.0001, kernel=linear;, score=0.610 total time=   0.0s
[CV 3/5] END C=900, degree=2, epsilon=0.0001, kernel=linear;, score=0.573 total time=   0.0s
[CV 4/5] END C=900, degree=2, epsilon=0.0001, kernel=linear;, score=0.571 total time=   0.0s
[CV 5/5] END C=900, degree=2, epsilon=0.0001, kernel=linear;, score=0.555 total time=   0.0s
[CV 1/5] END C=900, degree=2, epsilon=0.0001, kernel=poly;, score=0.583 total time=   0.0s
[CV 2/5] END C=900, degree=2, epsilon=0.0001, kernel=poly;, score=0.386 total time=   0.0s
[CV 3/5] END C=900, degree=2, epsilon=0.0001, kernel=poly;, score=0.374 total time=   0.0s
[CV 4/5] END C=900, degree=2, epsilon=0.0001, kernel=poly;, score=0.461 total time=   0.0s
[CV 5/5] END C=900, degree=2, epsilon=0.0001, kernel=poly;, score=0.295 total time=   0.0s
[CV 1/5] END C=900, degree=2, epsilon=1e-05, kernel=linear;, score=0.709 total time=   0.0s
[CV 2/5] END C=900, degree=2, epsilon=1e-05, kernel=linear;, score=0.610 total time=   0.0s
[CV 3/5] END C=900, degree=2, epsilon=1e-05, kernel=linear;, score=0.573 total time=   0.0s
[CV 4/5] END C=900, degree=2, epsilon=1e-05, kernel=linear;, score=0.571 total time=   0.0s
[CV 5/5] END C=900, degree=2, epsilon=1e-05, kernel=linear;, score=0.555 total time=   0.0s
[CV 1/5] END C=900, degree=2, epsilon=1e-05, kernel=poly;, score=0.583 total time=   0.0s
[CV 2/5] END C=900, degree=2, epsilon=1e-05, kernel=poly;, score=0.386 total time=   0.0s
[CV 3/5] END C=900, degree=2, epsilon=1e-05, kernel=poly;, score=0.374 total time=   0.0s
[CV 4/5] END C=900, degree=2, epsilon=1e-05, kernel=poly;, score=0.461 total time=   0.0s
[CV 5/5] END C=900, degree=2, epsilon=1e-05, kernel=poly;, score=0.295 total time=   0.0s
[CV 1/5] END C=900, degree=2, epsilon=1e-06, kernel=linear;, score=0.709 total time=   0.0s
[CV 2/5] END C=900, degree=2, epsilon=1e-06, kernel=linear;, score=0.610 total time=   0.0s
[CV 3/5] END C=900, degree=2, epsilon=1e-06, kernel=linear;, score=0.573 total time=   0.0s
[CV 4/5] END C=900, degree=2, epsilon=1e-06, kernel=linear;, score=0.571 total time=   0.0s
[CV 5/5] END C=900, degree=2, epsilon=1e-06, kernel=linear;, score=0.555 total time=   0.0s
[CV 1/5] END C=900, degree=2, epsilon=1e-06, kernel=poly;, score=0.583 total time=   0.0s
[CV 2/5] END C=900, degree=2, epsilon=1e-06, kernel=poly;, score=0.386 total time=   0.0s
[CV 3/5] END C=900, degree=2, epsilon=1e-06, kernel=poly;, score=0.374 total time=   0.0s
[CV 4/5] END C=900, degree=2, epsilon=1e-06, kernel=poly;, score=0.461 total time=   0.0s
[CV 5/5] END C=900, degree=2, epsilon=1e-06, kernel=poly;, score=0.295 total time=   0.0s

*** SVR Best Parameters ***
C: 700
cache_size: 200
coef0: 0.0
degree: 2
epsilon: 1e-06
gamma: scale
kernel: linear
max_iter: -1
shrinking: True
tol: 0.001
verbose: False

 SVR train score 0.683, SVR test score: 0.734
DecisionTree models:
dt = DecisionTreeRegressor(random_state=0)

parameters = dict(min_samples_leaf=np.arange(9, 13, 1, int), 
                  max_depth = np.arange(4,7,1, int),
                  min_impurity_decrease = [0, 1, 2],
)

model_dt = GridSearchCV(dt, parameters, cv=5, verbose=3)
model_dt = model_dt.fit(X_train,y_train.values.ravel())

best_params(model_dt, 'model_DT')
Fitting 5 folds for each of 36 candidates, totalling 180 fits
[CV 1/5] END max_depth=4, min_impurity_decrease=0, min_samples_leaf=9;, score=0.817 total time=   0.0s
[CV 2/5] END max_depth=4, min_impurity_decrease=0, min_samples_leaf=9;, score=0.819 total time=   0.0s
[CV 3/5] END max_depth=4, min_impurity_decrease=0, min_samples_leaf=9;, score=0.846 total time=   0.0s
[CV 4/5] END max_depth=4, min_impurity_decrease=0, min_samples_leaf=9;, score=0.812 total time=   0.0s
[CV 5/5] END max_depth=4, min_impurity_decrease=0, min_samples_leaf=9;, score=0.796 total time=   0.0s
[CV 1/5] END max_depth=4, min_impurity_decrease=0, min_samples_leaf=10;, score=0.818 total time=   0.0s
[CV 2/5] END max_depth=4, min_impurity_decrease=0, min_samples_leaf=10;, score=0.819 total time=   0.0s
[CV 3/5] END max_depth=4, min_impurity_decrease=0, min_samples_leaf=10;, score=0.846 total time=   0.0s
[CV 4/5] END max_depth=4, min_impurity_decrease=0, min_samples_leaf=10;, score=0.815 total time=   0.0s
[CV 5/5] END max_depth=4, min_impurity_decrease=0, min_samples_leaf=10;, score=0.796 total time=   0.0s
[CV 1/5] END max_depth=4, min_impurity_decrease=0, min_samples_leaf=11;, score=0.818 total time=   0.0s
[CV 2/5] END max_depth=4, min_impurity_decrease=0, min_samples_leaf=11;, score=0.820 total time=   0.0s
[CV 3/5] END max_depth=4, min_impurity_decrease=0, min_samples_leaf=11;, score=0.845 total time=   0.0s
[CV 4/5] END max_depth=4, min_impurity_decrease=0, min_samples_leaf=11;, score=0.815 total time=   0.0s
[CV 5/5] END max_depth=4, min_impurity_decrease=0, min_samples_leaf=11;, score=0.796 total time=   0.0s
[CV 1/5] END max_depth=4, min_impurity_decrease=0, min_samples_leaf=12;, score=0.818 total time=   0.0s
[CV 2/5] END max_depth=4, min_impurity_decrease=0, min_samples_leaf=12;, score=0.820 total time=   0.0s
[CV 3/5] END max_depth=4, min_impurity_decrease=0, min_samples_leaf=12;, score=0.845 total time=   0.0s
[CV 4/5] END max_depth=4, min_impurity_decrease=0, min_samples_leaf=12;, score=0.815 total time=   0.0s
[CV 5/5] END max_depth=4, min_impurity_decrease=0, min_samples_leaf=12;, score=0.796 total time=   0.0s
[CV 1/5] END max_depth=4, min_impurity_decrease=1, min_samples_leaf=9;, score=0.817 total time=   0.0s
[CV 2/5] END max_depth=4, min_impurity_decrease=1, min_samples_leaf=9;, score=0.819 total time=   0.0s
[CV 3/5] END max_depth=4, min_impurity_decrease=1, min_samples_leaf=9;, score=0.846 total time=   0.0s
[CV 4/5] END max_depth=4, min_impurity_decrease=1, min_samples_leaf=9;, score=0.812 total time=   0.0s
[CV 5/5] END max_depth=4, min_impurity_decrease=1, min_samples_leaf=9;, score=0.796 total time=   0.0s
[CV 1/5] END max_depth=4, min_impurity_decrease=1, min_samples_leaf=10;, score=0.818 total time=   0.0s
[CV 2/5] END max_depth=4, min_impurity_decrease=1, min_samples_leaf=10;, score=0.819 total time=   0.0s
[CV 3/5] END max_depth=4, min_impurity_decrease=1, min_samples_leaf=10;, score=0.846 total time=   0.0s
[CV 4/5] END max_depth=4, min_impurity_decrease=1, min_samples_leaf=10;, score=0.815 total time=   0.0s
[CV 5/5] END max_depth=4, min_impurity_decrease=1, min_samples_leaf=10;, score=0.796 total time=   0.0s
[CV 1/5] END max_depth=4, min_impurity_decrease=1, min_samples_leaf=11;, score=0.818 total time=   0.0s
[CV 2/5] END max_depth=4, min_impurity_decrease=1, min_samples_leaf=11;, score=0.820 total time=   0.0s
[CV 3/5] END max_depth=4, min_impurity_decrease=1, min_samples_leaf=11;, score=0.845 total time=   0.0s
[CV 4/5] END max_depth=4, min_impurity_decrease=1, min_samples_leaf=11;, score=0.815 total time=   0.0s
[CV 5/5] END max_depth=4, min_impurity_decrease=1, min_samples_leaf=11;, score=0.796 total time=   0.0s
[CV 1/5] END max_depth=4, min_impurity_decrease=1, min_samples_leaf=12;, score=0.818 total time=   0.0s
[CV 2/5] END max_depth=4, min_impurity_decrease=1, min_samples_leaf=12;, score=0.820 total time=   0.0s
[CV 3/5] END max_depth=4, min_impurity_decrease=1, min_samples_leaf=12;, score=0.845 total time=   0.0s
[CV 4/5] END max_depth=4, min_impurity_decrease=1, min_samples_leaf=12;, score=0.815 total time=   0.0s
[CV 5/5] END max_depth=4, min_impurity_decrease=1, min_samples_leaf=12;, score=0.796 total time=   0.0s
[CV 1/5] END max_depth=4, min_impurity_decrease=2, min_samples_leaf=9;, score=0.817 total time=   0.0s
[CV 2/5] END max_depth=4, min_impurity_decrease=2, min_samples_leaf=9;, score=0.819 total time=   0.0s
[CV 3/5] END max_depth=4, min_impurity_decrease=2, min_samples_leaf=9;, score=0.846 total time=   0.0s
[CV 4/5] END max_depth=4, min_impurity_decrease=2, min_samples_leaf=9;, score=0.812 total time=   0.0s
[CV 5/5] END max_depth=4, min_impurity_decrease=2, min_samples_leaf=9;, score=0.796 total time=   0.0s
[CV 1/5] END max_depth=4, min_impurity_decrease=2, min_samples_leaf=10;, score=0.818 total time=   0.0s
[CV 2/5] END max_depth=4, min_impurity_decrease=2, min_samples_leaf=10;, score=0.819 total time=   0.0s
[CV 3/5] END max_depth=4, min_impurity_decrease=2, min_samples_leaf=10;, score=0.846 total time=   0.0s
[CV 4/5] END max_depth=4, min_impurity_decrease=2, min_samples_leaf=10;, score=0.815 total time=   0.0s
[CV 5/5] END max_depth=4, min_impurity_decrease=2, min_samples_leaf=10;, score=0.796 total time=   0.0s
[CV 1/5] END max_depth=4, min_impurity_decrease=2, min_samples_leaf=11;, score=0.818 total time=   0.0s
[CV 2/5] END max_depth=4, min_impurity_decrease=2, min_samples_leaf=11;, score=0.820 total time=   0.0s
[CV 3/5] END max_depth=4, min_impurity_decrease=2, min_samples_leaf=11;, score=0.845 total time=   0.0s
[CV 4/5] END max_depth=4, min_impurity_decrease=2, min_samples_leaf=11;, score=0.815 total time=   0.0s
[CV 5/5] END max_depth=4, min_impurity_decrease=2, min_samples_leaf=11;, score=0.796 total time=   0.0s
[CV 1/5] END max_depth=4, min_impurity_decrease=2, min_samples_leaf=12;, score=0.818 total time=   0.0s
[CV 2/5] END max_depth=4, min_impurity_decrease=2, min_samples_leaf=12;, score=0.820 total time=   0.0s
[CV 3/5] END max_depth=4, min_impurity_decrease=2, min_samples_leaf=12;, score=0.845 total time=   0.0s
[CV 4/5] END max_depth=4, min_impurity_decrease=2, min_samples_leaf=12;, score=0.815 total time=   0.0s
[CV 5/5] END max_depth=4, min_impurity_decrease=2, min_samples_leaf=12;, score=0.796 total time=   0.0s
[CV 1/5] END max_depth=5, min_impurity_decrease=0, min_samples_leaf=9;, score=0.827 total time=   0.0s
[CV 2/5] END max_depth=5, min_impurity_decrease=0, min_samples_leaf=9;, score=0.807 total time=   0.0s
[CV 3/5] END max_depth=5, min_impurity_decrease=0, min_samples_leaf=9;, score=0.853 total time=   0.0s
[CV 4/5] END max_depth=5, min_impurity_decrease=0, min_samples_leaf=9;, score=0.821 total time=   0.0s
[CV 5/5] END max_depth=5, min_impurity_decrease=0, min_samples_leaf=9;, score=0.787 total time=   0.0s
[CV 1/5] END max_depth=5, min_impurity_decrease=0, min_samples_leaf=10;, score=0.825 total time=   0.0s
[CV 2/5] END max_depth=5, min_impurity_decrease=0, min_samples_leaf=10;, score=0.810 total time=   0.0s
[CV 3/5] END max_depth=5, min_impurity_decrease=0, min_samples_leaf=10;, score=0.852 total time=   0.0s
[CV 4/5] END max_depth=5, min_impurity_decrease=0, min_samples_leaf=10;, score=0.816 total time=   0.0s
[CV 5/5] END max_depth=5, min_impurity_decrease=0, min_samples_leaf=10;, score=0.787 total time=   0.0s
[CV 1/5] END max_depth=5, min_impurity_decrease=0, min_samples_leaf=11;, score=0.826 total time=   0.0s
[CV 2/5] END max_depth=5, min_impurity_decrease=0, min_samples_leaf=11;, score=0.819 total time=   0.0s
[CV 3/5] END max_depth=5, min_impurity_decrease=0, min_samples_leaf=11;, score=0.851 total time=   0.0s
[CV 4/5] END max_depth=5, min_impurity_decrease=0, min_samples_leaf=11;, score=0.816 total time=   0.0s
[CV 5/5] END max_depth=5, min_impurity_decrease=0, min_samples_leaf=11;, score=0.787 total time=   0.0s
[CV 1/5] END max_depth=5, min_impurity_decrease=0, min_samples_leaf=12;, score=0.826 total time=   0.0s
[CV 2/5] END max_depth=5, min_impurity_decrease=0, min_samples_leaf=12;, score=0.820 total time=   0.0s
[CV 3/5] END max_depth=5, min_impurity_decrease=0, min_samples_leaf=12;, score=0.851 total time=   0.0s
[CV 4/5] END max_depth=5, min_impurity_decrease=0, min_samples_leaf=12;, score=0.816 total time=   0.0s
[CV 5/5] END max_depth=5, min_impurity_decrease=0, min_samples_leaf=12;, score=0.787 total time=   0.0s
[CV 1/5] END max_depth=5, min_impurity_decrease=1, min_samples_leaf=9;, score=0.827 total time=   0.0s
[CV 2/5] END max_depth=5, min_impurity_decrease=1, min_samples_leaf=9;, score=0.807 total time=   0.0s
[CV 3/5] END max_depth=5, min_impurity_decrease=1, min_samples_leaf=9;, score=0.853 total time=   0.0s
[CV 4/5] END max_depth=5, min_impurity_decrease=1, min_samples_leaf=9;, score=0.821 total time=   0.0s
[CV 5/5] END max_depth=5, min_impurity_decrease=1, min_samples_leaf=9;, score=0.787 total time=   0.0s
[CV 1/5] END max_depth=5, min_impurity_decrease=1, min_samples_leaf=10;, score=0.825 total time=   0.0s
[CV 2/5] END max_depth=5, min_impurity_decrease=1, min_samples_leaf=10;, score=0.810 total time=   0.0s
[CV 3/5] END max_depth=5, min_impurity_decrease=1, min_samples_leaf=10;, score=0.852 total time=   0.0s
[CV 4/5] END max_depth=5, min_impurity_decrease=1, min_samples_leaf=10;, score=0.816 total time=   0.0s
[CV 5/5] END max_depth=5, min_impurity_decrease=1, min_samples_leaf=10;, score=0.787 total time=   0.0s
[CV 1/5] END max_depth=5, min_impurity_decrease=1, min_samples_leaf=11;, score=0.826 total time=   0.0s
[CV 2/5] END max_depth=5, min_impurity_decrease=1, min_samples_leaf=11;, score=0.819 total time=   0.0s
[CV 3/5] END max_depth=5, min_impurity_decrease=1, min_samples_leaf=11;, score=0.851 total time=   0.0s
[CV 4/5] END max_depth=5, min_impurity_decrease=1, min_samples_leaf=11;, score=0.816 total time=   0.0s
[CV 5/5] END max_depth=5, min_impurity_decrease=1, min_samples_leaf=11;, score=0.787 total time=   0.0s
[CV 1/5] END max_depth=5, min_impurity_decrease=1, min_samples_leaf=12;, score=0.826 total time=   0.0s
[CV 2/5] END max_depth=5, min_impurity_decrease=1, min_samples_leaf=12;, score=0.820 total time=   0.0s
[CV 3/5] END max_depth=5, min_impurity_decrease=1, min_samples_leaf=12;, score=0.851 total time=   0.0s
[CV 4/5] END max_depth=5, min_impurity_decrease=1, min_samples_leaf=12;, score=0.816 total time=   0.0s
[CV 5/5] END max_depth=5, min_impurity_decrease=1, min_samples_leaf=12;, score=0.787 total time=   0.0s
[CV 1/5] END max_depth=5, min_impurity_decrease=2, min_samples_leaf=9;, score=0.827 total time=   0.0s
[CV 2/5] END max_depth=5, min_impurity_decrease=2, min_samples_leaf=9;, score=0.807 total time=   0.0s
[CV 3/5] END max_depth=5, min_impurity_decrease=2, min_samples_leaf=9;, score=0.853 total time=   0.0s
[CV 4/5] END max_depth=5, min_impurity_decrease=2, min_samples_leaf=9;, score=0.821 total time=   0.0s
[CV 5/5] END max_depth=5, min_impurity_decrease=2, min_samples_leaf=9;, score=0.787 total time=   0.0s
[CV 1/5] END max_depth=5, min_impurity_decrease=2, min_samples_leaf=10;, score=0.825 total time=   0.0s
[CV 2/5] END max_depth=5, min_impurity_decrease=2, min_samples_leaf=10;, score=0.810 total time=   0.0s
[CV 3/5] END max_depth=5, min_impurity_decrease=2, min_samples_leaf=10;, score=0.852 total time=   0.0s
[CV 4/5] END max_depth=5, min_impurity_decrease=2, min_samples_leaf=10;, score=0.816 total time=   0.0s
[CV 5/5] END max_depth=5, min_impurity_decrease=2, min_samples_leaf=10;, score=0.787 total time=   0.0s
[CV 1/5] END max_depth=5, min_impurity_decrease=2, min_samples_leaf=11;, score=0.826 total time=   0.0s
[CV 2/5] END max_depth=5, min_impurity_decrease=2, min_samples_leaf=11;, score=0.819 total time=   0.0s
[CV 3/5] END max_depth=5, min_impurity_decrease=2, min_samples_leaf=11;, score=0.851 total time=   0.0s
[CV 4/5] END max_depth=5, min_impurity_decrease=2, min_samples_leaf=11;, score=0.816 total time=   0.0s
[CV 5/5] END max_depth=5, min_impurity_decrease=2, min_samples_leaf=11;, score=0.787 total time=   0.0s
[CV 1/5] END max_depth=5, min_impurity_decrease=2, min_samples_leaf=12;, score=0.826 total time=   0.0s
[CV 2/5] END max_depth=5, min_impurity_decrease=2, min_samples_leaf=12;, score=0.820 total time=   0.0s
[CV 3/5] END max_depth=5, min_impurity_decrease=2, min_samples_leaf=12;, score=0.851 total time=   0.0s
[CV 4/5] END max_depth=5, min_impurity_decrease=2, min_samples_leaf=12;, score=0.816 total time=   0.0s
[CV 5/5] END max_depth=5, min_impurity_decrease=2, min_samples_leaf=12;, score=0.787 total time=   0.0s
[CV 1/5] END max_depth=6, min_impurity_decrease=0, min_samples_leaf=9;, score=0.813 total time=   0.0s
[CV 2/5] END max_depth=6, min_impurity_decrease=0, min_samples_leaf=9;, score=0.795 total time=   0.0s
[CV 3/5] END max_depth=6, min_impurity_decrease=0, min_samples_leaf=9;, score=0.854 total time=   0.0s
[CV 4/5] END max_depth=6, min_impurity_decrease=0, min_samples_leaf=9;, score=0.814 total time=   0.0s
[CV 5/5] END max_depth=6, min_impurity_decrease=0, min_samples_leaf=9;, score=0.788 total time=   0.0s
[CV 1/5] END max_depth=6, min_impurity_decrease=0, min_samples_leaf=10;, score=0.818 total time=   0.0s
[CV 2/5] END max_depth=6, min_impurity_decrease=0, min_samples_leaf=10;, score=0.801 total time=   0.0s
[CV 3/5] END max_depth=6, min_impurity_decrease=0, min_samples_leaf=10;, score=0.852 total time=   0.0s
[CV 4/5] END max_depth=6, min_impurity_decrease=0, min_samples_leaf=10;, score=0.808 total time=   0.0s
[CV 5/5] END max_depth=6, min_impurity_decrease=0, min_samples_leaf=10;, score=0.788 total time=   0.0s
[CV 1/5] END max_depth=6, min_impurity_decrease=0, min_samples_leaf=11;, score=0.820 total time=   0.0s
[CV 2/5] END max_depth=6, min_impurity_decrease=0, min_samples_leaf=11;, score=0.809 total time=   0.0s
[CV 3/5] END max_depth=6, min_impurity_decrease=0, min_samples_leaf=11;, score=0.849 total time=   0.0s
[CV 4/5] END max_depth=6, min_impurity_decrease=0, min_samples_leaf=11;, score=0.807 total time=   0.0s
[CV 5/5] END max_depth=6, min_impurity_decrease=0, min_samples_leaf=11;, score=0.788 total time=   0.0s
[CV 1/5] END max_depth=6, min_impurity_decrease=0, min_samples_leaf=12;, score=0.821 total time=   0.0s
[CV 2/5] END max_depth=6, min_impurity_decrease=0, min_samples_leaf=12;, score=0.814 total time=   0.0s
[CV 3/5] END max_depth=6, min_impurity_decrease=0, min_samples_leaf=12;, score=0.849 total time=   0.0s
[CV 4/5] END max_depth=6, min_impurity_decrease=0, min_samples_leaf=12;, score=0.808 total time=   0.0s
[CV 5/5] END max_depth=6, min_impurity_decrease=0, min_samples_leaf=12;, score=0.788 total time=   0.0s
[CV 1/5] END max_depth=6, min_impurity_decrease=1, min_samples_leaf=9;, score=0.813 total time=   0.0s
[CV 2/5] END max_depth=6, min_impurity_decrease=1, min_samples_leaf=9;, score=0.795 total time=   0.0s
[CV 3/5] END max_depth=6, min_impurity_decrease=1, min_samples_leaf=9;, score=0.854 total time=   0.0s
[CV 4/5] END max_depth=6, min_impurity_decrease=1, min_samples_leaf=9;, score=0.814 total time=   0.0s
[CV 5/5] END max_depth=6, min_impurity_decrease=1, min_samples_leaf=9;, score=0.788 total time=   0.0s
[CV 1/5] END max_depth=6, min_impurity_decrease=1, min_samples_leaf=10;, score=0.818 total time=   0.0s
[CV 2/5] END max_depth=6, min_impurity_decrease=1, min_samples_leaf=10;, score=0.801 total time=   0.0s
[CV 3/5] END max_depth=6, min_impurity_decrease=1, min_samples_leaf=10;, score=0.852 total time=   0.0s
[CV 4/5] END max_depth=6, min_impurity_decrease=1, min_samples_leaf=10;, score=0.808 total time=   0.0s
[CV 5/5] END max_depth=6, min_impurity_decrease=1, min_samples_leaf=10;, score=0.788 total time=   0.0s
[CV 1/5] END max_depth=6, min_impurity_decrease=1, min_samples_leaf=11;, score=0.820 total time=   0.0s
[CV 2/5] END max_depth=6, min_impurity_decrease=1, min_samples_leaf=11;, score=0.809 total time=   0.0s
[CV 3/5] END max_depth=6, min_impurity_decrease=1, min_samples_leaf=11;, score=0.849 total time=   0.0s
[CV 4/5] END max_depth=6, min_impurity_decrease=1, min_samples_leaf=11;, score=0.807 total time=   0.0s
[CV 5/5] END max_depth=6, min_impurity_decrease=1, min_samples_leaf=11;, score=0.788 total time=   0.0s
[CV 1/5] END max_depth=6, min_impurity_decrease=1, min_samples_leaf=12;, score=0.821 total time=   0.0s
[CV 2/5] END max_depth=6, min_impurity_decrease=1, min_samples_leaf=12;, score=0.814 total time=   0.0s
[CV 3/5] END max_depth=6, min_impurity_decrease=1, min_samples_leaf=12;, score=0.849 total time=   0.0s
[CV 4/5] END max_depth=6, min_impurity_decrease=1, min_samples_leaf=12;, score=0.808 total time=   0.0s
[CV 5/5] END max_depth=6, min_impurity_decrease=1, min_samples_leaf=12;, score=0.788 total time=   0.0s
[CV 1/5] END max_depth=6, min_impurity_decrease=2, min_samples_leaf=9;, score=0.813 total time=   0.0s
[CV 2/5] END max_depth=6, min_impurity_decrease=2, min_samples_leaf=9;, score=0.795 total time=   0.0s
[CV 3/5] END max_depth=6, min_impurity_decrease=2, min_samples_leaf=9;, score=0.854 total time=   0.0s
[CV 4/5] END max_depth=6, min_impurity_decrease=2, min_samples_leaf=9;, score=0.814 total time=   0.0s
[CV 5/5] END max_depth=6, min_impurity_decrease=2, min_samples_leaf=9;, score=0.788 total time=   0.0s
[CV 1/5] END max_depth=6, min_impurity_decrease=2, min_samples_leaf=10;, score=0.818 total time=   0.0s
[CV 2/5] END max_depth=6, min_impurity_decrease=2, min_samples_leaf=10;, score=0.801 total time=   0.0s
[CV 3/5] END max_depth=6, min_impurity_decrease=2, min_samples_leaf=10;, score=0.852 total time=   0.0s
[CV 4/5] END max_depth=6, min_impurity_decrease=2, min_samples_leaf=10;, score=0.808 total time=   0.0s
[CV 5/5] END max_depth=6, min_impurity_decrease=2, min_samples_leaf=10;, score=0.788 total time=   0.0s
[CV 1/5] END max_depth=6, min_impurity_decrease=2, min_samples_leaf=11;, score=0.820 total time=   0.0s
[CV 2/5] END max_depth=6, min_impurity_decrease=2, min_samples_leaf=11;, score=0.809 total time=   0.0s
[CV 3/5] END max_depth=6, min_impurity_decrease=2, min_samples_leaf=11;, score=0.849 total time=   0.0s
[CV 4/5] END max_depth=6, min_impurity_decrease=2, min_samples_leaf=11;, score=0.807 total time=   0.0s
[CV 5/5] END max_depth=6, min_impurity_decrease=2, min_samples_leaf=11;, score=0.788 total time=   0.0s
[CV 1/5] END max_depth=6, min_impurity_decrease=2, min_samples_leaf=12;, score=0.821 total time=   0.0s
[CV 2/5] END max_depth=6, min_impurity_decrease=2, min_samples_leaf=12;, score=0.814 total time=   0.0s
[CV 3/5] END max_depth=6, min_impurity_decrease=2, min_samples_leaf=12;, score=0.849 total time=   0.0s
[CV 4/5] END max_depth=6, min_impurity_decrease=2, min_samples_leaf=12;, score=0.808 total time=   0.0s
[CV 5/5] END max_depth=6, min_impurity_decrease=2, min_samples_leaf=12;, score=0.788 total time=   0.0s

*** DecisionTreeRegressor Best Parameters ***
ccp_alpha: 0.0
criterion: mse
max_depth: 5
max_features: None
max_leaf_nodes: None
min_impurity_decrease: 0
min_impurity_split: None
min_samples_leaf: 12
min_samples_split: 2
min_weight_fraction_leaf: 0.0
random_state: 0
splitter: best

 DecisionTreeRegressor train score 0.856, DecisionTreeRegressor test score: 0.880
RandomForest models:
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
