# Loan Risk Assessment

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

sns.set(color_codes=True)
% matplotlib inline

df = pd.read_csv("challengeTrain.csv", skipinitialspace=True)
print(df.shape)
df.head(3)

df_test = pd.read_csv("challengeTest.csv", skipinitialspace=True)
print(df_test.shape)

df.head(3)

df.info()

## Remove duplicated rows

print("Number of duplicated rows:", df.duplicated().sum())
print("--------------------------------------")

df.drop_duplicates(keep='first', inplace=True)
print("Cleaned data shape:", df.shape)

### NA count per column

null_columns = df.columns[df.isnull().any()]
df[null_columns].isnull().sum()

# Univariate plotting

df.select_dtypes(include=[np.number]).hist(figsize=(15, 15), layout=(4, 4))
plt.show()

df = df.drop(['indCreditBureau', 'indSimin', 'indXlist', 'numMortgages', 'sumExternalDefault'], axis=1)
print(df.shape)
df.head()

df_test = df_test.drop(['indCreditBureau', 'indSimin', 'indXlist', 'numMortgages', 'sumExternalDefault'], axis=1)

#### Although, numLoans column has the high variablility it will be deleted because half of it is missing and prediciting half of the data isn't a wise move

print(df.numLoans.value_counts())

df = df.drop(["numLoans"], axis=1)
df_test = df_test.drop(["numLoans"], axis=1)

< br >
## <span style="color:blue"><i>customerID</i> column</span>

### Handle duplications in <span style="color:blue"><i>customerID</i> column</span>

print("Unique customerID:", len(df["customerID"].unique()))
print("Duplicated customerID:", df["customerID"].duplicated().sum())

### Create <span style="color:blue"><i>custID_dup</i> column</span> to categorize the <i>customerID</i> by the number of duplications

df['custID_dup'] = df.customerID.map(df.customerID.value_counts())
df["custID_dup"].value_counts()

#  Display cutomerID that are duplicated 4 times

df.loc[df['custID_dup'] == 4]

# The above rows -as an exmaple - shows identical rows with differences in only one column.

# Duplications will be removed further more by ignoring variations in the < b > < i > channel < / i > < / b > column, 
# since - due to miscommunications - the same customer could have been contaced by multiple persons

grouped = df.columns.drop("channel")
df.drop_duplicates(subset=grouped, inplace=True)
df = df.reset_index(drop=True)
df["custID_dup"].value_counts()

df.loc[df['custID_dup'] == 4]

#### Deleting the duplicated rows

df = df.drop(df.index[[74114, 504180]])
df = df.reset_index(drop=True)

#### Deleting the duplicated rows for [custID_dup = 6]

df.loc[df['custID_dup'] == 6]

df = df.drop(df.index[[236404, 279276, 417934]])
df = df.reset_index(drop=True)

#### Deleting sample from the duplicated rows for [custID_dup = 3]

df.loc[df['custID_dup'] == 3]

df.loc[df['customerID'] == "DR_00040163862"]

df = df.drop(df.index[[36894, 452258]])
df = df.reset_index(drop=True)

df = df.drop(["custID_dup"], axis=1)

< br >
# <span style="color:blue"><i>sex</i></span> column

## <span style="color:blue"><i>sex</i> column</span> convert into one-hot encoding

df.sex.value_counts()

df2 = pd.get_dummies(df.sex, prefix="sex", prefix_sep="_")

df = pd.concat([df, df2], axis=1, )

del (df2)

df = df.reindex_axis(['customerID'] + list(df.columns[-2:]) + list(df.columns[2:12]), axis=1)

df.columns

df2_test = pd.get_dummies(df_test.sex, prefix="sex", prefix_sep="_")

df_test = pd.concat([df_test, df2_test], axis=1, )

del (df2_test)

df_test = df_test.reindex_axis(['customerID'] + list(df_test.columns[-2:]) + list(df_test.columns[2:11]), axis=1)

df_test.columns

< br >
## <span style="color:blue"><i>status</i> column</span>

### <span style="color:blue"><i>status</i> column</span> convert missing values to NA

df.status.value_counts()

df["status"] = df["status"].replace('Unknown', np.NaN)

df.status.value_counts()

df2 = pd.get_dummies(df.status, prefix="status", prefix_sep="_")

df = pd.concat([df, df2], axis=1, )

df = df.drop(["status"], axis=1)

df_test["status"] = df_test["status"].replace('Unknown', np.NaN)

df_test.status.value_counts()

df2_test = pd.get_dummies(df_test.status, prefix="status", prefix_sep="_")

df_test = pd.concat([df_test, df2_test], axis=1, )

df_test = df_test.drop(["status"], axis=1)

< br >
# <span style="color:blue"><i>salary</i></span> column

### <span style="color:blue"><i>salary</i> column</span> convert missing values to NA

df.salary.value_counts()

df["salary"] = df["salary"].replace('None', np.NaN)
df["salary"] = df["salary"].replace('Unknown', np.NaN)

df.salary.value_counts()

df_test["salary"] = df_test["salary"].replace('None', np.NaN)
df_test["salary"] = df_test["salary"].replace('Unknown', np.NaN)

## <span style="color:blue"><i>salary</i> column</span> convert to ordinal

# Define category labels
df["salary"].unique()

# categories are categorized based on their priorioty

cat_list = {"salary": {"<650": 0,
                       "[650,1000)": 1,
                       "[1000,1300)": 2,
                       "[1300,1500)": 3,
                       "[1500,2000)": 4,
                       "[2000,3000)": 5,
                       "[3000,5000)": 6,
                       "[5000,8000)": 7,
                       ">8000": 8
                       }
            }

df.replace(cat_list, inplace=True)
df.salary.value_counts()

df_test.replace(cat_list, inplace=True)

df.salary.isnull().sum(axis=0)

< br >
# <span style="color:blue"><i>age</i></span> column

## <span style="color:blue"><i>age</i> column</span> compute missing values

# Since it's normal to increase have higher salary with older age, a boxplot will be plotted to see variance of age with salary.
# It's apparent that the mean age of each salary group increases with time.

plt.subplots(figsize=(15, 10))
sns.boxplot(x='salary', y='age', data=df)

# The missing NA will be replaced with median(because there are a lot of outliers in salaries, especially in the lower salaries)

df.loc[df.age.isnull(), 'age'] = df.groupby('salary').age.transform('median')

df_test.loc[df_test.age.isnull(), 'age'] = df.groupby('salary').age.transform('median')

null_columns = df.columns[df.isnull().any()]
df[null_columns].isnull().sum()

df[df.age.isnull()]

df['NA_row_count'] = df.apply(lambda x: x.isnull().sum(), axis=1)
df["NA_row_count"].value_counts()

df.loc[df["NA_row_count"] == 4]

df = df.drop(df[df["NA_row_count"] == 4].index)

df.loc[df["NA_row_count"] == 3]

df = df.drop(df[df["NA_row_count"] == 3].index)

del (df["NA_row_count"])

< br >
## <span style="color:blue"><i>channel</i> column</span>

### <span style="color:blue"><i>channel</i> column</span> convert missing values to NA

df.channel.value_counts()

df["channel"] = df["channel"].replace('Unknown', np.NaN)

df.channel.value_counts()

df_test["channel"] = df_test["channel"].replace('Unknown', np.NaN)

df2 = pd.get_dummies(df.channel, prefix="channel", prefix_sep="_")

df = pd.concat([df, df2], axis=1, )

df = df.drop(["channel"], axis=1)

df2_test = pd.get_dummies(df_test.channel, prefix="channel", prefix_sep="_")

df_test = pd.concat([df_test, df2_test], axis=1, )

df_test = df_test.drop(["channel"], axis=1)

< br >
## <span style="color:blue"><i>previous</i> </span>column

### <span style="color:blue"><i>previous</i> column</span> convert missing values to NA

df.previous.value_counts()

# categories are categorized based on their priorioty

cat_list = {"previous": {"Unpaid": 0,
                         "Restructuring": 1,
                         "Refinancing": 2,
                         "Default": 3,
                         "Normal": 4
                         }
            }

df.replace(cat_list, inplace=True)
df.previous.value_counts()

df_test.replace(cat_list, inplace=True)

## Rearranging target column

df2 = df["target"]

df = df.drop(["target"], axis=1)

df = pd.concat([df, df2], axis=1, )

# Modeling

df2 = df.copy()
df2_test = df_test.copy()

df2 = df2.drop("customerID", axis=1)
df2_test = df2_test.drop("customerID", axis=1)

df2.isnull().sum()

df2 = df2.dropna(axis=0, how="any")

df2_test.isnull().sum()

median_value = df2_test['age'].median()
df2_test['age'] = df2_test['age'].fillna(median_value)

median_value = df2_test['externalScore'].median()
df2_test['externalScore'] = df2_test['externalScore'].fillna(median_value)

median_value = df2_test['salary'].median()
df2_test['salary'] = df2_test['salary'].fillna(median_value)

X = df2.drop('target', axis=1)
Y = df2.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

### Logistic Regression

glmMod = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
                            intercept_scaling=1, class_weight=None,
                            random_state=None, solver='liblinear', max_iter=100,
                            multi_class='ovr', verbose=2)

glmMod.fit(X_train, Y_train)

glmMod.score(X_test, Y_test)

### Gradient Boosting

gbMod = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=200, subsample=1.0,
                                   min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                   max_depth=3,
                                   init=None, random_state=None, max_features=None, verbose=0)

gbMod.fit(X_train, Y_train)

gbMod.score(X_test, Y_test)

test_labels = gbMod.predict_proba(np.array(X_test.values))[:, 1]

roc_auc_score(Y_test, test_labels, average='macro', sample_weight=None)

# Test Set

predictions = gbMod.predict_proba(df2_test)
predictions

df2_test.shape

predictions = pd.DataFrame(predictions, columns=['Prediction_0', 'prediction_1'])

predictions.head()

# We drop the probability of 0 and keep the probability of 1 (our target)
predictions = predictions.drop('Prediction_0', axis=1)

# We concatenate the original test dataset with the prediction of someone defaulting
submission = pd.concat([df_test, predictions], axis=1)

submission.to_csv('submission.csv', index=False)

