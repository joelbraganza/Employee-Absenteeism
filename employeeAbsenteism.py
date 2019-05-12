#!python3
# employeeAbsenteism.py

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import os, random,sys, math, pprint
from scipy.stats import chi2_contingency # for correlation between categorical variables.
from scipy import stats # for one-way ANOVA test
from fancyimpute import KNN
import statsmodels.formula.api as smf # to check the relationship between 1 categorical and 1 continuous variable.
                                      # source: (https://analyticsdefined.com/anova-test-part-1/)
from statsmodels.formula.api import ols # https://www.marsja.se/four-ways-to-conduct-one-way-anovas-using-python/
import statsmodels.api as sm

from sklearn.model_selection import train_test_split # for sampling the whole dataset into training and testing
from sklearn.tree import DecisionTreeRegressor #library for Decision Tree regression
from sklearn.ensemble import RandomForestRegressor # library for random forest regression
from sklearn.linear_model import LinearRegression # library for Linear Regression

from sklearn.metrics import mean_squared_error,r2_score # using error metrics for checking errors


                                      


df = pd.read_excel(r'C:\JOEL\important-PDFs\edwisor\Project1_EmployeeAbsenteism\Absenteeism_at_work_Project.xls')
print (df.shape)

target = 'Absenteeism time in hours'
numerical_list = ['Age', 'Work load Average/day ','Transportation expense', 'Distance from Residence to Work' ,'Service time', 
                   'Hit target', 'Weight', 'Height', 'Body mass index']
categorical_list =['ID', 'Reason for absence', 'Month of absence', 'Day of the week', 'Seasons','Disciplinary failure',
                   'Education', 'Social drinker', 'Social smoker','Son', 'Pet']
columns_list = list(df.columns)



# distribution graphs code, used for column name in the continuous-variables. 
sns.distplot(column_name,fit=norm,norm_hist=True,color='blue',kde_kws={'label':'KDE') plots both KDE and distribution curve
plt.axvline(meanVal, color = 'r', linestyle = 'dashed', linewidth = 2) plots the mean line of the graph vertically in the histogram.
plt.title('column_name')

# creating a missing-value percentage dataframe function to print out the % of missing values.
def missing(dataframe):
    df_missing = pd.DataFrame(dataframe.isnull().sum())
    df_missing = df_missing.reset_index()
    df_missing = df_missing.rename(columns = {'index':'variables',0:'missing_percentage'})
    df_missing['missing_percentage'] = (df_missing['missing_percentage']*100)/len(dataframe)
    df_missing = df_missing.sort_values('missing_percentage',ascending = False).reset_index(drop=True)
    return (df_missing)

# missing-percentage BAR graph
y_pos = np.arange(len(df.columns))
plt.barh(y_pos, missing(df).missing_percentage)
plt.yticks(y_pos, missing(df).variables)
plt.title('Missing Percentage')

print ('Before preprocessing missing values:','\n')
print (missing(df))

# as 'Absenteeism time in hours' is the target-variable, we will drop the rows with null-values in this variable.
df = df.drop(df[df['Absenteeism time in hours'].isnull()].index, axis=0)

# Below code is used for making the density and distribution graph for every column of continuous variables, same is used after preprocessing
# sns.distplot(column,fit=norm,norm_hist=True,color='blue',kde_kws={'label':'KDE') plots both KDE and distribution curve
# plt.axvline(meanVal, color = 'r', linestyle = 'dashed', linewidth = 2) plots the mean line of the graph vertically in the histogram.
# plt.title('column_name')

# Below code is for making the boxplot of all the continuous variables, same is used after preprocessing.
# df.boxplot(column=column_name, figsize=(2,4))

df2 = df.copy() # creating a copy of dataframe on which to perform all the preprocessing and models, so incase anything goes wrong we have the original.


# an imputation loop which does the mean/median/KNN impuation automatically by random selecting
# a value in every variable and selecting the best imputation for that variable.

for col in numerical_list:
    variable = list(df2[col])
    variable = [str(j) for j in variable]
    variable = [None if j=='nan' else float(j) for j in variable]
    indices = [j for j in range(len(variable)) if variable[j]!=None] # made a list of non-null valued indices of the column.
    randomIndex = random.choice(indices)
    print ('random index: ',randomIndex)
    actualVal = df2[col].iloc[randomIndex]  # storing the actual value
    df2[col].iloc[randomIndex]=np.NaN  #setting the value to null
    meanVal = df2[col].dropna().mean()
    median = df2[col].dropna().median()
    df2_knn = pd.DataFrame(KNN(k=3).fit_transform(df2),columns = df2.columns)
    #df2copy[col] = [j[0] for j in list(KNN(k = 3).fit_transform(pd.DataFrame(df2copy[col])))] # for a single column
    knn = df2_knn[col].iloc[randomIndex]
    df2[col].iloc[randomIndex] = actualVal # setting the original value back
    values = [meanVal,median,knn]
    difference = [abs(actualVal-c) for c in values]
    closest = min(difference)
    if difference.index(closest)==0:
        #df2[col]=df2[col].fillna(df[col]2.mean())
        print ('close mean for col:',col,' imputing with mean for all missing values')
        df2[col] = df2[col].fillna(round(meanVal))
    elif difference.index(closest)==1:
        #method='median'
        print ('close median for col:',col,' imputing with median for all missing values')
        df2[col] = df2[col].fillna(round(median))
    else:
        #method='knn'
        print ('close knn for col:',col,' imputing with knn for all missing values')
        df2[col]=df2_knn[col]
        #df2[col] = [j[0] for j in list(KNN(k = 3).fit_transform(pd.DataFrame(df2[col])))]





# as seen in the graph, there are no outliers in 'Weight','Distance from Residence to Work' and 'Body mass index', so we exclude these variables from
# the outlier-analysis procedure.
outlier_list = ['Age', 'Work load Average/day ','Transportation expense','Service time','Hit target', 'Height']

# doing outlier analysis and imputing outliers with null-values.
for col in outlier_list:
    i = columns_list.index(col)
    column = df2.iloc[:,i].dropna(axis=0)
    c75,c25 = np.percentile(column,[75,25])
    iqr =c75-c25
    min_value = c25 - (iqr*1.5)
    max_value = c75 + (iqr*1.5)
    print (col, 'min: ',min_value,' max:',max_value, 'iqr: ',iqr,'c75: ',c75,' c25:',c25)
    df2.loc[df2[col]<min_value,col]=np.nan
    df2.loc[df2[col]>max_value,col]=np.nan


# imputing the null-values created from the outlier-removal with knn
df2= pd.DataFrame(KNN(k=3).fit_transform(df2),columns = df2.columns) # the updated knn version in fancyimpute module.
        
# seeing if there are any missing values after imputation.
print ('after preprocessing missing values:','\n')
print (missing(df2))

# feature selection.
# Correlation-analysis for removing correlated-continuous variables.
df_corr = df2.loc[:,numerical_list].dropna()
corr = df_corr.corr()

# code for seeing the correlation plot
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), 
            cmap=sns.diverging_palette(220, 50, as_cmap=True),
            square=True, ax=ax, annot = True)
print (corr) # to see the numeric correlation values.
print ('From correlation analysis, we can see that Weight and Body mass index have a very high correlation = 0.89 ~ 0.9, which is very high, so we drop Weight variable')
df2 = pd.DataFrame(data = df2, columns = [col for col in df2.columns if col!='Weight'])
numerical_list.remove('Weight') # update the numeric-continuous list
columns_list = list(df2.columns) # update the column list




# using statsmodel-library for doing one-way ANOVA test.
# can be used only when target variable is continuous.
# compares the sample group means, sees if there is some difference in the group-means.
# the features (dependent variables) should be 2 or more categorical variables. 
for col in categorical_list:
    fStat, pVal = stats.f_oneway(df2[col], df2['Absenteeism time in hours'])
    print(col,' P-value: ',pVal,' f-statistic: ',fStat)
    if pVal<0.05:
        print ('There is a relationship between col: ',col,' and the target variable')
    else:
        print ('No relationship between col: ',col,' and the target variable')


df2.to_csv(r'C:\JOEL\important-PDFs\edwisor\Project1_EmployeeAbsenteism\ImputedPython.csv',index=False)

# histogram to show variables contributing to the absence of employees.
df2.hist(column='ID',bins = len(set(df2['ID'].values)),)
plt.xticks(ticks = np.arange(1,37),labels = list(set(df['ID'].values)))

df2.hist(column='Social smoker',bins = len(set(df2['Social smoker'].values)),)
plt.xticks(ticks = np.arange(0,2),labels = list(set(df4['Social smoker'].values)))
plt.title('Social smoker')

df2.hist(column='Month of absence',bins = len(set(df2['Month of absence'].values)),)
plt.xticks(ticks = np.arange(0,2),labels = list(set(df2['Month of absence'].values)))
plt.title('Month of absence')

# same code for histogram of the rest of the variables, column = column_name of variable, ticks = (startValue, len(column)), title=column_name.             

# normalizing the data for stopping dominance of any one variable over the other in the model.
for col in numerical_list:
    if col!='Absenteeism time in hours':
        i = columns_list.index(col)
        df2.iloc[:,i] = (df2.iloc[:,i]-min(df2.iloc[:,i]))/(max(df2.iloc[:,i])-min(df2.iloc[:,i]))


print (df2.head())

def MAPE(y_true,preds):
    values=[]
    for (x,y) in zip(list(y_true),list(preds)):
        if x==0:
            values.append(abs(y)) # otherwise you get ZeroDivisionError: as 0-y/0 is not defined
        else:    
            values.append((abs(x-y)/x))
    return (sum(values)/len(values))*100    
    #mape = np.mean(np.abs((y_true-preds)/y_true))*100
    #return mape



# beginning to work on models.
# making training and testing dataset to be used by all models.
X_train, X_test, Y_train, Y_test = train_test_split( df2.iloc[:, :len(df2.columns)], df2['Absenteeism time in hours'], test_size = 0.20)


# Working Decision-tree regression model
# preparing the model using training data.
c50_model = DecisionTreeRegressor(max_depth = 4).fit(X_train,Y_train)

# to check for over fitting for training data, we calculate RMSE.
train_preds = c50_model.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(Y_train,train_preds))

# to check accuracy of the model, we calculate RMSE for test data 
testPreds = c50_model.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(Y_test,testPreds))

print ('Decision-tree Regressor (error-metrics):')
print("RMSE training data = "+str(rmse_train))
print("RMSE testing data = "+str(rmse_test))
print ('Mean Absolute Percentage Error: ',MAPE(Y_test,testPreds))
print("R-square Score(coefficient of determination) = "+str(r2_score(Y_test,testPreds)))
print ('\n\n')




# preparing the model using training data.
RF_model = RandomForestRegressor(n_estimators = 500).fit(X_train,Y_train)

# Calculating RMSE for training data to check for over fitting
RF_train_preds = RF_model.predict(X_train)
RF_RMSE_train = np.sqrt(mean_squared_error(Y_train,RF_train_preds))

# Calculating RMSE for test data to check accuracy
RF_test_preds = RF_model.predict(X_test)
RF_RMSE_test =np.sqrt(mean_squared_error(Y_test,RF_test_preds))

print ('Random-forest Regressor (error-metrics):')
print("RMSE training data ="+str(RF_RMSE_train))
print("RMSE testing data = "+str(RF_RMSE_test))
print ('Mean Absolute Percentage Error: ',MAPE(Y_test,RF_test_preds))
print("R-square Score(coefficient of determination) = "+str(r2_score(Y_test,RF_test_preds)))
print ('\n\n')


# Linear Regression
# preparing the model over training data.
LR_model = LinearRegression().fit(X_train , Y_train)

#  RMSE of training data to check if there is over fitting
LR_train_preds = LR_model.predict(X_train)
LR_RMSE_train  = np.sqrt(mean_squared_error(Y_train,LR_train_preds))

#  RMSE for test data to check accuracy of model.
LR_test_preds = LR_model.predict(X_test)
LR_RMSE_test  = np.sqrt(mean_squared_error(Y_test,LR_test_preds))

print ('Linear Regression (error-metrics):')
print("Root Mean Squared Error For Training data = "+str(LR_RMSE_train))
print("Root Mean Squared Error For Test data = "+str(LR_RMSE_test))
print ('Mean Absolute Percentage Error: ',MAPE(Y_test,LR_test_preds))
print("R^2 Score(coefficient of determination) = "+str(r2_score(Y_test,LR_test_preds)))
print ('\n\n')


#after PCA*************************************************************************************************************************************************************
print ('***********************************************************************************************************************************')
print ('Creating dummies for categorical-variables and doing Principal-Component-Analysis')
df4 = pd.get_dummies(data = df2, columns = categorical_list)

print ('no. of variables now: ',len(df4.columns)) # has 117 variables

from sklearn.decomposition import PCA

# Converting data to numpy array
X = df4.values

# Data has 117 variables so no of components of PCA = 117
pca = PCA(n_components=117)
pca.fit(X)

# The amount of variance that each PC explains
var= pca.explained_variance_ratio_

# Cumulative Variance explains
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

plt.plot(var1)
plt.xticks(ticks = np.arange(0,120,5))
plt.xlabel('No. of variables/Features')
plt.ylabel('Proportion of variance explained for the target variable')
plt.title('PCA')
plt.show()
# from the graph we can see that, at 45 clusters, 99.5% of the variance was explained, at 64 variables, 100% was explained.
# so we go for 45 variables, which explains pretty much of the variance, rather than all 64
pca = PCA(n_components=45)

# Fitting the selected components to the data
pca.fit(X)

# Using train_test_split sampling function for test and train data split
X_train, X_test, Y_train, Y_test= train_test_split(X, df4['Absenteeism time in hours'], test_size=0.2)

#X_train, X_test, Y_train, Y_test = train_test_split( df4.iloc[:, :len(df4.columns)], df4['Absenteeism time in hours'], test_size = 0.20)


# Working Decision-tree regression model
# preparing the model using training data.
c50_model = DecisionTreeRegressor(max_depth = 4).fit(X_train,Y_train)

# to check for over fitting for training data, we calculate RMSE.
train_preds = c50_model.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(Y_train,train_preds))

# to check accuracy of the model, we calculate RMSE for test data 
testPreds = c50_model.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(Y_test,testPreds))

print ('Decision-tree Regressor (error-metrics):')
print("RMSE training data = "+str(rmse_train))
print("RMSE testing data = "+str(rmse_test))
print ('Mean Absolute Percentage Error: ',MAPE(Y_test,testPreds))
print("R-square Score(coefficient of determination) = "+str(r2_score(Y_test,testPreds)))
print ('\n\n')




# preparing the model using training data.
RF_model = RandomForestRegressor(n_estimators = 500).fit(X_train,Y_train)

# Calculating RMSE for training data to check for over fitting
RF_train_preds = RF_model.predict(X_train)
RF_RMSE_train = np.sqrt(mean_squared_error(Y_train,RF_train_preds))

# Calculating RMSE for test data to check accuracy
RF_test_preds = RF_model.predict(X_test)
RF_RMSE_test =np.sqrt(mean_squared_error(Y_test,RF_test_preds))

print ('Random-forest Regressor (error-metrics):')
print("RMSE training data ="+str(RF_RMSE_train))
print("RMSE testing data = "+str(RF_RMSE_test))
print ('Mean Absolute Percentage Error: ',MAPE(Y_test,RF_test_preds))
print("R-square Score(coefficient of determination) = "+str(r2_score(Y_test,RF_test_preds)))
print ('\n\n')


# Linear Regression
# preparing the model over training data.
LR_model = LinearRegression().fit(X_train , Y_train)

#  RMSE of training data to check if there is over fitting
LR_train_preds = LR_model.predict(X_train)
LR_RMSE_train  = np.sqrt(mean_squared_error(Y_train,LR_train_preds))

#  RMSE for test data to check accuracy of model.
LR_test_preds = LR_model.predict(X_test)
LR_RMSE_test  = np.sqrt(mean_squared_error(Y_test,LR_test_preds))

print ('Linear Regression (error-metrics):')
print("Root Mean Squared Error For Training data = "+str(LR_RMSE_train))
print("Root Mean Squared Error For Test data = "+str(LR_RMSE_test))
print ('Mean Absolute Percentage Error: ',MAPE(Y_test,LR_test_preds))
print("R^2 Score(coefficient of determination) = "+str(r2_score(Y_test,LR_test_preds)))
print ('\n\n')

print ('End of program')




