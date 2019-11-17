#https://github.com/mrdbourke/your-first-kaggle-submission/blob/master/kaggle-titanic-dataset-example-submission-workflow.ipynb
%matplotlib inline
 
import math, time, random, datetime

# Data splitting
from sklearn.model_selection import train_test_split

# Data Manipulation
import numpy as np
import pandas as pd

# Visualization 
import matplotlib.pyplot as plt
import missingno
import seaborn as sns
plt.style.use('seaborn-whitegrid')

# Preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize

# Machine learning
import catboost
from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier, Pool, cv


#load_data
train = pd.read_csv('C:/Users/chinm/Desktop/BIM/Projects/employee-attrition/EA.csv')






# # Split_data
# y=train.Attrition
# x=train.drop('Attrition',axis=1)
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#or another way to split
# x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.5,test_size=0.5,# random_state=123)

# x_train.describe()
# x_test.describe()
# y_train.describe()
# y_test.describe()

missingno.matrix(train, figsize = (30,10))
# plt.show()
train.isnull().sum()

#To perform our data analysis, let's create two new dataframes
df_bin = pd.DataFrame() # for discretised continuous variables eg: age in google forms of ABR if 19-25 you represent all by 1 etc.
df_con = pd.DataFrame() # for continuous variables

# As a general rule of thumb, features with a datatype of object could be considered categorical features. And those which are floats or ints (numbers) could be considered numerical features.
# However, as we dig deeper, we might find features which are numerical may actually be categorical.
# Different data types in the dataset
train.dtypes


# How many people left?
fig = plt.figure(figsize=(20,1))
sns.countplot(y='Attrition', data=train);
print(train.Attrition.value_counts())
# plt.show()
# Yes: 237  No: 1233
# plt.show()

df_bin['Attrition'] = np.where(train['Attrition'] == 'Yes', 1, 0) # change left(yes) to 1 and retained(no) to 0
df_con['Attrition'] = np.where(train['Attrition'] == 'Yes', 1, 0) # change left(yes) to 1 and retained(no) to 0
# df_bin['Attrition'] = train['Attrition']
# df_con['Attrition'] = train['Attrition']




def plot_count_dist(data, bin_df, label_column, target_column, figsize=(20, 5), use_bin_df=False):#only for categorical numerics
    """
    Function to plot counts and distributions of a label variable and 
    target variable side by side.
    ::param_data:: = target dataframe
    ::param_bin_df:: = binned dataframe for countplot
    ::param_label_column:: = binary labelled column
    ::param_target_column:: = column you want to view counts and distributions
    ::param_figsize:: = size of figure (width, height)
    ::param_use_bin_df:: = whether or not to use the bin_df, default False
    """
    if use_bin_df: 
        fig = plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        sns.countplot(y=target_column, data=bin_df);
        plt.subplot(1, 2, 2)
        sns.distplot(data.loc[data[label_column] == "Yes"][target_column], 
                     kde_kws={"label": "Left"});
        sns.distplot(data.loc[data[label_column] == "No"][target_column], 
                     kde_kws={"label": "Retained"});
    else:
        fig = plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        sns.countplot(y=target_column, data=data);
        plt.subplot(1, 2, 2)
        sns.distplot(data.loc[data[label_column] == "Yes"][target_column], 
                     kde_kws={"label": "Left"});
        sns.distplot(data.loc[data[label_column] == "No"][target_column], 
                     kde_kws={"label": "Retained"});





#Business Travel Nominal Categorical Variable
sns.countplot(y="BusinessTravel",data=train)
# plt.show()
df_bin['BusinessTravel'] = train['BusinessTravel']
df_con['BusinessTravel'] = train['BusinessTravel']
print(train.BusinessTravel.value_counts())
# Travel_Rarely        1043
# Travel_Frequently     277
# Non-Travel            150
# sns.distplot(train.DistanceFromHome)




#DailyRate Ordinal Continuous
# sns.countplot(y="DailyRate",data=train)
sns.distplot(train.DailyRate)
# plt.show()
print(train.DailyRate.value_counts())
print("There are {} unique Ticket values.".format(len(train.DailyRate.unique())))
# Unique Length: 886
len(train)# 1470
df_con['DailyRate'] = train['DailyRate'] 
df_bin['DailyRate'] = pd.cut(train['DailyRate'], bins=6) # discretised




#Department Nominal Categorical
sns.countplot(y="Department",data=train)
# plt.show()
df_bin['Department'] = train['Department']
df_con['Department'] = train['Department']
print(train.Department.value_counts())
# Research & Development    961
# Sales                     446
# Human Resources            63



# DistanceFromHome Ordinal Continuous
sns.countplot(y="DistanceFromHome",data=train)
# plt.show()
sns.distplot(train.DistanceFromHome)
# plt.show()
print("There are {} unique Ticket values.".format(len(train.DistanceFromHome.unique())))
# Unique Length: 29
len(train)
# 1470
df_con['DistanceFromHome'] = train['DistanceFromHome'] 
df_bin['DistanceFromHome'] = pd.cut(train['DistanceFromHome'], bins=6) # discretised



# Education Ordinal Categorical
sns.countplot(y="Education",data=train)
# plt.show()
sns.distplot(train.Education)
# plt.show()
plot_count_dist(data=train,
                bin_df=train,
                label_column='Attrition', 
                target_column='Education', 
                figsize=(20,10), 
                use_bin_df=True)
# plt.show()
print(train.Education.value_counts())
# 3    572
# 4    398
# 2    282
# 1    170
# 5     48
df_bin['Education'] = train['Education']
df_con['Education'] = train['Education']




# EducationField nominal categorical
sns.countplot(y="EducationField",data=train)
# plt.show()
df_bin['EducationField'] = train['EducationField']
df_con['EducationField'] = train['EducationField']
print(train.EducationField.value_counts())
# Life Sciences       606
# Medical             464
# Marketing           159
# Technical Degree    132
# Other                82
# Human Resources      27



# EmployeeCount
print(train.EmployeeCount.value_counts())
# 1470 


# EmployeeNumber
print(train.EmployeeNumber.value_counts())
# 1470



# EnvironmentSatisfaction ordinal categorical
sns.countplot(y="EnvironmentSatisfaction",data=train)
# plt.show()
sns.distplot(train.EnvironmentSatisfaction)
# plt.show()
plot_count_dist(data=train,
                bin_df=train,
                label_column='Attrition', 
                target_column='EnvironmentSatisfaction', 
                figsize=(20,10), 
                use_bin_df=True)
# plt.show()
print(train.EnvironmentSatisfaction.value_counts())
# 3    453
# 4    446
# 2    287
# 1    284
df_bin['EnvironmentSatisfaction'] = train['EnvironmentSatisfaction']
df_con['EnvironmentSatisfaction'] = train['EnvironmentSatisfaction']




# Gender nominal categorical
df_bin['Gender'] = train['Gender']
df_bin['Gender'] = np.where(df_bin['Gender'] == 'Female', 1, 0) # change sex to 0 for male and 1 for female
df_con['Gender'] = train['Gender']
print(train.Gender.value_counts())
# Male      882
# Female    588
# fig = plt.figure(figsize=(10, 10))
sns.distplot(df_bin.loc[df_bin['Attrition'] == "Yes"]['Gender'], kde_kws={'label': 'Left'});
sns.distplot(df_bin.loc[df_bin['Attrition'] == "No"]['Gender'], kde_kws={'label': 'Retained'});



# Can ordinal be continuous?



# Hourly Rate ordinal continuous
sns.distplot(train.HourlyRate)
# sns.countplot(y="HourlyRate",data=train)
# plt.show()
print(train.HourlyRate.value_counts())
print("There are {} unique Ticket values.".format(len(train.HourlyRate.unique())))
# Unique Length: 71
len(train)# 1470
df_con['HourlyRate'] = train['HourlyRate'] 
df_bin['HourlyRate'] = pd.cut(train['HourlyRate'], bins=5) # discretised



# JobInvolvement ordinal categorical
sns.countplot(y="JobInvolvement",data=train)
# plt.show()
sns.distplot(train.JobInvolvement)
# plt.show()
plot_count_dist(data=train,
                bin_df=train,
                label_column='Attrition', 
                target_column='JobInvolvement', 
                figsize=(20,10), 
                use_bin_df=True)
# plt.show()
print(train.JobInvolvement.value_counts())
# 3    453
# 4    446
# 2    287
# 1    284
df_bin['JobInvolvement'] = train['JobInvolvement']
df_con['JobInvolvement'] = train['JobInvolvement']



# JobLevel ordinal categorical
sns.countplot(y="JobLevel",data=train)
# plt.show()
sns.distplot(train.JobLevel)
# plt.show()
plot_count_dist(data=train,
                bin_df=train,
                label_column='Attrition', 
                target_column='JobLevel', 
                figsize=(20,10), 
                use_bin_df=True)
# plt.show()
print(train.JobLevel.value_counts())
# 3    453
# 4    446
# 2    287
# 1    284
df_bin['JobLevel'] = train['JobLevel']
df_con['JobLevel'] = train['JobLevel']




# JobRole nominal categorical
sns.countplot(y="JobRole",data=train)
# plt.show()
df_bin['JobRole'] = train['JobRole']
df_con['JobRole'] = train['JobRole']
print(train.JobRole.value_counts())
# Sales Executive              326
# Research Scientist           292
# Laboratory Technician        259
# Manufacturing Director       145
# Healthcare Representative    131
# Manager                      102
# Sales Representative          83
# Research Director             80
# Human Resources               52




# JobSatisfaction ordinal categorical
sns.countplot(y="JobSatisfaction",data=train)
# plt.show()
sns.distplot(train.JobSatisfaction)
# plt.show()
plot_count_dist(data=train,
                bin_df=train,
                label_column='Attrition', 
                target_column='JobSatisfaction', 
                figsize=(20,10), 
                use_bin_df=True)
# plt.show()
print(train.JobSatisfaction.value_counts())
# 3    453
# 4    446
# 2    287
# 1    284
df_bin['JobSatisfaction'] = train['JobSatisfaction']
df_con['JobSatisfaction'] = train['JobSatisfaction']


# MaritalStatus nominal categorical
print(train.MaritalStatus.value_counts())
# Married     673
# Single      470
# Divorced    327
sns.countplot(y="MaritalStatus",data=train)
# plt.show()
df_bin['MaritalStatus'] = train['MaritalStatus']
df_con['MaritalStatus'] = train['MaritalStatus']



# MonthlyIncome ordinal continuous
sns.countplot(y="MonthlyIncome",data=train)
sns.distplot(train.MonthlyIncome)
# plt.show()
print(train.MonthlyIncome.value_counts())
print("There are {} unique Ticket values.".format(len(train.MonthlyIncome.unique())))
# Unique Length: 1349
len(train)# 1470
train.MonthlyIncome.describe()
df_con['MonthlyIncome'] = train['MonthlyIncome'] 
df_bin['MonthlyIncome'] = pd.cut(train['MonthlyIncome'], bins=5) # discretised



# NumCompaniesWorked nominal categorical
sns.countplot(y="NumCompaniesWorked",data=train)
# plt.show()
sns.distplot(train.NumCompaniesWorked)
# plt.show()
plot_count_dist(data=train,
                bin_df=train,
                label_column='Attrition', 
                target_column='NumCompaniesWorked', 
                figsize=(20,10), 
                use_bin_df=True)
# plt.show()
print(train.NumCompaniesWorked.value_counts())
# 1    521
# 0    197
# 3    159
# 2    146
# 4    139
# 7     74
# 6     70
# 5     63
# 9     52
# 8     49
df_bin['NumCompaniesWorked'] = train['NumCompaniesWorked']
df_con['NumCompaniesWorked'] = train['NumCompaniesWorked']



# Over18
print(train.Over18.value_counts())
#Y 1470



# OverTime Ordinal categorical
df_bin['OverTime'] = train['OverTime']
df_bin['OverTime'] = np.where(df_bin['OverTime'] == 'Yes', 1, 0) # change overtime to 0 for no and 1 for yes
df_con['OverTime'] = train['OverTime']
print(train.OverTime.value_counts())
# Male      882
# Female    588
# fig = plt.figure(figsize=(10, 10))
sns.distplot(df_bin.loc[df_bin['Attrition'] == "Yes"]['OverTime'], kde_kws={'label': 'Left'});
sns.distplot(df_bin.loc[df_bin['Attrition'] == "No"]['OverTime'], kde_kws={'label': 'Retained'});
# plt.show()

# PercentSalaryHike ordinal continuous
train.PercentSalaryHike.describe()
sns.distplot(train.PercentSalaryHike)
# sns.countplot(y="PercentSalaryHike",data=train)
# plt.show()
print(train.PercentSalaryHike.value_counts())
print("There are {} unique Ticket values.".format(len(train.PercentSalaryHike.unique())))
# Unique Length: 15
len(train)# 1470
df_con['PercentSalaryHike'] = train['PercentSalaryHike'] 
df_bin['PercentSalaryHike'] = pd.cut(train['PercentSalaryHike'], bins=5) # discretised



# PerformanceRating ordinal categorical
sns.countplot(y="PerformanceRating",data=train)
# plt.show()
sns.distplot(train.PerformanceRating)
# plt.show()
plot_count_dist(data=train,
                bin_df=train,
                label_column='Attrition', 
                target_column='PerformanceRating', 
                figsize=(20,10), 
                use_bin_df=True)
# plt.show()
print(train.PerformanceRating.value_counts())
# 3    1244
# 4     226
df_bin['PerformanceRating'] = train['PerformanceRating']
df_con['PerformanceRating'] = train['PerformanceRating']




# RelationshipSatisfaction ORDINAL CATEGORICAL
sns.countplot(y="RelationshipSatisfaction",data=train)
# plt.show()
sns.distplot(train.RelationshipSatisfaction)
# plt.show()
plot_count_dist(data=train,
                bin_df=train,
                label_column='Attrition', 
                target_column='RelationshipSatisfaction', 
                figsize=(20,10), 
                use_bin_df=True)
# plt.show()
print(train.RelationshipSatisfaction.value_counts())
# 3    459
# 4    432
# 2    303
# 1    276
df_bin['RelationshipSatisfaction'] = train['RelationshipSatisfaction']
df_con['RelationshipSatisfaction'] = train['RelationshipSatisfaction']






# StandardHours
print(train.StandardHours.value_counts())
# 80    1470







# StockOptionLevel ordinal categorical
sns.countplot(y="StockOptionLevel",data=train)
# plt.show()
sns.distplot(train.StockOptionLevel)
# plt.show()
plot_count_dist(data=train,
                bin_df=train,
                label_column='Attrition', 
                target_column='StockOptionLevel', 
                figsize=(20,10), 
                use_bin_df=True)
# plt.show()
print(train.StockOptionLevel.value_counts())
# 0    631
# 1    596
# 2    158
# 3     85
df_bin['StockOptionLevel'] = train['StockOptionLevel']
df_con['StockOptionLevel'] = train['StockOptionLevel']




# TotalWorkingYears ordinal continuous
train.TotalWorkingYears.describe()
sns.distplot(train.TotalWorkingYears)
# sns.countplot(y="TotalWorkingYears",data=train)
# plt.show()
print(train.TotalWorkingYears.value_counts())
print("There are {} unique Ticket values.".format(len(train.TotalWorkingYears.unique())))
# Unique Length: 15
len(train)# 1470
df_con['TotalWorkingYears'] = train['TotalWorkingYears'] 
df_bin['TotalWorkingYears'] = pd.cut(train['TotalWorkingYears'], bins=8) # discretised
# REMOVE OUTLIERS




# TrainingTimesLastYear ordinal categorical
sns.countplot(y="TrainingTimesLastYear",data=train)
# plt.show()
sns.distplot(train.TrainingTimesLastYear)
# plt.show()
plot_count_dist(data=train,
                bin_df=train,
                label_column='Attrition', 
                target_column='TrainingTimesLastYear', 
                figsize=(20,10), 
                use_bin_df=True)
# plt.show()
print(train.TrainingTimesLastYear.value_counts())
# 2    547
# 3    491
# 4    123
# 5    119
# 1     71
# 6     65
# 0     54
df_bin['TrainingTimesLastYear'] = train['TrainingTimesLastYear']
df_con['TrainingTimesLastYear'] = train['TrainingTimesLastYear']





# WorkLifeBalance ordinal categorical
sns.countplot(y="WorkLifeBalance",data=train)
# plt.show()
sns.distplot(train.WorkLifeBalance)
# plt.show()
plot_count_dist(data=train,
                bin_df=train,
                label_column='Attrition', 
                target_column='WorkLifeBalance', 
                figsize=(20,10), 
                use_bin_df=True)
# plt.show()
print(train.WorkLifeBalance.value_counts())
# 3    459
# 4    432
# 2    303
# 1    276
df_bin['WorkLifeBalance'] = train['WorkLifeBalance']
df_con['WorkLifeBalance'] = train['WorkLifeBalance']





# YearsAtCompany ordinal continuous
train.YearsAtCompany.describe()
sns.distplot(train.YearsAtCompany)
# sns.countplot(y="YearsAtCompany",data=train)
# plt.show()
print(train.YearsAtCompany.value_counts())
print("There are {} unique Ticket values.".format(len(train.YearsAtCompany.unique())))
# Unique Length: 37
len(train)# 1470
df_con['YearsAtCompany'] = train['YearsAtCompany'] 
df_bin['YearsAtCompany'] = pd.cut(train['YearsAtCompany'], bins=8) # discretised
# REMOVE OUTLIERS





# YearsInCurrentRole ordinal continuous
train.YearsInCurrentRole.describe()
sns.distplot(train.YearsInCurrentRole)
# sns.countplot(y="YearsInCurrentRole",data=train)
# plt.show()
print(train.YearsInCurrentRole.value_counts())
print("There are {} unique Ticket values.".format(len(train.YearsInCurrentRole.unique())))
# Unique Length: 19
len(train)# 1470
df_con['YearsInCurrentRole'] = train['YearsInCurrentRole'] 
df_bin['YearsInCurrentRole'] = pd.cut(train['YearsInCurrentRole'], bins=4) # discretised
# REMOVE OUTLIERS






# YearsSinceLastPromotion ordinal continuous
train.YearsSinceLastPromotion.describe()
sns.distplot(train.YearsSinceLastPromotion)
# sns.countplot(y="YearsSinceLastPromotion",data=train)
# plt.show()
print(train.YearsSinceLastPromotion.value_counts())
print("There are {} unique Ticket values.".format(len(train.YearsSinceLastPromotion.unique())))
# Unique Length: 16
len(train)# 1470
df_con['YearsSinceLastPromotion'] = train['YearsSinceLastPromotion'] 
df_bin['YearsSinceLastPromotion'] = pd.cut(train['YearsSinceLastPromotion'], bins=3) # discretised
# REMOVE OUTLIERS



# YearsWithCurrManager ordinal continuous
train.YearsWithCurrManager.describe()
sns.distplot(train.YearsWithCurrManager)
# sns.countplot(y="YearsWithCurrManager",data=train)
# plt.show()
print(train.YearsWithCurrManager.value_counts())
print("There are {} unique Ticket values.".format(len(train.YearsWithCurrManager.unique())))
# Unique Length: 18
len(train)# 1470
df_con['YearsWithCurrManager'] = train['YearsWithCurrManager'] 
df_bin['YearsWithCurrManager'] = pd.cut(train['YearsWithCurrManager'], bins=4) # discretised
# REMOVE OUTLIERS





df_bin.head()

#The line below is used to make a variable named one_hot_cols with columns of df_bin
one_hot_cols = df_bin.columns.tolist()

#Those columns are removed which are ordinal and don't have the necessity to be made one_hot because there order matters.
one_hot_cols.remove('Attrition')
one_hot_cols.remove('EnvironmentSatisfaction')
one_hot_cols.remove('JobInvolvement')
one_hot_cols.remove('JobLevel')
one_hot_cols.remove('JobSatisfaction')
one_hot_cols.remove('PerformanceRating')
one_hot_cols.remove('RelationshipSatisfaction')
one_hot_cols.remove('StockOptionLevel')
one_hot_cols.remove('WorkLifeBalance')
df_bin_enc = pd.get_dummies(df_bin, columns=one_hot_cols)
df_bin_enc.head()


one_hot_cols1 = df_con.columns.tolist()


df_BusinessTravel_one_hot = pd.get_dummies(df_con['BusinessTravel'], 
                                     prefix='businessTravel')

df_Department_one_hot = pd.get_dummies(df_con['Department'], 
                                prefix='department')

df_EducationField_one_hot = pd.get_dummies(df_con['EducationField'], 
                                   prefix='educationField')


df_Gender_one_hot = pd.get_dummies(df_con['Gender'], 
                                prefix='gender')

df_JobRole_one_hot = pd.get_dummies(df_con['JobRole'], 
                                prefix='jobRole')

df_MaritalStatus_one_hot = pd.get_dummies(df_con['MaritalStatus'], 
                                prefix='maritalStatus')

df_OverTime_one_hot = pd.get_dummies(df_con['OverTime'], 
                                prefix='overTime')



# df_BusinessTravel_one_hot,
# df_Department_one_hot,
# df_EducationField_one_hot,
# df_Gender_one_hot,
# df_JobRole_one_hot,
# df_MaritalStatus_one_hot 
# Combine the one hot encoded columns with df_con_enc
df_con_enc = pd.concat([df_con, 
						df_BusinessTravel_one_hot,
						df_Department_one_hot,
						df_EducationField_one_hot,
						df_Gender_one_hot,
						df_JobRole_one_hot,
						df_MaritalStatus_one_hot ], axis=1)
# Drop the original categorical columns (because now they've been one hot encoded)
df_con_enc = df_con_enc.drop(['Gender','EducationField', 'Department', 'OverTime','BusinessTravel','JobRole','MaritalStatus'], axis=1)
# Let's look at df_con_enc
df_con_enc.head(20)

# Step3: Start Building Machine Learning Models
# Let's seperate the data
# Seclect the dataframe we want to use first for predictions
selected_df = df_con_enc
selected_df.head()
# Split the dataframe into data and labels
X_train = selected_df.drop('Attrition', axis=1) # data
# y_train = selected_df.Attrition # labels
# Shape of the data (without labels)
X_train.shape
# Shape of the labels
y_train = selected_df.Attrition 
y_train.shape
# change left(yes) to 1 and retained(no) to 0




def fit_ml_algo(algo, X_train, y_train, cv):
    
    # One Pass
    model = algo.fit(X_train, y_train)
    acc = round(model.score(X_train, y_train) * 100, 2)
    # Cross Validation 
    train_pred = model_selection.cross_val_predict(algo, 
                                                  X_train, 
                                                  y_train, 
                                                  cv=cv, 
                                                  n_jobs = -1)
    # Cross-validation accuracy metric
    acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)
    
    return train_pred, acc, acc_cv




# model = LogisticRegression().fit(X_train, selected_df.Attrition)


# Logistic Regression
start_time = time.time()
train_pred_log, acc_log, acc_cv_log = fit_ml_algo(LogisticRegression(), X_train, y_train, 10)
log_time = (time.time() - start_time)
print("Accuracy: %s" % acc_log)
print("Accuracy CV 10-Fold: %s" % acc_cv_log)
print("Running Time: %s" % datetime.timedelta(seconds=log_time))


# k-Nearest Neighbours
start_time = time.time()
train_pred_knn, acc_knn, acc_cv_knn = fit_ml_algo(KNeighborsClassifier(), 
                                                  X_train, 
                                                  y_train, 
                                                  10)
knn_time = (time.time() - start_time)
print("Accuracy: %s" % acc_knn)
print("Accuracy CV 10-Fold: %s" % acc_cv_knn)
print("Running Time: %s" % datetime.timedelta(seconds=knn_time))



# Gaussian Naive Bayes
start_time = time.time()
train_pred_gaussian, acc_gaussian, acc_cv_gaussian = fit_ml_algo(GaussianNB(), 
                                                                      X_train, 
                                                                      y_train, 
                                                                           10)
gaussian_time = (time.time() - start_time)
print("Accuracy: %s" % acc_gaussian)
print("Accuracy CV 10-Fold: %s" % acc_cv_gaussian)
print("Running Time: %s" % datetime.timedelta(seconds=gaussian_time))




# Linear SVC
start_time = time.time()
train_pred_svc, acc_linear_svc, acc_cv_linear_svc = fit_ml_algo(LinearSVC(),
                                                                X_train, 
                                                                y_train, 
                                                                10)
linear_svc_time = (time.time() - start_time)
print("Accuracy: %s" % acc_linear_svc)
print("Accuracy CV 10-Fold: %s" % acc_cv_linear_svc)
print("Running Time: %s" % datetime.timedelta(seconds=linear_svc_time))



# Stochastic Gradient Descent
start_time = time.time()
train_pred_sgd, acc_sgd, acc_cv_sgd = fit_ml_algo(SGDClassifier(), 
                                                  X_train, 
                                                  y_train,
                                                  10)
sgd_time = (time.time() - start_time)
print("Accuracy: %s" % acc_sgd)
print("Accuracy CV 10-Fold: %s" % acc_cv_sgd)
print("Running Time: %s" % datetime.timedelta(seconds=sgd_time))



# Decision Tree Classifier
start_time = time.time()
train_pred_dt, acc_dt, acc_cv_dt = fit_ml_algo(DecisionTreeClassifier(), 
                                                                X_train, 
                                                                y_train,
                                                                10)
dt_time = (time.time() - start_time)
print("Accuracy: %s" % acc_dt)
print("Accuracy CV 10-Fold: %s" % acc_cv_dt)
print("Running Time: %s" % datetime.timedelta(seconds=dt_time))



# Gradient Boosting Trees
start_time = time.time()
train_pred_gbt, acc_gbt, acc_cv_gbt = fit_ml_algo(GradientBoostingClassifier(), 
                                                                       X_train, 
                                                                       y_train,
                                                                       10)
gbt_time = (time.time() - start_time)
print("Accuracy: %s" % acc_gbt)
print("Accuracy CV 10-Fold: %s" % acc_cv_gbt)
print("Running Time: %s" % datetime.timedelta(seconds=gbt_time))




# CatBoost Algorithm
# View the data for the CatBoost model
X_train.head()
# View the labels for the CatBoost model
y_train.head()
# Define the categorical features for the CatBoost model
cat_features = np.where(X_train.dtypes != np.float)[0]
cat_features
# Use the CatBoost Pool() function to pool together the training data and categorical feature labels
train_pool = Pool(X_train, 
                  y_train,
                  cat_features)
# CatBoost model definition
catboost_model = CatBoostClassifier(iterations=1000,
                                    custom_loss=['Accuracy'],
                                    loss_function='Logloss')

# Fit CatBoost model
catboost_model.fit(train_pool,
                   plot=False)
# CatBoost accuracy
acc_catboost = round(catboost_model.score(X_train, y_train) * 100, 2)

# Perform CatBoost cross-validation
# How long will this take?
start_time = time.time()

# Set params for cross-validation as same as initial model
cv_params = catboost_model.get_params()

# Run the cross-validation for 10-folds (same as the other models)
cv_data = cv(train_pool,
             cv_params,
             fold_count=10,
             plot=True)

# How long did it take?
catboost_time = (time.time() - start_time)

# CatBoost CV results save into a dataframe (cv_data), let's withdraw the maximum accuracy score
acc_cv_catboost = round(np.max(cv_data['test-Accuracy-mean']) * 100, 2)
# Print out the CatBoost model metrics
print("---CatBoost Metrics---")
print("Accuracy: {}".format(acc_catboost))
print("Accuracy cross-validation 10-Fold: {}".format(acc_cv_catboost))
print("Running Time: {}".format(datetime.timedelta(seconds=catboost_time)))

# Model Results
# Which model had the best cross-validation accuracy?
# Note: We care most about cross-validation metrics because the metrics we get from .fit() can randomly score higher than usual.
# Regular accuracy scores
models = pd.DataFrame({
    'Model': ['KNN', 'Logistic Regression', 'Naive Bayes', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree', 'Gradient Boosting Trees',
              'CatBoost'],
    'Score': [
        acc_knn, 
        acc_log,  
        acc_gaussian, 
        acc_sgd, 
        acc_linear_svc, 
        acc_dt,
        acc_gbt,
        acc_catboost
    ]})
print("---Reuglar Accuracy Scores---")
models.sort_values(by='Score', ascending=False)
#                         Model   Score
# 5               Decision Tree  100.00
# 6     Gradient Boosting Trees   92.31
# 7                    CatBoost   88.98
# 1         Logistic Regression   86.19
# 0                         KNN   85.92
# 4                  Linear SVC   83.95
# 3  Stochastic Gradient Decent   83.88
# 2                 Naive Bayes   69.80



cv_models = pd.DataFrame({
    'Model': ['KNN', 'Logistic Regression', 'Naive Bayes', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree', 'Gradient Boosting Trees',
              'CatBoost'],
    'Score': [
        acc_cv_knn, 
        acc_cv_log,      
        acc_cv_gaussian, 
        acc_cv_sgd, 
        acc_cv_linear_svc, 
        acc_cv_dt,
        acc_cv_gbt,
        acc_cv_catboost
    ]})
print('---Cross-validation Accuracy Scores---')
cv_models.sort_values(by='Score', ascending=False)
#                         Model  Score
# 7                    CatBoost  85.99
# 1         Logistic Regression  85.37
# 6     Gradient Boosting Trees  84.76
# 0                         KNN  81.77
# 3  Stochastic Gradient Decent  80.68
# 4                  Linear SVC  80.34
# 5               Decision Tree  77.82
# 2                 Naive Bayes  69.05