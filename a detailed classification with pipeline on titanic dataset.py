#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Ignore warnings 
import warnings
warnings.filterwarnings('ignore')


# In[4]:


# Data processing and analysis
import numpy as np
import pandas as pd
import math 
import re #it is used for reading text based on some regular expression 


# In[5]:


# Data visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno #for that we need to intall package by below code
#conda install -c conda-forge/label/gcc7 missingno    in anconda prompt 


# In[6]:


# Configure visualisations
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')
sns.set(context="notebook", palette="dark", style = 'whitegrid' , color_codes=True)


# In[7]:


# Classification algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis , QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb#  for this i have to install package  "  !pip install lightgbm "
import xgboost as xgb


# In[8]:


#!pip install lightgbm


# In[9]:


# Data preprocessing :
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, scale, LabelEncoder, OneHotEncoder


# In[10]:


# Modeling helper functions
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV , KFold , cross_val_score


# In[11]:


# Classification metrices
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report, precision_score,recall_score,f1_score 



# In[12]:


train=pd.read_csv(r"C:\Users\SAKET NANDAN\Documents\classification model with pipeline\train.csv")
test=pd.read_csv(r"C:\Users\SAKET NANDAN\Documents\classification model with pipeline\test.csv")
IDtest = test['PassengerId']


# # 3. Data Exploration

# ### 3.1 Explore the training set 

# #### View shape of training set

# In[13]:


print('The shape of the training set : {} '.format(train.shape))


# #### View profile report of training set

# In[14]:


import pandas_profiling as pp


# In[15]:


#pip install pandas-profiling  on anaconda prompt 


# In[16]:


pp.ProfileReport(train)


# #### Preview training set

# In[18]:


train.head()


# In[19]:


test.head()


# #### View concise summary of training set

# In[20]:


train.info()


# ###### It seems that several of the variables - Age, Cabin and Embarked contain missing values. Let's check it.

# ### Print variables containing missing values

# In[21]:


var1 = [col for col in train.columns if train[col].isnull().sum() != 0]

print(train[var1].isnull().sum())


# #### So, we are right that Age, Cabin and Embarked contain missing values.

# ### View statistical properties of training set

# In[22]:


train.describe()


# ### Types of Variables

# Now, we will classify the variables into categorical and numerical variables.

# In[24]:


# find categorical variables

categorical = [var for var in train.columns if train[var].dtype =='O']

print('There are {} categorical variables in training set.\n'.format(len(categorical)))

print('The categorical variables are :', categorical)


# In[25]:


# find numerical variables

numerical = [var for var in train.columns if train[var].dtype !='O']

print('There are {} numerical variables in training set.\n'.format(len(numerical)))

print('The numerical variables are :', numerical)


# ### 3.2 Explore the test set ¶

# ### View shape of test set

# In[26]:


print('The shape of the test set : {} '.format(test.shape))


# #### View profile report of test set

# In[27]:


pp.ProfileReport(test)


# ### Preview test set

# In[28]:


test.head()


# #### View concise summary of test set

# In[29]:


test.info()


# #### Print variables containing missing values

# In[30]:


var2 = [col for col in test.columns if test[col].isnull().sum() != 0]

print(test[var2].isnull().sum())


# #### So, we are right that Age, Cabin and Embarked contain missing values.

# ### View statistical properties of test set

# In[32]:


test.describe()


# # Types of Variables

# ###### Now, we will classify the variables into categorical and numerical variables.

# In[33]:


# find categorical variables

categorical = [var for var in test.columns if test[var].dtype =='O']

print('There are {} categorical variables in test set.\n'.format(len(categorical)))

print('The categorical variables are :', categorical)


# In[34]:


# find numerical variables

numerical = [var for var in test.columns if test[var].dtype !='O']

print('There are {} numerical variables in test set.\n'.format(len(numerical)))

print('The numerical variables are :', numerical)


# ### Observations about dataset

# ## 4. Data Visualization

# ### 4.1 Missing values 

# In[35]:


# view missing values in training set
msno.matrix(train, figsize = (30,10))


# so it shows that in age , cabin ,embarked has missing  values 

# # 4.2 Survived 

# In[36]:


train['Survived'].value_counts()


# ##### Here 0 stands for not survived and 1 stands for survived.
# 
# ##### So, 549 people survived and 342 people did not survive.
# 
# ##### Let's visualize it by plotting.

# In[37]:


fig, ax1 = plt.subplots(figsize=(6,6))
graph = sns.countplot(ax=ax1,x=train['Survived'], data = train, palette = 'PuBuGn_d')
graph.set_title('Distribution of people who survived', fontsize = 12)
graph.set_xticklabels(graph.get_xticklabels(),rotation=30)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# ##### Now females have higher probability of survival than males.
# ##### Let' check it

# In[38]:


train.groupby('Survived')['Sex'].value_counts()


# In[39]:


fig, ax1 = plt.subplots(figsize=(6,6))
graph = sns.countplot(ax=ax1,x=train['Survived'], data = train, hue='Sex', palette = 'PuBuGn_d')
graph.set_title('Distribution of people who survived', fontsize = 12)
graph.set_xticklabels(graph.get_xticklabels(),rotation=30)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# ##### Let's check the percentage of survival for males and females separately.

# In[40]:


females = train[train['Sex'] == 'female']
females.head()


# In[41]:


females['Survived'].value_counts()/len(females)


# In[42]:


males = train[train['Sex'] == 'male']
males.head()


# In[43]:


males['Survived'].value_counts()/len(males)


# ###### As expected females have higher probability of survival (value 1) 74.20% than males 18.89%.
# 

# In[44]:


# create the first of two pie-charts and set current axis
plt.figure(figsize=(8,6))
plt.subplot(1, 2, 1)   # (rows, columns, panel number)
labels1 = females['Survived'].value_counts().index
size1 = females['Survived'].value_counts()
colors1=['cyan','pink']
plt.pie(size1, labels = labels1, colors = colors1, shadow = True, autopct='%1.1f%%',startangle = 90)
plt.title('Percentage of females who survived', fontsize = 20)
plt.legend(['1:Survived', '0:Not Survived'], loc=0)
plt.show()

# create the second of two pie-charts and set current axis
plt.figure(figsize=(8,6))
plt.subplot(1, 2, 2)   # (rows, columns, panel number)
labels2 = males['Survived'].value_counts().index
size2 = males['Survived'].value_counts()
colors2=['pink','cyan']
plt.pie(size2, labels = labels2, colors = colors2, shadow = True, autopct='%1.1f%%',startangle = 90)
plt.title('Percentage of males who survived', fontsize = 20)
plt.legend(['0:Not Survived','1:Survived'])
plt.show()


# ###### From the above pie-charts, we can deduce that females probability of survival is 74.2% (cyan color) while males probability of survival is 18.9% (cyan color).
# 

# ## 4.3 Sex

# In[45]:


train['Sex'].value_counts()


# In[46]:


fig, ax1 = plt.subplots(figsize=(6,6))
graph = sns.countplot(ax=ax1,x=train['Sex'], data=train, palette = 'bone')
graph.set_title('Distribution of sex among passengers', fontsize = 12)
graph.set_xticklabels(graph.get_xticklabels(),rotation=30)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# In[47]:


train['Sex'].value_counts()/len(train)


# In[48]:


plt.figure(figsize=(8,6))
labels = train['Sex'].value_counts().index
size = train['Sex'].value_counts()
colors=['cyan','pink']
plt.pie(size, labels = labels, shadow = True, colors=colors, autopct='%1.1f%%',startangle = 90)
plt.title('Percentage distribution of sex among passengers', fontsize = 20)
plt.legend()
plt.show()


# In[49]:


train.groupby('Pclass')['Sex'].value_counts()


# ### 4.4 Pclass 

# In[50]:


fig, ax = plt.subplots(figsize=(8,6))
graph = sns.countplot(ax=ax,x=train['Pclass'], data=train, palette = 'bone')
graph.set_title('Number of people in different classes', fontsize = 12)
graph.set_xticklabels(graph.get_xticklabels(),rotation=30)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# In[51]:


fig, ax = plt.subplots(figsize=(8,6))
graph = sns.countplot(ax=ax,x=train['Pclass'], data=train, hue='Survived', palette = 'bone')
graph.set_title('Distribution of people segregated by survival', fontsize = 12)
graph.set_xticklabels(graph.get_xticklabels(),rotation=30)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# ###### Here 0 stands for not survived and 1 stands for survived.
# 
# ###### So, we can see that Pclass plays a major role in survival.
# 
# ###### Majority of people survived in Pclass 1 while a large number of people do not survive in Pclass 3.
# 
# 

# In[52]:


# percentage of survivors per class
sns.factorplot('Pclass', 'Survived', data = train)


# ###### The above plot indicates the percentage of survivors per class.

# ## 4.5 Embarked

# In[53]:


fig, ax = plt.subplots(figsize=(8,6))
graph = sns.countplot(ax=ax,x=train['Embarked'], data=train, palette = 'bone')
graph.set_title('Number of people across different embarkment', fontsize = 12)
graph.set_xticklabels(graph.get_xticklabels(),rotation=30)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# In[54]:


fig, ax = plt.subplots(figsize=(8,6))
graph = sns.countplot(ax=ax,x=train['Embarked'], data=train, hue='Survived', palette = 'bone')
graph.set_title('Number of people across different embarkment', fontsize = 12)
graph.set_xticklabels(graph.get_xticklabels(),rotation=30)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# ###### We can see that port of embarkment plays a major role in survival probability.

# ## 4.6 Age

# In[55]:


x = train['Age']
plt.figure(figsize=(8,6))
plt.hist(x, bins=25, color='g')
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.title('Age distribution of passengers', fontsize = 20)
plt.show()


# ##### We can see that majority of passengers are aged between 20 and 40.
# 
# ###### We will again visit this data visualization section in Feature Engineering section.
# 

# # 5. Data Preprocessing 

# ## 5.1 Remove redundant features

# The Ticket and PassengerId are redundant features. So, we will remove them from the dataset.

# In[56]:


train.drop(['Ticket', 'PassengerId'], axis = 1, inplace = True)
test.drop(['Ticket','PassengerId'], axis = 1, inplace = True)


# ## 5.2 Imputation of missing values in Age

# We will make additional column with the title of the person (Mr, Mrs, Miss, etc).
# 
# Then, we impute the missing values in age with the median age for each title.
# 
# Let's first make a function to extract title from Name feature.

# In[57]:


# function to extract title from Name feature
def passenger_title(passenger):
    line = passenger
    if re.search('Mrs', line):
        return 'Mrs'
    elif re.search('Mr', line):
        return 'Mr'
    elif re.search('Miss', line):
        return 'Miss'
    elif re.search('Master', line):
        return 'Master'
    else:
        return 'Other'


# In[58]:


# extract title  
train['Title'] = train['Name'].apply(passenger_title)
test['Title'] = test['Name'].apply(passenger_title)


# In[60]:


# fill missing age, with median from title segregation: funtion
def fill_age(passenger):
    
    # determine age by group 
    temp = train.groupby(train.Title).median()
    
    age, title = passenger
    
    if age == age:
        return age
    else:
        if title == 'Mr':
            return temp.Age['Mr']
        elif title == 'Miss':
            return temp.Age['Miss']
        elif title == ['Mrs']:
            return temp.Age['Mrs']
        elif title == 'Master':
            return temp.Age['Master']
        else:
            return temp.Age['Other']
        


# In[61]:


# fill age according to title
train['Age'] = train[['Age', 'Title']].apply(fill_age, axis = 1)
test['Age'] = test[['Age', 'Title']].apply(fill_age, axis = 1)
# Remove column Name, it is not us


# In[62]:


# Remove column Name, it is not useful for predictions and we extracted the title already
train.drop('Name', axis = 1, inplace = True)
test.drop('Name', axis = 1, inplace = True)


# In[63]:


# Remove column Title, it is not useful for predictions and we imputed the age already
train.drop('Title', axis = 1, inplace = True)
test.drop('Title', axis = 1, inplace = True)


# ## 5.3 Imputation of missing values in Cabin

# To extract missing values in Cabin, we extract Deck from Cabin and add 'Unknown' where NA.

# In[64]:


def isNaN(num):
    return num != num # checks if cell is NaN


# In[65]:


# get the first letter of cabin 
def first_letter_of_cabin(cabin):
    if not isNaN(cabin):
        return cabin[0]
    else:
        return 'Unknown'


# In[66]:


train['Deck'] = train['Cabin'].apply(first_letter_of_cabin)
test['Deck'] = test['Cabin'].apply(first_letter_of_cabin)


# In[67]:


# drop old variable Cabin
train.drop('Cabin', axis = 1, inplace = True)
test.drop('Cabin', axis = 1, inplace = True)


# ## 5.4 Imputation of missing values in Embarked ¶

# We impute Embarked with the most frequent port (S).

# In[68]:


train["Embarked"].fillna("S", inplace = True)
test['Embarked'].fillna("S", inplace = True)


#  Let's again check for missing values

# In[69]:


train.isnull().sum()


# In[70]:


test.isnull().sum()


# In[71]:


#we can replace missing value in fare by taking median of all fares of those passengers 
#who share 3rd Passenger class and Embarked from 'S' 
test['Fare'].fillna(test['Fare'].median(), inplace = True)


# In[72]:


test.isnull().sum()


# ## 5.5 Outlier Detection

# The Age and Fare variable contain putliers. Now, let's check for outliers in Age and Fare.
# 
# Let's draw boxplots to visualise outliers in the above variables.

# In[73]:


# draw boxplots to visualize outliers

plt.figure(figsize=(15,10))


plt.subplot(1, 2, 1)
fig = train.boxplot(column='Age')
fig.set_title('')
fig.set_ylabel('Age')


plt.subplot(1, 2, 2)
fig = train.boxplot(column='Fare')
fig.set_title('')
fig.set_ylabel('Fare')


# In[74]:


# find outliers in Age variable

IQR = train.Age.quantile(0.75) - train.Age.quantile(0.25)
Lower_fence = train.Age.quantile(0.25) - (IQR * 3)
Upper_fence = train.Age.quantile(0.75) + (IQR * 3)
print('Age outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=max(0, Lower_fence), upperboundary=Upper_fence))


# In[75]:


# find outliers in Fare variable

IQR = train.Fare.quantile(0.75) - train.Fare.quantile(0.25)
Lower_fence = train.Fare.quantile(0.25) - (IQR * 3)
Upper_fence = train.Fare.quantile(0.75) + (IQR * 3)
print('Fare outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=max(0, Lower_fence), upperboundary=Upper_fence))


# ###### Since, Age and Fare do not have values less than 0. So, we assume their minimum values to be 0.
# 
# ###### I will use top-coding approach to cap maximum values and remove outliers from the above variables.

# In[76]:


def max_value(df, variable, top):
    return np.where(df[variable]>top, top, df[variable])

for df in [train, test]:
    df['Age'] = max_value(df, 'Age', 81.0)
    df['Fare'] = max_value(df, 'Fare', 100.2688)


# Let's check that the above variables are capped at their maximum values.

# In[77]:


train.Age.max(), test.Age.max()


# In[78]:


train.Fare.max(), test.Fare.max()


# # 6. Feature Engineering 

# In this section, we will make additional columns for future analysis.

# ## 6.1 Categorize passengers as male, female or child

# Children have much larger probability of survival than men or women. So, we will categorize the passengers as men, women or child.

# In[79]:


# label minors as child, and remaining people as female or male
def male_female_child(passenger):
    # take the age and sex
    age, sex = passenger
    
    # compare age, return child if under 16, otherwise leave sex
    if age < 16:
        return 'child'
    else:
        return sex


# In[80]:


# new columns called person specifying if the person was female, male or child
train['Person'] = train[['Age', 'Sex']].apply(male_female_child, axis = 1)
test['Person'] = test[['Age', 'Sex']].apply(male_female_child, axis = 1)


# In[81]:


# Number of male, female and children on board
train['Person'].value_counts()


# In[82]:


# age segregated by class
fig = sns.FacetGrid(train, hue = 'Person', aspect = 4)
fig.map(sns.kdeplot, 'Age', shade = True)
fig.add_legend()


# In[83]:


# age segregated by class
fig = sns.FacetGrid(train, hue = 'Pclass', aspect = 4)
fig.map(sns.kdeplot, 'Age', shade = True)
fig.add_legend()


# We can see that the peak over 0 age for classes 2 and 3, coincides with the classes that had children. Class 1 did not have a lot of children, unsurprisingly. Note also, that older people were high class.
# 

# In[84]:


sns.factorplot('Pclass', 'Survived', hue = 'Person', data = train)


# We can see that males have lower probability of survival than females and children, regardless of the class they were in.
# As for women and children, being in class 3 meant that their chances of survival were lower.
# 

# ## 6.2 Make additional variable : travel alone

# In[85]:


def travel_alone(df):
    df['Alone'] = df.Parch + df.SibSp
    df['Alone'].loc[df['Alone'] > 0] = 'With Family'
    df['Alone'].loc[df['Alone'] == 0] = 'Alone'
    
    return df


# 0 indicates that person is travelling with family and 1 indicates that he is travelling alone.

# In[86]:


train = travel_alone(train)
test = travel_alone(test)


# In[87]:


# check how many passengers are travelling with family and alone
train['Alone'].value_counts()


# So, 537 people are travelling alone and 354 people are travelling with family.

# In[88]:


fig, ax = plt.subplots(figsize=(6,6))
graph = sns.countplot(ax=ax,x=train['Alone'], data = train, palette = 'PuBuGn_d')
graph.set_title('Distribution of people travelling alone or with family', fontsize = 12)
graph.set_xticklabels(graph.get_xticklabels(),rotation=30)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# In[89]:


fig, ax = plt.subplots(figsize=(6,6))
graph = sns.countplot(ax=ax,x=train['Alone'], data = train, hue = 'Survived', palette = 'PuBuGn_d')
graph.set_title('Distribution of people travelling alone or with family', fontsize = 12)
graph.set_xticklabels(graph.get_xticklabels(),rotation=30)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# We can see that almost half number of people who are travelling with family survived whereas large number of people travelling alone did not survive.
# 
# So, travelling alone or with family plays a major role in deciding the survival probability.

# In[90]:


# percentage of survivors depending on traveling alone or with family
sns.factorplot('Alone', 'Survived', hue = 'Person', data = train)


# In[91]:


train.head()


# In[92]:


fig, ax = plt.subplots(figsize=(6,6))
graph = sns.countplot(ax=ax,x=train['Deck'], data = train[train.Deck != 'Unknown'], hue = 'Survived', palette = 'PuBuGn_d')
graph.set_title('Distribution of people on each deck', fontsize = 12)
graph.set_xticklabels(graph.get_xticklabels(),rotation=30)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# The people who are on deck C and B have larger probability of survival.

# ## 6.3 Correlation of features with target

# In[93]:


train.corr()['Survived']


# We can see that Survived is negatively correlated with Pclass,Age,SibSp,Embarked,Deck,Person,Alone and positively correlated with Parch and Fare.
# 
# We can also plot a heatmap to visualize the relationship between features.

# In[94]:


corr=train.corr()#["Survived"]
plt.figure(figsize=(10, 10))
sns.heatmap(corr, vmax=.8, linewidths=0.01, square=True,annot=True,cmap='YlGnBu',linecolor="white")
plt.title('Correlation between features')
plt.show()


# # 7. Categorical Variable Encoding

# Now, let's take a look at train and test set.

# In[95]:


train.head()


# Drop the Sex variable.

# In[97]:


train.drop('Sex', axis=1, inplace=True)
test.drop('Sex', axis=1, inplace=True)


# There are 4 variables that need to be categorical encoded.
# 
# They are Embarked,Deck,Person and Alone

# In[98]:


train['Alone'] = pd.get_dummies(train['Alone'])
test['Alone'] = pd.get_dummies(test['Alone'])


# In[99]:


labelenc=LabelEncoder()

categorical=['Embarked','Deck','Person']
for col in categorical:
    train[col]=labelenc.fit_transform(train[col])
    test[col]=labelenc.fit_transform(test[col])

train.head()


# In[100]:


test.head()


# # 8. Feature Scaling

# We need to do Feature Scaling first before proceeding with modeling.

# In[101]:


train_cols = train.columns
test_cols = test.columns


# In[102]:


scaler = StandardScaler()
train[['Age', 'Fare']] = scaler.fit_transform(train[['Age', 'Fare']])
test[['Age', 'Fare']] = scaler.transform(test[['Age', 'Fare']])


# # 9. Modelling

# In[105]:


# Declare feature vector and target variable
X = train.drop(labels = ['Survived'],axis = 1)
y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# ## 9.1 Predict accuracy with different algorithms

# I predict accuracy with 15 popular classifiers and evaluate their performance.

# In[106]:


names = ["Logistic Regression", "Nearest Neighbors", "Naive Bayes", "Linear SVM", "RBF SVM", 
         "Gaussian Process", "Decision Tree", "Random Forest", "AdaBoost", "Gradient Boosting", 
         "LDA", "QDA", "Neural Net", "LightGBM", "XGBoost" ]    


# In[107]:


classifiers = [
    LogisticRegression(),
    KNeighborsClassifier(5),
    GaussianNB(),
    SVC(kernel="linear", C=0.025),
    SVC(kernel = "rbf", gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    MLPClassifier(alpha=1, max_iter=1000),
    lgb.LGBMClassifier(),    
    xgb.XGBClassifier()
   ]


# In[108]:


accuracy_scores = []

# iterate over classifiers and predict accuracy
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    score = round(score, 4)
    accuracy_scores.append(score)
    print(name ,' : ' , score)


# In[109]:


classifiers_performance = pd.DataFrame({"Classifiers": names, "Accuracy Scores": accuracy_scores})
classifiers_performance


# In[110]:


classifiers_performance.sort_values(by = 'Accuracy Scores' , ascending = False)[['Classifiers', 'Accuracy Scores']]


# ## 9.2 Plot the classifier accuracy scores 

# In[111]:


fig, ax = plt.subplots(figsize=(8,6))
x = classifiers_performance['Accuracy Scores']
y = classifiers_performance['Classifiers']
ax.barh(y, x, align='center', color='green')
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Accuracy Scores')
ax.set_ylabel('Classifiers', rotation=0)
ax.set_title('Classifier Accuracy Scores')
plt.show()


# # 10. Feature Selection

# In this section, we will see how to improve model performance by feature selection.
# 
# We will visualize feature importance with random forest classifier and drop the least important feature, rebuild the model and check effect on accuracy.
# 
# For a comprehensive overview on feature selection techniques, please see the kernel -

# ## 10.1 Feature Importance with Random Forest model 

# Until now, I have used all the features given in the model. Now, I will select only the important features, build the model using these features and see its effect on accuracy.
# 
# First, I will create the Random Forest model as follows:-

# In[112]:


# instantiate the classifier with n_estimators = 100
clf = RandomForestClassifier(n_estimators=100, random_state=0)


# fit the classifier to the training set
clf.fit(X_train, y_train)


# Now, I will use the feature importance variable to see feature importance scores.

# In[113]:


# view the feature scores
feature_scores = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
feature_scores


# In[114]:


feature_scores.values


# In[115]:


feature_scores.index


# We can see that the most important feature is Person and least important feature is Alone.

# ## 10.2 Visualize feature scores

# In[116]:


# Creating a seaborn bar plot to visualize feature scores
f, ax = plt.subplots(figsize=(8,6))
ax = sns.barplot(x=feature_scores.values, y=feature_scores.index, palette='spring')
ax.set_title("Visualize feature scores of the features")
ax.set_yticklabels(feature_scores.index)
ax.set_xlabel("Feature importance score")
ax.set_ylabel("Features")
plt.show()


# ## 10.3 Drop least important feature 

# Now, I will drop the least important feature Alone from the model, rebuild the model and check its effect on accuracy.

# In[117]:


# drop the least important feature from X_train, X_test and test set for further analysis
X1_train = X_train.drop(['Alone'], axis=1)
X1_test = X_test.drop(['Alone'], axis=1)
test = test.drop(['Alone'], axis=1)


# In[118]:


accuracy_scores1 = []

# iterate over classifiers and predict accuracy
for name, clf in zip(names, classifiers):
    clf.fit(X1_train, y_train)
    score = clf.score(X1_test, y_test)
    score = round(score, 4)
    accuracy_scores1.append(score)
    print(name ,' : ' , score)


# In[119]:


classifiers_performance1 = pd.DataFrame({"Classifiers": names, "Accuracy Scores": accuracy_scores, 
                                         "Accuracy Scores1": accuracy_scores1})
classifiers_performance1


# ###### We can see that Gaussian Process has the maximum accuracy of 0.8441.
# 
# ###### We will use the Gaussian Process Classifier to plot the confusion-matrix.
# 
# 

# # 11. Confusion matrix

# In[120]:


# instantiate the XGBoost classifier
gpc_clf = GaussianProcessClassifier(1.0 * RBF(1.0))


# fit the classifier to the modified training set
gpc_clf.fit(X1_train, y_train)


# In[121]:


# predict on the test set
y1_pred = gpc_clf.predict(X1_test)


# In[122]:


# print the accuracy
print('Gaussian Process Classifier model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y1_pred)))


# In[123]:


# print confusion-matrix

cm = confusion_matrix(y_test, y1_pred)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])


# The confusion matrix shows 156 + 93 = 249 correct predictions and 19 + 27 = 46 incorrect predictions.
# 
# In this case, we have
# 
# True Positives (Actual Positive:1 and Predict Positive:1) - 156
# 
# True Negatives (Actual Negative:0 and Predict Negative:0) - 93
# 
# False Positives (Actual Negative:0 but Predict Positive:1) - 19 (Type I error)
# 
# False Negatives (Actual Positive:1 but Predict Negative:0) - 27 (Type II error)

# In[125]:


# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


# # 12. Classification Metrices

# ## 12.1 Classification Report

# Classification Report is another way to evaluate the classification model performance.
# 
# It displays the precision, recall, f1 and support scores for the model.
# 
# We can print a classification report as follows:-

# In[126]:


print(classification_report(y_test, y1_pred))


# ## 12.2 Classification Accuracy

# In[127]:


TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]


# In[128]:


# print classification accuracy

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))


# ## 12.3 Classification Error 

# In[129]:


# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))


# ## 12.4 Precision

# Precision can be defined as the percentage of correctly predicted positive outcomes out of all the predicted positive outcomes. It can be given as the ratio of true positives (TP) to the sum of true and false positives (TP + FP).
# 
# So, Precision identifies the proportion of correctly predicted positive outcome. It is more concerned with the positive class than the negative class.
# 
# Mathematically, precision can be defined as the ratio of TP to (TP + FP).

# In[130]:


# print precision score

precision = TP / float(TP + FP)

print('Precision : {0:0.4f}'.format(precision))


# ## 12.5 Recall 

# Recall can be defined as the percentage of correctly predicted positive outcomes out of all the actual positive outcomes. It can be given as the ratio of true positives (TP) to the sum of true positives and false negatives (TP + FN).
# 
# Recall is also called Sensitivity.
# 
# Recall identifies the proportion of correctly predicted actual positives.
# 
# Mathematically, Recall can be given as the ratio of TP to (TP + FN).

# In[131]:


recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall))


# ## 12.6 True Positive Rate

# True Positive Rate is synonymous with Recall.

# In[132]:


true_positive_rate = TP / float(TP + FN)

print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))


# ## 12.7 False Positive Rate

# In[133]:


false_positive_rate = FP / float(FP + TN)

print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))


# ## 12.8 Specificity (True Negative Rate) 

# Specificity is also called True Negative Rate

# In[134]:


specificity = TN / (TN + FP)

print('Specificity : {0:0.4f}'.format(specificity))


# ## 12.9 f1-score

# f1-score is the weighted harmonic mean of precision and recall.
# 
# The best possible f1-score would be 1.0 and the worst would be 0.0.
# 
# f1-score is the harmonic mean of precision and recall.
# 
# So, f1-score is always lower than accuracy measures as they embed precision and recall into their computation.
# 
# The weighted average of f1-score should be used to compare classifier models, not global accuracy.

# ## 12.10 Support 

# Support is the actual number of occurrences of the class in our dataset.

# # 13. Cross Validation

# In[135]:


# iterate over classifiers and calculate cross-validation score
for name, clf in zip(names, classifiers):
    scores = cross_val_score(clf, X1_train, y_train, cv = 10, scoring='accuracy')
    print(name , ':{:.4f}'.format(scores.mean()))


# # 14. Hyperparameter Optimization using GridSearch CV

# I choose the top 3 classifiers with maximum accuracy for ensemble modeling.
# 
# They are AdaBoost, LightGBM and Gradient Boosting.
# 
# So, we will tune the hyperparameters of these models before proceeding.

# ### AdaBoost Classifier Parameters tuning

# In[136]:


abc_params = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "n_estimators": [1, 2]
             }

dtc_clf = DecisionTreeClassifier(random_state = 0, max_features = "auto", class_weight = "balanced", max_depth = None)

abc_clf = AdaBoostClassifier(base_estimator = dtc_clf)


abc_grid_search = GridSearchCV(estimator = abc_clf,  
                               param_grid = abc_params,
                               scoring = 'accuracy',
                               cv = 5,
                               verbose=0)

abc_grid_search.fit(X1_train, y_train)


# In[137]:


# examine the best model

# best score achieved during the GridSearchCV
print('AdaBoost GridSearch CV best score : {:.4f}\n\n'.format(abc_grid_search.best_score_))

# print parameters that give the best results
print('AdaBoost Parameters that give the best results :','\n\n', (abc_grid_search.best_params_))

# print estimator that was chosen by the GridSearch
abc_best = abc_grid_search.best_estimator_
print('\n\nXGBoost Estimator that was chosen by the search :','\n\n', (abc_best))


# # LightGBM Parameters tuning¶

# In[139]:


lgb_clf = lgb.LGBMClassifier()


lgb_params={'learning_rate': [0.005],
    'num_leaves': [6,8,12,16],
    'objective' : ['binary'],
    'colsample_bytree' : [0.5, 0.6],
    'subsample' : [0.65,0.66],
    'reg_alpha' : [1,1.2],
    'reg_lambda' : [1,1.2,1.4],
    }


lgb_grid_search = GridSearchCV(estimator = lgb_clf,  
                               param_grid = lgb_params,
                               scoring = 'accuracy',
                               cv = 5,
                               verbose=0)


lgb_grid_search.fit(X1_train, y_train)


# In[140]:


# examine the best model

# best score achieved during the GridSearchCV
print('LightGBM GridSearch CV best score : {:.4f}\n\n'.format(lgb_grid_search.best_score_))

# print parameters that give the best results
print('LightGBM Parameters that give the best results :','\n\n', (lgb_grid_search.best_params_))

# print estimator that was chosen by the GridSearch
lgb_best = lgb_grid_search.best_estimator_
print('\n\nLightGBM Estimator that was chosen by the search :','\n\n', (lgb_best))


# # Gradient Boost Parameters tuning

# In[141]:


gbc_clf = GradientBoostingClassifier()

gbc_params = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }

gbc_grid_search = GridSearchCV(estimator = gbc_clf, 
                               param_grid = gbc_params, 
                               scoring = "accuracy", 
                               cv = 5,
                               verbose = 0)

gbc_grid_search.fit(X1_train,y_train)


# In[142]:


# examine the best model

# best score achieved during the GridSearchCV
print('Gradient Boosting GridSearch CV best score : {:.4f}\n\n'.format(gbc_grid_search.best_score_))

# print parameters that give the best results
print('Gradient Boosting Parameters that give the best results :','\n\n', (gbc_grid_search.best_params_))

# print estimator that was chosen by the GridSearch
gbc_best = gbc_grid_search.best_estimator_
print('\n\nGradient Boosting Estimator that was chosen by the search :','\n\n', (gbc_best))


# # 15. Ensemble Modeling

# I decided to choose a voting classifier to combine the predictions coming from the above 3 classifiers.

# In[143]:


votingC = VotingClassifier(estimators=[('abc', abc_best), ('lgb',lgb_best), ('gbc',gbc_best)], voting='soft')

votingC = votingC.fit(X1_train, y_train)


# # 16. Submission

# In[144]:


test_Survived = pd.Series(votingC.predict(test), name="Survived")

submission = pd.concat([IDtest,test_Survived],axis=1)


submission.to_csv("titanic_submission.csv", index=False)


# # 17. Conclusion 

# In this notebook, we have build a classification model on the famous titanic dataset.
# 
# We have used a voting ensemble classifier for making predictions.

# In[ ]:




