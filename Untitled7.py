
# coding: utf-8

# In[143]:


### 1 Defining Libraries and Install The Data
### 2 Examine the Data and Filling the Missing Values
### 3 Pivoting Data , Graphs , Examining the Varibles (Categoric , Numeric , ..)
### 4 Data Wrangling and Feature Enginnering
### 5 Modelling



### 1

## Libraries
from matplotlib import pyplot as plt
import seaborn as seab
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier


## Upload - Install data
train = pd.read_csv("C:/Users/Ugur/Desktop/Titanic/Titanic Kaggle Original/train.csv")
test = pd.read_csv("C:/Users/Ugur/Desktop/Titanic/Titanic Kaggle Original/test.csv")


### 2

## Examine data
train.sample(5)
test.sample(5)
train.head()
test.head()

## Detecting Missing Values 
print ("The shape of the train data is (row, column):"+ str(train.shape))
print (train.info())
print ("The shape of the test data is (row, column):"+ str(test.shape))
print (test.info())


## Finding missing values
total = train.isnull().sum().sort_values(ascending = False)
percent = round(train.isnull().sum().sort_values(ascending = False)/len(train)*100, 2)
trainmv = pd.concat([total, percent], axis = 1,keys= ['Total', 'Percent'])
print(trainmv)

total = test.isnull().sum().sort_values(ascending = False)
percent = round(test.isnull().sum().sort_values(ascending = False)/len(test)*100, 2)
testmv = pd.concat([total, percent], axis = 1,keys= ['Total', 'Percent'])
print(testmv)


print('Train columns with null values:\n', train.isnull().sum())
print("-"*10)

print('Test columns with null values:\n', test.isnull().sum())
print("-"*10)


## Filling Missing Data

#For Embarked Feature

percent = pd.DataFrame(round(train.Embarked.value_counts(dropna=False, normalize=True)*100,2))
total = pd.DataFrame(train.Embarked.value_counts(dropna=False))

total.columns = ["Total"]
percent.columns = ['Percent']
mv_embarked = pd.concat([total, percent], axis = 1)
print(mv_embarked)

                    # There are 2 null values for Embarked feature #

n_embarked = train[train.Embarked.isnull()]

print(n_embarked)

            #both null values have fare of $80 , are of Pclass 1 and female Sex. 


#Fare  distribution among all Pclass and Embarked feature values ##

#fig, ax = plt.subplots(figsize=(16,12),ncols=2)
#ax1 = sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=train, ax = ax[0]);
#ax2 = sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=test, ax = ax[1]);
#ax1.set_title("Training Set", fontsize = 18)
#ax2.set_title('Test Set',  fontsize = 18)
#fig.show()
## Charts are only to show us which values can be assignedd for Embarked values #

#in both training set and test set, the average fare closest to $80 are in the C Embarked values. 

# let's fill in the missing values as "C"

train.Embarked.fillna("C", inplace=True)


#For Cabin Feature

print("Train Cabin missing: " + str(train.Cabin.isnull().sum()/len(train.Cabin)))
print("Test Cabin missing: " + str(test.Cabin.isnull().sum()/len(test.Cabin)))

                                #Train Data Cabin missing: % 77
                                 #Test Data Cabin missing: % 78

           # We can predict the cabin class with using the mean of other cabin classes #
                   # Firstly we should combine test and train datasets #
                   # And We will assign all the null values as "N" #

survivers = train.Survived

train.drop(["Survived"],axis=1, inplace=True)

all_data = pd.concat([train,test], ignore_index=False)

all_data.Cabin.fillna("N", inplace=True)


all_data.Cabin = [i[0] for i in all_data.Cabin]

with_N = all_data[all_data.Cabin == "N"]

without_N = all_data[all_data.Cabin != "N"]

                                #  mean of each cabin letter
    
CFV =all_data.groupby("Cabin")['Fare'].mean().sort_values() 

print(CFV)

                   # we can assign these means to the unknown cabins with a function

def cbn_est(i):
    a = 0
    if i<16:
        a = "G"
    elif i>=16 and i<27:
        a = "F"
    elif i>=27 and i<38:
        a = "T"
    elif i>=38 and i<47:
        a = "A"
    elif i>= 47 and i<53:
        a = "E"
    elif i>= 53 and i<54:
        a = "D"
    elif i>=54 and i<116:
        a = 'C'
    else:
        a = "B"
    return a

                              #applying the function#

with_N['Cabin'] = with_N.Fare.apply(lambda x: cbn_est(x))

                           # get back to the train #
    
all_data = pd.concat([with_N, without_N], axis=0)

                    # With PassengerId we can separate train and test.#
    
all_data.sort_values(by = 'PassengerId', inplace=True)

                    # Separation of train and test from all_data #
train = all_data[:891]
test = all_data[891:]

                    # adding saved target variable with train #
train['Survived'] = survivers


# For Age Feature

print ("Train age missing value: " + str((train.Age.isnull().sum()/len(train))*100)+str("%"))
print ("Test age missing value: " + str((test.Age.isnull().sum()/len(test))*100)+str("%"))

                         #Train age missing value: %19.86 #
                         #Test age missing value: %20.57 #
    


# For Fare Feature

fare_null = test[test.Fare.isnull()]

 #We can take the average of the values wherePclass is 3, Sex is male and Embarked is S and than fill to Null value#

missing_value = test[(test.Pclass == 3) & (test.Embarked == "S") & (test.Sex == "male")].Fare.mean()

                    ## replace the test.fare null values with test.fare mean#
test.Fare.fillna(missing_value, inplace=True)

                #describe datasets which doesnt have missing values (except Age)
trd = train.describe()
ted = test.describe()

print(trd)
print(ted)


### 3


## Distribution of categorical values
trdc = train.describe(include=['O'])
print(trdc)

# RESULTS # 
# names are unique #
# %65 of male #
# S port has most passengers #
# %22 of values are duplicated for ticket variable #


##Pivoting features

                                    #Pivot Class#
pvt_pclass = train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(pvt_pclass)

                                      #Pivot Sex#
pvt_sex = train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(pvt_sex)

                                      #Pivot Sibsp#
pvt_sibsp = train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(pvt_sibsp)

                                      #Pivot Parch#
pvt_parch = train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(pvt_parch)

                                        #Pivot Cabin#
pvt_cabin = train[["Cabin", "Survived"]].groupby(['Cabin'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(pvt_cabin)


## Overview - Survived
survived_summary = train.groupby("Survived")
srv = survived_summary.mean().reset_index()
print(srv)



# RESULTS #
# only 38% passenger survived during that tragedy.#
# ~74% female passenger survived, while only ~19% male passenger survived.#
# ~63% first class passengers survived, while only 24% lower class passenger survived.#
# Survival Possibility is better for 3,2 and 1 parch #

##Observations For Age - Correlating Numeric Values

u = seab.FacetGrid(train, col='Survived')
u.map(plt.hist, 'Age', bins=40)


##Observations for Pclass - Correlating Numeric Values

grid = seab.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.7, bins=40)
grid.add_legend();

#Observation Results 

#Infants (Age <=4) had high survival rate.
#Oldest passengers (Age = 80) survived.
#Large number of 15-25 year olds did not survive.
#Most passengers are in 15-35 age range.






### 4 Data Wrangling and Feature Engineering

## Data Wrangling 

                           #Titles#

                    ## Getting the title from the name and assign rare values to other titles #

train["title"] = [i.split('.')[0] for i in train.Name]
train["title"] = [i.split(',')[1] for i in train.title]
test["title"] = [i.split('.')[0] for i in test.Name]
test["title"]= [i.split(',')[1] for i in test.title]

train["title"] = [i.replace('Ms', 'Miss') for i in train.title]
train["title"] = [i.replace('Mlle', 'Miss') for i in train.title]
train["title"] = [i.replace('Mme', 'Mrs') for i in train.title]
train["title"] = [i.replace('Dr', 'rare') for i in train.title]
train["title"] = [i.replace('Col', 'rare') for i in train.title]
train["title"] = [i.replace('Major', 'rare') for i in train.title]
train["title"] = [i.replace('Don', 'rare') for i in train.title]
train["title"] = [i.replace('Jonkheer', 'rare') for i in train.title]
train["title"] = [i.replace('Sir', 'rare') for i in train.title]
train["title"] = [i.replace('Lady', 'rare') for i in train.title]
train["title"] = [i.replace('Capt', 'rare') for i in train.title]
train["title"] = [i.replace('the Countess', 'rare') for i in train.title]
train["title"] = [i.replace('Rev', 'rare') for i in train.title]


test['title'] = [i.replace('Ms', 'Miss') for i in test.title]
test['title'] = [i.replace('Dr', 'rare') for i in test.title]
test['title'] = [i.replace('Col', 'rare') for i in test.title]
test['title'] = [i.replace('Dona', 'rare') for i in test.title]
test['title'] = [i.replace('Rev', 'rare') for i in test.title]

                    # family size #

train['family_size'] = train.SibSp + train.Parch+1
test['family_size'] = test.SibSp + test.Parch+1

def family_group(size):
    a = ''
    if (size <= 1):
        a = 'loner'
    elif (size <= 4):
        a = 'small'
    else:
        a = 'large'
    return a

train['family_group'] = train['family_size'].map(family_group)
test['family_group'] = test['family_size'].map(family_group)

titlecount = train[['title', 'Survived']].groupby(['title'], as_index=False).mean()
print(titlecount)

     # We will assign ordinal values for both test and train data for title varible with survived mean #

train["title"] = [i.replace('Mrs', '4') for i in train.title]
train["title"] = [i.replace('Mr', '3') for i in train.title]
train["title"] = [i.replace('Miss', '2') for i in train.title]
train["title"] = [i.replace('Master', '1') for i in train.title]
train["title"] = [i.replace('rare', '5') for i in train.title]

test["title"] = [i.replace('Mrs', '4') for i in test.title]
test["title"] = [i.replace('Mr', '3') for i in test.title]
test["title"] = [i.replace('Miss', '2') for i in test.title]
test["title"] = [i.replace('Master', '1') for i in test.title]
test["title"] = [i.replace('rare', '5') for i in test.title]
    
                         # is alone
train['is_alone'] = [1 if i<2 else 0 for i in train.family_size]
test['is_alone'] = [1 if i<2 else 0 for i in test.family_size]
  
    # calculated ticket fare(Some of family ticket fares are too high , thats why we will divide the fare value to family size)

train['calculated_fare'] = train.Fare/train.family_size
test['calculated_fare'] = test.Fare/test.family_size


                         #Embarked

train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

        #With this results we will assign ordinal values to Embarked stations#


train["Embarked"] = [i.replace('S', '2') for i in train.Embarked]
train["Embarked"] = [i.replace('C', '1') for i in train.Embarked]
train["Embarked"] = [i.replace('Q', '3') for i in train.Embarked]


test["Embarked"] = [i.replace('S', '2') for i in test.Embarked]
test["Embarked"] = [i.replace('C', '1') for i in test.Embarked]
test["Embarked"] = [i.replace('Q', '3') for i in test.Embarked]



             # We will assign numeric values for Sex column #

train["Sex"] = [i.replace('female', '1') for i in train.Sex]
train["Sex"] = [i.replace('male', '0') for i in train.Sex]

test["Sex"] = [i.replace('female', '1') for i in test.Sex]
test["Sex"] = [i.replace('male', '0') for i in test.Sex]


b = test.head()
print(b)



                #### We used Fare Band just for assigning ####

#train['FareBand'] = pd.qcut(train['calculated_fare'], 4) #
#train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True) #

# Fare bands are [(-0.001, 7.25] < (7.25, 8.3] < (8.3, 23.667] < (23.667, 512.329]] #
# Defining fare group function and assigning values to the new column

 
## fare group for calculating fare #

def fare_group(calculated_fare):
    b = ''
    if (calculated_fare >= -0.001 and calculated_fare <= 7.25):
        b = 1
    elif (calculated_fare > 7.25 and calculated_fare <= 8.3):
        b = 2
    elif (calculated_fare > 8.3 and calculated_fare <= 23.667):
        b = 3
    else:
        b = 4
    return b


train['fare_group2'] = train['calculated_fare'].map(fare_group)
test['fare_group2'] = test['calculated_fare'].map(fare_group)


                      #Filling missing values on age column#
           #We will create a new varible for assigning mean to the null values of Age#
train['ps']=train['Pclass'].map(str)+train['Sex'].map(str)
test['ps']=test['Pclass'].map(str)+test['Sex'].map(str)

            # The mean of new varible will be assigned to the null values #

train.Age = train.groupby('ps')['Age'].apply(lambda x: x.fillna(x.mean()))
train.Age = train.Age.fillna(train.Age.mean())



test.Age = test.groupby('ps')['Age'].apply(lambda x: x.fillna(x.mean()))
test.Age = test.Age.fillna(test.Age.mean())



#Age band will be created and than deleted 

 #b =train['AgeBand'] = pd.cut(train['Age'], 5)
 #train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


#[(0.34, 16.336] < (16.336, 32.252] < (32.252, 48.168] < (48.168, 64.084] < (64.084, 80.0]]  # Age Bands 

# Defining age group function and assigning values to the new column

def Age_group(Age):
    c = ''
    if (Age >= -0.001 and Age <= 16.336):
        c = 1
    elif (Age > 16.336 and Age <= 32.252):
        c = 2
    elif (Age > 32.252 and Age <= 48.168):
        c = 3
    elif (Age > 48.168 and Age <= 64.084):
        c = 4
    else:
        c = 5
    return c



train['Age_group2'] = train['Age'].map(Age_group)
test['Age_group2'] = test['Age'].map(Age_group)

                   #Removing Unnecessary Columns from Datasets #

train = train.drop(['Name'], axis=1)
train = train.drop(['Age'], axis=1)
train = train.drop(['SibSp'], axis=1)
train = train.drop(['Ticket'], axis=1)
train = train.drop(['Fare'], axis=1)
train = train.drop(['calculated_fare'], axis=1)
train = train.drop(['Parch'], axis=1)
train = train.drop(['family_size'], axis=1)
train = train.drop(['family_group'], axis=1)
train = train.drop(['Cabin'], axis=1)
train = train.drop(['ps'], axis=1)
train = train.drop(['PassengerId'], axis=1)


test = test.drop(['Name'], axis=1)
test = test.drop(['Age'], axis=1)
test = test.drop(['Ticket'], axis=1)
test = test.drop(['SibSp'], axis=1)
test = test.drop(['Fare'], axis=1)
test = test.drop(['calculated_fare'], axis=1)
test = test.drop(['Parch'], axis=1)
test = test.drop(['family_size'], axis=1)
test = test.drop(['family_group'], axis=1)
test = test.drop(['Cabin'], axis=1)
test = test.drop(['ps'], axis=1)

g = test
print(g)

h = train
print(h)


### 5 Modelling

## Preparing Test And Train Datasets

X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# Implementation of Logistic Regression Model

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log

s = Y_pred
print(s)



#Confidence Level - 79.01 
# There are still things to do...









