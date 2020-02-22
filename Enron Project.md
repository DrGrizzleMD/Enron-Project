## Using Machine Learning To Identify Fraud in Enron Scandal

### By: Jonathan Grays

### February 2020

### Table of Contents
<li><a href="#overview">Overview</a></li>
<li><a href="#projectgoal">Project Goal</a></li>
<li><a href="#questionsaboutdataset">Questions about Dataset</a></li>
    <ol>
    <li><a href="#howmanypeople">How many people are in the dataset?</a></li>
    <li><a href="#namesofpeople">Names of people in dataset?</a></li>
    <li><a href="#featuresperperson">What features do we have for each person?</a></li>
    <li><a href="#whoarepois">Who are the POIs?</a></li>
    <li><a href="#useful">Do any features stick out as initially useful?</a></li>
    </ol>
<li><a href="#nans">Dealing with NaNs</a></li>
<li><a href="#customfeatures">Custom Feature Exploration</a></li>
<li><a href="#missingvalues">Missing Values</a></li>
<li><a href="#selectingfeatures">Selecting Features</a></li>
<li><a href="#parametertuning">Parameter Tuning</a></li>
<li><a href="#wrapup">Project Wrap Up</a></li>

<a id='overview'></a>
### Overview

Write an overview of Enron's history leading up to and including fraud

<a id='projectgoal'></a>
### Project Goal
The goal of this project is to create a machine learning program/algorithm that will try to identify persons of interest in the Enron Financial Scandal.  Machine learning is a useful means to do this as it can be scaled up to datasets of almost any size.  Fortunately, since we know the persons of interest ahead of times, we can use a supervised algorithm to construct our identifier.  This can be accomplished by picking the features within our dataset that separate our POIs from the non-POIS in the best way.

We will first start our project with answering some basic questions about the data.  Once we have the basic information we will move on to visualizing our features and any correlations/outliers that may exist.


```python
# Import starting packages
# Others will be added as necessary

import sys
sys.path.append("..\\tools\\")
import pickle
import sklearn
import pandas as pd
import numpy as np
import pprint

# Bring data into a dictionary

enron_dict = pd.read_pickle(r"final_project_dataset.pkl")

# Ensure correct datatype
print('Dataset Type: ', type(enron_dict))
```

    ('Dataset Type: ', <type 'dict'>)
    

<a id='questionsaboutdataset'></a>
### Questions about Dataset

1. How many people are in the dataset?
2. Names of people in dataset?
3. What features do we have for them? (What info is recorded?)
4. Who are the POIs? (Persons of Interest)
5. Do any features stick out as initially useful?

<a id='howmanypeople'></a>
1. <u>How many people are in the dataset?</u>


```python
# Print the number of people in dataset

print("Number of people in dataset: ", len(enron_dict))
```

    ('Number of people in dataset: ', 146)
    

<a id='namesofpeople'></a>
2. <u>Names of people in dataset?</u>


```python
# Using pretty print to display names of people in the dataset
# Will be sorted by Last Name

printing = pprint.PrettyPrinter()

names = sorted(enron_dict.keys())
```

<a id='featuresperperson'></a>
3. <u>What features do we have for each person?</u>


```python
# Printing an example person and their corresponding dictionary

print("One example from the dataset:")

printing.pprint(enron_dict['BLACHMAN JEREMY M'])
```

    One example from the dataset:
    {'bonus': 850000,
     'deferral_payments': 'NaN',
     'deferred_income': 'NaN',
     'director_fees': 'NaN',
     'email_address': 'jeremy.blachman@enron.com',
     'exercised_stock_options': 765313,
     'expenses': 84208,
     'from_messages': 14,
     'from_poi_to_this_person': 25,
     'from_this_person_to_poi': 2,
     'loan_advances': 'NaN',
     'long_term_incentive': 831809,
     'other': 272,
     'poi': False,
     'restricted_stock': 189041,
     'restricted_stock_deferred': 'NaN',
     'salary': 248546,
     'shared_receipt_with_poi': 2326,
     'to_messages': 2475,
     'total_payments': 2014835,
     'total_stock_value': 954354}
    

<a id='whoarepois'></a>
4. <u>Who are the POIs?</u>


```python
# Import dictionary as a dataframe

enron_df = pd.DataFrame(enron_dict)

# Flip Names to be row headers
enron_df = enron_df.transpose()

# Display names and number of POIs

poi_list = enron_df.index[enron_df['poi'] == True].tolist()

print("Number of POIs: ", len(poi_list))

printing.pprint(poi_list)
```

    ('Number of POIs: ', 18)
    ['BELDEN TIMOTHY N',
     'BOWEN JR RAYMOND M',
     'CALGER CHRISTOPHER F',
     'CAUSEY RICHARD A',
     'COLWELL WESLEY',
     'DELAINEY DAVID W',
     'FASTOW ANDREW S',
     'GLISAN JR BEN F',
     'HANNON KEVIN P',
     'HIRKO JOSEPH',
     'KOENIG MARK E',
     'KOPPER MICHAEL J',
     'LAY KENNETH L',
     'RICE KENNETH D',
     'RIEKER PAULA H',
     'SHELBY REX',
     'SKILLING JEFFREY K',
     'YEAGER F SCOTT']
    

<a id='useful'></a>
5. <u>Do any features stick out as initially useful?</u>

- Since we are dealing with financial fraud, features like 'bonus', 'salary', 'total_payments', 'exercised_stock_options' and 'total_stock_value' seem like a good place to start. Also, 'from_poi_to_this_person' and 'from_this_person_to_poi' caught my attention as well since these could lead to patterns of communication with the POIs (paper trails).

- I will create a new feature for this dataset that is derived from the emails to and from a POI for a given person in our list.  A lot of direct communication with a POI may not be all that useful to us (if say a non-POI directly reported to a POI, they would certainly have had a good amount of email correspondence).  So instead, let's transform these features into a proportion of email correspondence with a POI to total emails sent/received per person.

- Before creating this new feature, I will need to convert my columns that contain numeric data into float types.


```python
# Sets selected columns (all but email_address and POI) to numeric/float type

cols = ['salary', 'to_messages', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus','restricted_stock_deferred',
        'deferred_income', 'total_stock_value', 'expenses', 'from_poi_to_this_person', 'exercised_stock_options',
        'from_messages', 'other', 'from_this_person_to_poi', 'long_term_incentive', 'shared_receipt_with_poi',
        'restricted_stock','director_fees']
enron_df[cols] = enron_df[cols].apply(pd.to_numeric, errors='coerce', axis=1)

enron_df = enron_df.reset_index()

enron_df.rename(columns={"index": "name"}, inplace=True)
```

<a id='nans'></a>

- There are quite a few NaN values in this dataset, which could be a bit discouraging.  Depending on the column that those NaNs are under they could be omitted or overwritten with zeroes.

- Now lets visualize some of the financial data to see if we can see any trends and/or outliers that would shape our perspective of this data.
</br>

- First we shall plot Salary and Bonus information


```python
# Plot of salary (x-axis) and bonus (y-axis)

import matplotlib.pyplot as plt
%matplotlib inline

plt.title('Salary vs Bonus', fontsize = 18)
plt.scatter('salary', 'bonus', data = enron_df)
plt.xlabel('Salary (in millions)', fontsize = 15)
plt.ylabel('Bonus (in millions)', fontsize = 15)
plt.show()


```


![png](output_18_0.png)


- Immediately we see a very distinct outlier.  This entry is definitely skewing our data.  In it's current state we can't really infer much of what the graph is trying to tell us.  We should take a look at the entry to make sure this is not an error.


```python
enron_df[(enron_df['salary']>2000000)][['name','salary','bonus','poi']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>salary</th>
      <th>bonus</th>
      <th>poi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>130</th>
      <td>TOTAL</td>
      <td>26704229.0</td>
      <td>97343619.0</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



- The name "TOTAL" must be referring to some sort of calculation done on the data (likely the sum of all salaries in the dataset).  Since this information is also something we can calculate ourselves, I will be removing this entry and replotting.


```python
## remove the outlier and verify it is gone.

df_remove_total = enron_df[ enron_df['name'] == 'TOTAL' ]

enron_df = enron_df.drop(df_remove_total.index, axis=0)
    
enron_df[(enron_df['salary']>2000000)][['name','salary','bonus','poi']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>salary</th>
      <th>bonus</th>
      <th>poi</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
# Replot salary vs bonus // Also importing Seaborn for graph readability

import seaborn as sns

sns.lmplot(x='salary', y= 'bonus', hue='poi', data=enron_df, palette='Set1', height=10, markers=['x','o'])
plt.title('Salary vs Bonus', fontsize = 18)
plt.xlabel('Salary (in millions)', fontsize = 15)
plt.ylabel('Bonus (in millions)', fontsize = 15)
```




    Text(25.6363,0.5,'Bonus (in millions)')




![png](output_23_1.png)


- Now we have a much better look at the Salary and Bonus data!

- Immediately, a few outliers stand out.  Let's find out who they are.


```python
# Looking at 

print("Salary Outliers: \n", enron_df[(enron_df['salary'] > 1000000)][['name', 'salary', 'poi']])

print("Bonus Outliers: \n", enron_df[(enron_df['bonus'] > 4000000)][['name', 'bonus', 'poi']])
```

    ('Salary Outliers: \n',                    name     salary    poi
    47       FREVERT MARK A  1060932.0  False
    79        LAY KENNETH L  1072321.0   True
    122  SKILLING JEFFREY K  1111258.0   True)
    ('Bonus Outliers: \n',                    name      bonus    poi
    0       ALLEN PHILLIP K  4175000.0  False
    7      BELDEN TIMOTHY N  5249999.0   True
    78      LAVORATO JOHN J  8000000.0  False
    79        LAY KENNETH L  7000000.0   True
    122  SKILLING JEFFREY K  5600000.0   True)
    

- This has identified 3 POIs!  Two of which (Kenneth Lay and Jeffrey Skilling) are noted in both outliers lists... interesting.


- Just who are these two?  Research indicates that these two men are:
    * <b>Jeffrey Skilling</b>: Former CEO and Chairman of Enron.  Resigned from company in August 2001
    * <b>Kenneth Lay</b>: CEO, Founder and Chairman of Enron during the scandal.


- Beings that these men were both CEOs of Enron, it does makes sense that they would be at the top of the earnings for both earnings and salary.  What is suspicious is that the top three salaried people listed made nearly DOUBLE what the fourth ranked salaried person made.  That seems like quite a gap, even for a CEO.


<a id='customfeatures'></a>
### Custom Feature Exploration

- Now let's plot out and explore the custom features we created, "fraction_emails_to_poi" and "fraction_emails_to_poi".


```python
# Added a feature listing the fraction of emails to and from POIs
# Also fills in 'NaN' spaces
enron_df['fraction_emails_to_poi'] = enron_df['from_this_person_to_poi'].fillna(0.0)/ (enron_df['to_messages'].fillna(0.0) + enron_df['from_this_person_to_poi'].fillna(0.0))

enron_df['fraction_emails_from_poi'] = enron_df['from_poi_to_this_person'].fillna(0.0)/ (enron_df['from_messages'].fillna(0.0) + enron_df['from_poi_to_this_person'].fillna(0.0))

enron_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 145 entries, 0 to 145
    Data columns (total 24 columns):
    name                         145 non-null object
    bonus                        81 non-null float64
    deferral_payments            38 non-null float64
    deferred_income              48 non-null float64
    director_fees                16 non-null float64
    email_address                145 non-null object
    exercised_stock_options      101 non-null float64
    expenses                     94 non-null float64
    from_messages                86 non-null float64
    from_poi_to_this_person      86 non-null float64
    from_this_person_to_poi      86 non-null float64
    loan_advances                3 non-null float64
    long_term_incentive          65 non-null float64
    other                        92 non-null float64
    poi                          145 non-null object
    restricted_stock             109 non-null float64
    restricted_stock_deferred    17 non-null float64
    salary                       94 non-null float64
    shared_receipt_with_poi      86 non-null float64
    to_messages                  86 non-null float64
    total_payments               124 non-null float64
    total_stock_value            125 non-null float64
    fraction_emails_to_poi       86 non-null float64
    fraction_emails_from_poi     86 non-null float64
    dtypes: float64(21), object(3)
    memory usage: 28.3+ KB
    


```python
# Adding 'fraction_emails_to_poi' to enron dictionary

for name in enron_dict:
    
    enron_dict[name]['fraction_emails_to_poi'] = float(enron_dict[name]['from_this_person_to_poi'])/(float(enron_dict[name]['to_messages']) + (float(enron_dict[name]['from_this_person_to_poi'])))
    enron_dict[name]['fraction_emails_from_poi'] = float(enron_dict[name]['from_poi_to_this_person'])/(float(enron_dict[name]['from_poi_to_this_person']) + (float(enron_dict[name]['from_messages'])))
    
printing.pprint(enron_dict['BLACHMAN JEREMY M'])
```

    {'bonus': 850000,
     'deferral_payments': 'NaN',
     'deferred_income': 'NaN',
     'director_fees': 'NaN',
     'email_address': 'jeremy.blachman@enron.com',
     'exercised_stock_options': 765313,
     'expenses': 84208,
     'fraction_emails_from_poi': 0.6410256410256411,
     'fraction_emails_to_poi': 0.0008074283407347598,
     'from_messages': 14,
     'from_poi_to_this_person': 25,
     'from_this_person_to_poi': 2,
     'loan_advances': 'NaN',
     'long_term_incentive': 831809,
     'other': 272,
     'poi': False,
     'restricted_stock': 189041,
     'restricted_stock_deferred': 'NaN',
     'salary': 248546,
     'shared_receipt_with_poi': 2326,
     'to_messages': 2475,
     'total_payments': 2014835,
     'total_stock_value': 954354}
    


```python
# Boxplot for out "fraction_emails_to_poi" feature

sns.boxplot(x='poi',y='fraction_emails_from_poi',data= enron_df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0xd05ce08>




![png](output_30_1.png)


- It seems that the email percentage sent to PoI from other POIs is higher than the percentage sent by non-POIs.  This could be attributed to most of the POIs being upper management and above (indicating they would work closely with eachother).  Regardless, I will include this in the list of features for the algorithm.


```python
# Boxplot for out "fraction_emails_from_poi" feature

sns.boxplot(x='poi',y='fraction_emails_to_poi',data= enron_df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0xd1fac48>




![png](output_32_1.png)


- There does not seem to be much of a difference when it comes to the emails sent from a POI.  This will not be used to train our machine learning algorithm.

<a id='missingvalues'></a>
### Missing Values

- Now, let's take a look at the features that are missing values.  Depending on the number of missing values in the respective features we may choose to omit them from the dataset.


```python
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 50)

enron_df.isnull().sum()
```




    name                           0
    bonus                         64
    deferral_payments            107
    deferred_income               97
    director_fees                129
    email_address                  0
    exercised_stock_options       44
    expenses                      51
    from_messages                 59
    from_poi_to_this_person       59
    from_this_person_to_poi       59
    loan_advances                142
    long_term_incentive           80
    other                         53
    poi                            0
    restricted_stock              36
    restricted_stock_deferred    128
    salary                        51
    shared_receipt_with_poi       59
    to_messages                   59
    total_payments                21
    total_stock_value             20
    fraction_emails_to_poi        59
    fraction_emails_from_poi      59
    dtype: int64



- From the dataframe info listed above, we can see several features are missing a lot of information.  We should as much info out as we feel comfortable with to cut down on any additional noise they would provide.  Such attributes I recommend removing are:

    * email_address
    * deferral_payments
    * loan_advances
    * restricted_stock_deferred
    * director_fees

</br>

- After looking at the data I also recommend removing the rows for Total, as this info can be calculated separately if need be.  We should also remove the info for Eugene Lockhart as all of his information is NaN.


```python
# Removing features

enron_df.drop(['email_address', 'deferral_payments', 'loan_advances',
               'restricted_stock_deferred', 'director_fees'], axis = 1, inplace = True)

# Replacing NaN values with 0.0
enron_df.fillna(0.0, inplace=True)

enron_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>bonus</th>
      <th>deferred_income</th>
      <th>exercised_stock_options</th>
      <th>expenses</th>
      <th>from_messages</th>
      <th>from_poi_to_this_person</th>
      <th>from_this_person_to_poi</th>
      <th>long_term_incentive</th>
      <th>other</th>
      <th>poi</th>
      <th>restricted_stock</th>
      <th>salary</th>
      <th>shared_receipt_with_poi</th>
      <th>to_messages</th>
      <th>total_payments</th>
      <th>total_stock_value</th>
      <th>fraction_emails_to_poi</th>
      <th>fraction_emails_from_poi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ALLEN PHILLIP K</td>
      <td>4175000.0</td>
      <td>-3081055.0</td>
      <td>1729541.0</td>
      <td>13868.0</td>
      <td>2195.0</td>
      <td>47.0</td>
      <td>65.0</td>
      <td>304805.0</td>
      <td>152.0</td>
      <td>False</td>
      <td>126027.0</td>
      <td>201955.0</td>
      <td>1407.0</td>
      <td>2902.0</td>
      <td>4484442.0</td>
      <td>1729541.0</td>
      <td>0.021908</td>
      <td>0.020963</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BADUM JAMES P</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>257817.0</td>
      <td>3486.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>182466.0</td>
      <td>257817.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BANNANTINE JAMES M</td>
      <td>0.0</td>
      <td>-5104.0</td>
      <td>4046157.0</td>
      <td>56301.0</td>
      <td>29.0</td>
      <td>39.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>864523.0</td>
      <td>False</td>
      <td>1757552.0</td>
      <td>477.0</td>
      <td>465.0</td>
      <td>566.0</td>
      <td>916197.0</td>
      <td>5243487.0</td>
      <td>0.000000</td>
      <td>0.573529</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BAXTER JOHN C</td>
      <td>1200000.0</td>
      <td>-1386055.0</td>
      <td>6680544.0</td>
      <td>11200.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1586055.0</td>
      <td>2660303.0</td>
      <td>False</td>
      <td>3942714.0</td>
      <td>267102.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5634343.0</td>
      <td>10623258.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BAY FRANKLIN R</td>
      <td>400000.0</td>
      <td>-201641.0</td>
      <td>0.0</td>
      <td>129142.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>69.0</td>
      <td>False</td>
      <td>145796.0</td>
      <td>239671.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>827696.0</td>
      <td>63014.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Taking dataframe and converting to dictionary for processing
enron_df.set_index('name', inplace = True)

enron_dict = enron_df.to_dict('index')

# Removing entry from dictionary because of lack of data
enron_dict.pop('LOCKHART EUGENE E',0)

#Removing outlier as this will skew data
enron_dict.pop('TOTAL',0)

```




    0




```python
# Creating features list

complete_features_list = enron_df.columns.tolist()

# Removing 'name' and moving 'poi' to the beginning of the list
complete_features_list.pop(0)

features_list = ['poi']

for n in complete_features_list:
    features_list.append(n)

printing.pprint(features_list)
```

    ['poi',
     'deferred_income',
     'exercised_stock_options',
     'expenses',
     'from_messages',
     'from_poi_to_this_person',
     'from_this_person_to_poi',
     'long_term_incentive',
     'other',
     'poi',
     'restricted_stock',
     'salary',
     'shared_receipt_with_poi',
     'to_messages',
     'total_payments',
     'total_stock_value',
     'fraction_emails_to_poi',
     'fraction_emails_from_poi']
    

<a id='selectingfeatures'></a>
### Selecting Features

- To decide on the best features to use for our project, I will run three lists created from the features I am the most interested in testing.  I will then use the list that has the best scores to run my final test on.  


```python
# Importing necessary feature metrics and classifiers

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from feature_format import featureFormat, targetFeatureSplit
```


```python
# Creating multiple lists of features for testing

list_one = ['poi','salary','bonus','fraction_emails_to_poi','exercised_stock_options']
list_two = ['poi','bonus','fraction_emails_to_poi','exercised_stock_options']
list_three = ['poi','salary','fraction_emails_to_poi','exercised_stock_options']
```


```python
# Creating test_class as to not repeat code for testing

def test_class(classifier, features_list, enron_dict):
    
    data = featureFormat(enron_dict, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, 
                                                                            test_size = 0.33, random_state = 42)
    
    
    clf = classifier
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    
    return {'Accuracy': accuracy_score(labels_test,pred),'Precision': precision_score(labels_test,pred),
            'Recall': recall_score(labels_test,pred)}
```

- <b> Validation:</b> We are using train_test_split method to split our data and make sure that 30% is left over for testing.


```python
# Attempting with full list of features

print("Naive Bayes stats: ", test_class(GaussianNB(), features_list, enron_dict))
print("Decision Tree stats: ", test_class(DecisionTreeClassifier(), features_list, enron_dict))
print("K Nearest Neighbors: ", test_class(KNeighborsClassifier(n_neighbors=3), features_list, enron_dict))
```

    ('Naive Bayes stats: ', {'Recall': 0.4, 'Precision': 0.6666666666666666, 'Accuracy': 0.9166666666666666})
    ('Decision Tree stats: ', {'Recall': 1.0, 'Precision': 1.0, 'Accuracy': 1.0})
    ('K Nearest Neighbors: ', {'Recall': 0.4, 'Precision': 0.6666666666666666, 'Accuracy': 0.9166666666666666})
    


```python
# Attempting with first list of features

print("Test List One: ", list_one)
print("Naive Bayes stats: ", test_class(GaussianNB(), list_one, enron_dict))
print("Decision Tree stats: ", test_class(DecisionTreeClassifier(), list_one, enron_dict))
print("K Nearest Neighbors: ", test_class(KNeighborsClassifier(n_neighbors=3), list_one, enron_dict))
```

    ('Test List One: ', ['poi', 'salary', 'bonus', 'fraction_emails_to_poi', 'exercised_stock_options'])
    ('Naive Bayes stats: ', {'Recall': 0.0, 'Precision': 0.0, 'Accuracy': 0.8372093023255814})
    ('Decision Tree stats: ', {'Recall': 0.5, 'Precision': 0.2857142857142857, 'Accuracy': 0.8372093023255814})
    ('K Nearest Neighbors: ', {'Recall': 0.5, 'Precision': 0.5, 'Accuracy': 0.9069767441860465})
    


```python
# Attempting with second list of features

print("Test List Two: ", list_two)
print("Naive Bayes stats: ", test_class(GaussianNB(), list_two, enron_dict))
print("Decision Tree stats: ", test_class(DecisionTreeClassifier(), list_two, enron_dict))
print("K Nearest Neighbors: ", test_class(KNeighborsClassifier(n_neighbors=3), list_two, enron_dict))
```

    ('Test List Two: ', ['poi', 'bonus', 'fraction_emails_to_poi', 'exercised_stock_options'])
    ('Naive Bayes stats: ', {'Recall': 0.25, 'Precision': 0.3333333333333333, 'Accuracy': 0.8809523809523809})
    ('Decision Tree stats: ', {'Recall': 0.5, 'Precision': 0.2, 'Accuracy': 0.7619047619047619})
    ('K Nearest Neighbors: ', {'Recall': 0.5, 'Precision': 0.5, 'Accuracy': 0.9047619047619048})
    


```python
# Attempting with third list of features

print("Test List Three: ", list_three)
print("Naive Bayes stats: ", test_class(GaussianNB(), list_three, enron_dict))
print("Decision Tree stats: ", test_class(DecisionTreeClassifier(), list_three, enron_dict))
print("K Nearest Neighbors: ", test_class(KNeighborsClassifier(n_neighbors=3), list_three, enron_dict))
```

    ('Test List Three: ', ['poi', 'salary', 'fraction_emails_to_poi', 'exercised_stock_options'])
    ('Naive Bayes stats: ', {'Recall': 0.25, 'Precision': 0.25, 'Accuracy': 0.8604651162790697})
    ('Decision Tree stats: ', {'Recall': 0.0, 'Precision': 0.0, 'Accuracy': 0.7674418604651163})
    ('K Nearest Neighbors: ', {'Recall': 0.0, 'Precision': 0.0, 'Accuracy': 0.8372093023255814})
    


```python
print("K Nearest Neighbors: ", test_class(KNeighborsClassifier(n_neighbors=3), list_one, enron_dict))
print("K Nearest Neighbors: ", test_class(KNeighborsClassifier(n_neighbors=2), list_one, enron_dict))
print("K Nearest Neighbors: ", test_class(KNeighborsClassifier(n_neighbors=4), list_one, enron_dict))
print("K Nearest Neighbors: ", test_class(KNeighborsClassifier(n_neighbors=5), list_one, enron_dict))
```

    ('K Nearest Neighbors: ', {'Recall': 0.5, 'Precision': 0.5, 'Accuracy': 0.9069767441860465})
    ('K Nearest Neighbors: ', {'Recall': 0.5, 'Precision': 0.5, 'Accuracy': 0.9069767441860465})
    ('K Nearest Neighbors: ', {'Recall': 0.5, 'Precision': 1.0, 'Accuracy': 0.9534883720930233})
    ('K Nearest Neighbors: ', {'Recall': 0.5, 'Precision': 1.0, 'Accuracy': 0.9534883720930233})
    

- Based on the results printed from the various combinations of feature lists and algorithms, I have decided to use the <u><b>K Nearest Neighbors</b></u> algorithm with feautre list number <b>ONE</b>:
    * <b>'poi'
    * <b>'salary'
    * <b>'bonus'
    * <b>'fraction_emails_to_poi'
    * <b>'exercised_stock_options'</b>
    

<a id='parametertuning'></a>
### Parameter Tuning

- After choosing the correct algorithm and feature list, I tried tuning the parameters of the algorithm.  Mainly, the <b>n_neighbors</b> parameter.  Upon adjustment, the accuracy and precision scores did seem to get a slight boost by changing the n_neighbors value from it's default 3 to 4.  4 and 5 n_neighbors had the same score so I decided to just use 4.

</br>

- Here is the comparison:


```python
# Comparison of n_neighbors parameter 3 vs 4

print("K Nearest Neighbors (3): ", test_class(KNeighborsClassifier(n_neighbors=3), list_one, enron_dict))
print("K Nearest Neighbors (4): ", test_class(KNeighborsClassifier(n_neighbors=4), list_one, enron_dict))
```

    ('K Nearest Neighbors (3): ', {'Recall': 0.5, 'Precision': 0.5, 'Accuracy': 0.9069767441860465})
    ('K Nearest Neighbors (4): ', {'Recall': 0.5, 'Precision': 1.0, 'Accuracy': 0.9534883720930233})
    


```python
# The winning combination of our project

print('Features List chosen: ', list_one)
print("K Nearest Neighbors: ", test_class(KNeighborsClassifier(n_neighbors=4), list_one, enron_dict))
```

    ('Features List chosen: ', ['poi', 'salary', 'bonus', 'fraction_emails_to_poi', 'exercised_stock_options'])
    ('K Nearest Neighbors: ', {'Recall': 0.5, 'Precision': 1.0, 'Accuracy': 0.9534883720930233})
    

- One thing to note is that I originally ran the features lists with 'total_stock_options' in place of 'exercised_stock_options'.  This resulted in many of the test runs having slightly lower accuracies.  But, more importanly, the precision and recall scores were consistently in the range of .2-.3

- Upon changing this one feature, we were able to achieve our current scores of:
    - Accuracy: 89.7%
    - Precision Score: 60%
    - Recall Score: 60%


```python
%run poi_id.py
```

    {0L: {'salary': 201955.0, 'to_messages': 2902.0, 'name': 'ALLEN PHILLIP K', 'long_term_incentive': 304805.0, 'bonus': 4175000.0, 'fraction_emails_to_poi': 0.021907650825749917, 'total_stock_value': 1729541.0, 'expenses': 13868.0, 'exercised_stock_options': 1729541.0, 'from_messages': 2195.0, 'other': 152.0, 'from_this_person_to_poi': 65.0, 'poi': False, 'total_payments': 4484442.0, 'deferred_income': -3081055.0, 'shared_receipt_with_poi': 1407.0, 'restricted_stock': 126027.0, 'from_poi_to_this_person': 47.0, 'fraction_emails_from_poi': 0.02096342551293488}, 1L: {'salary': 0.0, 'to_messages': 0.0, 'name': 'BADUM JAMES P', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 257817.0, 'expenses': 3486.0, 'exercised_stock_options': 257817.0, 'from_messages': 0.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 182466.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 0.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 2L: {'salary': 477.0, 'to_messages': 566.0, 'name': 'BANNANTINE JAMES M', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 5243487.0, 'expenses': 56301.0, 'exercised_stock_options': 4046157.0, 'from_messages': 29.0, 'other': 864523.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 916197.0, 'deferred_income': -5104.0, 'shared_receipt_with_poi': 465.0, 'restricted_stock': 1757552.0, 'from_poi_to_this_person': 39.0, 'fraction_emails_from_poi': 0.5735294117647058}, 3L: {'salary': 267102.0, 'to_messages': 0.0, 'name': 'BAXTER JOHN C', 'long_term_incentive': 1586055.0, 'bonus': 1200000.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 10623258.0, 'expenses': 11200.0, 'exercised_stock_options': 6680544.0, 'from_messages': 0.0, 'other': 2660303.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 5634343.0, 'deferred_income': -1386055.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 3942714.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 4L: {'salary': 239671.0, 'to_messages': 0.0, 'name': 'BAY FRANKLIN R', 'long_term_incentive': 0.0, 'bonus': 400000.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 63014.0, 'expenses': 129142.0, 'exercised_stock_options': 0.0, 'from_messages': 0.0, 'other': 69.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 827696.0, 'deferred_income': -201641.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 145796.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 5L: {'salary': 80818.0, 'to_messages': 0.0, 'name': 'BAZELIDES PHILIP J', 'long_term_incentive': 93750.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 1599641.0, 'expenses': 0.0, 'exercised_stock_options': 1599641.0, 'from_messages': 0.0, 'other': 874.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 860136.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 0.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 6L: {'salary': 231330.0, 'to_messages': 7315.0, 'name': 'BECK SALLY W', 'long_term_incentive': 0.0, 'bonus': 700000.0, 'fraction_emails_to_poi': 0.05012336060251915, 'total_stock_value': 126027.0, 'expenses': 37172.0, 'exercised_stock_options': 0.0, 'from_messages': 4343.0, 'other': 566.0, 'from_this_person_to_poi': 386.0, 'poi': False, 'total_payments': 969068.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 2639.0, 'restricted_stock': 126027.0, 'from_poi_to_this_person': 144.0, 'fraction_emails_from_poi': 0.03209271227991977}, 7L: {'salary': 213999.0, 'to_messages': 7991.0, 'name': 'BELDEN TIMOTHY N', 'long_term_incentive': 0.0, 'bonus': 5249999.0, 'fraction_emails_to_poi': 0.013334979627114458, 'total_stock_value': 1110705.0, 'expenses': 17355.0, 'exercised_stock_options': 953136.0, 'from_messages': 484.0, 'other': 210698.0, 'from_this_person_to_poi': 108.0, 'poi': True, 'total_payments': 5501630.0, 'deferred_income': -2334434.0, 'shared_receipt_with_poi': 5521.0, 'restricted_stock': 157569.0, 'from_poi_to_this_person': 228.0, 'fraction_emails_from_poi': 0.3202247191011236}, 8L: {'salary': 0.0, 'to_messages': 0.0, 'name': 'BELFER ROBERT', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': -44093.0, 'expenses': 0.0, 'exercised_stock_options': 3285.0, 'from_messages': 0.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 102500.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 0.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 9L: {'salary': 216582.0, 'to_messages': 0.0, 'name': 'BERBERIAN DAVID', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 2493616.0, 'expenses': 11892.0, 'exercised_stock_options': 1624396.0, 'from_messages': 0.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 228474.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 869220.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 10L: {'salary': 187922.0, 'to_messages': 383.0, 'name': 'BERGSIEKER RICHARD P', 'long_term_incentive': 180250.0, 'bonus': 250000.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 659249.0, 'expenses': 59175.0, 'exercised_stock_options': 0.0, 'from_messages': 59.0, 'other': 427316.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 618850.0, 'deferred_income': -485813.0, 'shared_receipt_with_poi': 233.0, 'restricted_stock': 659249.0, 'from_poi_to_this_person': 4.0, 'fraction_emails_from_poi': 0.06349206349206349}, 11L: {'salary': 0.0, 'to_messages': 523.0, 'name': 'BHATNAGAR SANJAY', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0019083969465648854, 'total_stock_value': 0.0, 'expenses': 0.0, 'exercised_stock_options': 2604490.0, 'from_messages': 29.0, 'other': 137864.0, 'from_this_person_to_poi': 1.0, 'poi': False, 'total_payments': 15456290.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 463.0, 'restricted_stock': -2604490.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 12L: {'salary': 213625.0, 'to_messages': 1607.0, 'name': 'BIBI PHILIPPE A', 'long_term_incentive': 369721.0, 'bonus': 1000000.0, 'fraction_emails_to_poi': 0.004953560371517028, 'total_stock_value': 1843816.0, 'expenses': 38559.0, 'exercised_stock_options': 1465734.0, 'from_messages': 40.0, 'other': 425688.0, 'from_this_person_to_poi': 8.0, 'poi': False, 'total_payments': 2047593.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 1336.0, 'restricted_stock': 378082.0, 'from_poi_to_this_person': 23.0, 'fraction_emails_from_poi': 0.36507936507936506}, 13L: {'salary': 248546.0, 'to_messages': 2475.0, 'name': 'BLACHMAN JEREMY M', 'long_term_incentive': 831809.0, 'bonus': 850000.0, 'fraction_emails_to_poi': 0.0008074283407347598, 'total_stock_value': 954354.0, 'expenses': 84208.0, 'exercised_stock_options': 765313.0, 'from_messages': 14.0, 'other': 272.0, 'from_this_person_to_poi': 2.0, 'poi': False, 'total_payments': 2014835.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 2326.0, 'restricted_stock': 189041.0, 'from_poi_to_this_person': 25.0, 'fraction_emails_from_poi': 0.6410256410256411}, 14L: {'salary': 0.0, 'to_messages': 0.0, 'name': 'BLAKE JR. NORMAN P', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 0.0, 'expenses': 1279.0, 'exercised_stock_options': 0.0, 'from_messages': 0.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 1279.0, 'deferred_income': -113784.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 0.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 15L: {'salary': 278601.0, 'to_messages': 1858.0, 'name': 'BOWEN JR RAYMOND M', 'long_term_incentive': 974293.0, 'bonus': 1350000.0, 'fraction_emails_to_poi': 0.00800854244527496, 'total_stock_value': 252055.0, 'expenses': 65907.0, 'exercised_stock_options': 0.0, 'from_messages': 27.0, 'other': 1621.0, 'from_this_person_to_poi': 15.0, 'poi': True, 'total_payments': 2669589.0, 'deferred_income': -833.0, 'shared_receipt_with_poi': 1593.0, 'restricted_stock': 252055.0, 'from_poi_to_this_person': 140.0, 'fraction_emails_from_poi': 0.8383233532934131}, 16L: {'salary': 0.0, 'to_messages': 1486.0, 'name': 'BROWN MICHAEL', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0006724949562878278, 'total_stock_value': 0.0, 'expenses': 49288.0, 'exercised_stock_options': 0.0, 'from_messages': 41.0, 'other': 0.0, 'from_this_person_to_poi': 1.0, 'poi': False, 'total_payments': 49288.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 761.0, 'restricted_stock': 0.0, 'from_poi_to_this_person': 13.0, 'fraction_emails_from_poi': 0.24074074074074073}, 17L: {'salary': 248017.0, 'to_messages': 1088.0, 'name': 'BUCHANAN HAROLD G', 'long_term_incentive': 304805.0, 'bonus': 500000.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 1014505.0, 'expenses': 600.0, 'exercised_stock_options': 825464.0, 'from_messages': 125.0, 'other': 1215.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 1054637.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 23.0, 'restricted_stock': 189041.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 18L: {'salary': 261516.0, 'to_messages': 0.0, 'name': 'BUTTS ROBERT H', 'long_term_incentive': 175000.0, 'bonus': 750000.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 417619.0, 'expenses': 9410.0, 'exercised_stock_options': 0.0, 'from_messages': 0.0, 'other': 150656.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 1271582.0, 'deferred_income': -75000.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 417619.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 19L: {'salary': 330546.0, 'to_messages': 3523.0, 'name': 'BUY RICHARD B', 'long_term_incentive': 769862.0, 'bonus': 900000.0, 'fraction_emails_to_poi': 0.019755147468002224, 'total_stock_value': 3444470.0, 'expenses': 0.0, 'exercised_stock_options': 2542813.0, 'from_messages': 1053.0, 'other': 400572.0, 'from_this_person_to_poi': 71.0, 'poi': False, 'total_payments': 2355702.0, 'deferred_income': -694862.0, 'shared_receipt_with_poi': 2333.0, 'restricted_stock': 901657.0, 'from_poi_to_this_person': 156.0, 'fraction_emails_from_poi': 0.12903225806451613}, 20L: {'salary': 240189.0, 'to_messages': 2598.0, 'name': 'CALGER CHRISTOPHER F', 'long_term_incentive': 375304.0, 'bonus': 1250000.0, 'fraction_emails_to_poi': 0.009531071292413268, 'total_stock_value': 126027.0, 'expenses': 35818.0, 'exercised_stock_options': 0.0, 'from_messages': 144.0, 'other': 486.0, 'from_this_person_to_poi': 25.0, 'poi': True, 'total_payments': 1639297.0, 'deferred_income': -262500.0, 'shared_receipt_with_poi': 2188.0, 'restricted_stock': 126027.0, 'from_poi_to_this_person': 199.0, 'fraction_emails_from_poi': 0.5801749271137027}, 21L: {'salary': 261809.0, 'to_messages': 312.0, 'name': 'CARTER REBECCA C', 'long_term_incentive': 75000.0, 'bonus': 300000.0, 'fraction_emails_to_poi': 0.0219435736677116, 'total_stock_value': 0.0, 'expenses': 0.0, 'exercised_stock_options': 0.0, 'from_messages': 15.0, 'other': 540.0, 'from_this_person_to_poi': 7.0, 'poi': False, 'total_payments': 477557.0, 'deferred_income': -159792.0, 'shared_receipt_with_poi': 196.0, 'restricted_stock': 307301.0, 'from_poi_to_this_person': 29.0, 'fraction_emails_from_poi': 0.6590909090909091}, 22L: {'salary': 415189.0, 'to_messages': 1892.0, 'name': 'CAUSEY RICHARD A', 'long_term_incentive': 350000.0, 'bonus': 1000000.0, 'fraction_emails_to_poi': 0.0063025210084033615, 'total_stock_value': 2502063.0, 'expenses': 30674.0, 'exercised_stock_options': 0.0, 'from_messages': 49.0, 'other': 307895.0, 'from_this_person_to_poi': 12.0, 'poi': True, 'total_payments': 1868758.0, 'deferred_income': -235000.0, 'shared_receipt_with_poi': 1585.0, 'restricted_stock': 2502063.0, 'from_poi_to_this_person': 58.0, 'fraction_emails_from_poi': 0.5420560747663551}, 23L: {'salary': 0.0, 'to_messages': 0.0, 'name': 'CHAN RONNIE', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 0.0, 'expenses': 0.0, 'exercised_stock_options': 0.0, 'from_messages': 0.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 0.0, 'deferred_income': -98784.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 32460.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 24L: {'salary': 0.0, 'to_messages': 0.0, 'name': 'CHRISTODOULOU DIOMEDES', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 6077885.0, 'expenses': 0.0, 'exercised_stock_options': 5127155.0, 'from_messages': 0.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 0.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 950730.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 25L: {'salary': 0.0, 'to_messages': 0.0, 'name': 'CLINE KENNETH W', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 189518.0, 'expenses': 0.0, 'exercised_stock_options': 0.0, 'from_messages': 0.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 0.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 662086.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 26L: {'salary': 288542.0, 'to_messages': 1758.0, 'name': 'COLWELL WESLEY', 'long_term_incentive': 0.0, 'bonus': 1200000.0, 'fraction_emails_to_poi': 0.006218202374222725, 'total_stock_value': 698242.0, 'expenses': 16514.0, 'exercised_stock_options': 0.0, 'from_messages': 40.0, 'other': 101740.0, 'from_this_person_to_poi': 11.0, 'poi': True, 'total_payments': 1490344.0, 'deferred_income': -144062.0, 'shared_receipt_with_poi': 1132.0, 'restricted_stock': 698242.0, 'from_poi_to_this_person': 240.0, 'fraction_emails_from_poi': 0.8571428571428571}, 27L: {'salary': 0.0, 'to_messages': 764.0, 'name': 'CORDES WILLIAM R', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 1038185.0, 'expenses': 0.0, 'exercised_stock_options': 651850.0, 'from_messages': 12.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 0.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 58.0, 'restricted_stock': 386335.0, 'from_poi_to_this_person': 10.0, 'fraction_emails_from_poi': 0.45454545454545453}, 28L: {'salary': 314288.0, 'to_messages': 102.0, 'name': 'COX DAVID', 'long_term_incentive': 0.0, 'bonus': 800000.0, 'fraction_emails_to_poi': 0.03773584905660377, 'total_stock_value': 495633.0, 'expenses': 27861.0, 'exercised_stock_options': 117551.0, 'from_messages': 33.0, 'other': 494.0, 'from_this_person_to_poi': 4.0, 'poi': False, 'total_payments': 1101393.0, 'deferred_income': -41250.0, 'shared_receipt_with_poi': 71.0, 'restricted_stock': 378082.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 29L: {'salary': 184899.0, 'to_messages': 0.0, 'name': 'CUMBERLAND MICHAEL S', 'long_term_incentive': 275000.0, 'bonus': 325000.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 207940.0, 'expenses': 22344.0, 'exercised_stock_options': 0.0, 'from_messages': 0.0, 'other': 713.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 807956.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 207940.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 30L: {'salary': 206121.0, 'to_messages': 714.0, 'name': 'DEFFNER JOSEPH M', 'long_term_incentive': 335349.0, 'bonus': 600000.0, 'fraction_emails_to_poi': 0.005571030640668524, 'total_stock_value': 159211.0, 'expenses': 41626.0, 'exercised_stock_options': 17378.0, 'from_messages': 74.0, 'other': 25553.0, 'from_this_person_to_poi': 4.0, 'poi': False, 'total_payments': 1208649.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 552.0, 'restricted_stock': 141833.0, 'from_poi_to_this_person': 115.0, 'fraction_emails_from_poi': 0.6084656084656085}, 31L: {'salary': 365163.0, 'to_messages': 3093.0, 'name': 'DELAINEY DAVID W', 'long_term_incentive': 1294981.0, 'bonus': 3000000.0, 'fraction_emails_to_poi': 0.16450567260940033, 'total_stock_value': 3614261.0, 'expenses': 86174.0, 'exercised_stock_options': 2291113.0, 'from_messages': 3069.0, 'other': 1661.0, 'from_this_person_to_poi': 609.0, 'poi': True, 'total_payments': 4747979.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 2097.0, 'restricted_stock': 1323148.0, 'from_poi_to_this_person': 66.0, 'fraction_emails_from_poi': 0.021052631578947368}, 32L: {'salary': 492375.0, 'to_messages': 2181.0, 'name': 'DERRICK JR. JAMES V', 'long_term_incentive': 484000.0, 'bonus': 800000.0, 'fraction_emails_to_poi': 0.009086778736937756, 'total_stock_value': 8831913.0, 'expenses': 51124.0, 'exercised_stock_options': 8831913.0, 'from_messages': 909.0, 'other': 7482.0, 'from_this_person_to_poi': 20.0, 'poi': False, 'total_payments': 550981.0, 'deferred_income': -1284000.0, 'shared_receipt_with_poi': 1401.0, 'restricted_stock': 1787380.0, 'from_poi_to_this_person': 64.0, 'fraction_emails_from_poi': 0.065775950668037}, 33L: {'salary': 210500.0, 'to_messages': 0.0, 'name': 'DETMERING TIMOTHY J', 'long_term_incentive': 415657.0, 'bonus': 425000.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 2027865.0, 'expenses': 52255.0, 'exercised_stock_options': 2027865.0, 'from_messages': 0.0, 'other': 1105.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 1204583.0, 'deferred_income': -775241.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 315068.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 34L: {'salary': 250100.0, 'to_messages': 2572.0, 'name': 'DIETRICH JANET R', 'long_term_incentive': 556416.0, 'bonus': 600000.0, 'fraction_emails_to_poi': 0.005413766434648105, 'total_stock_value': 1865087.0, 'expenses': 3475.0, 'exercised_stock_options': 1550019.0, 'from_messages': 63.0, 'other': 473.0, 'from_this_person_to_poi': 14.0, 'poi': False, 'total_payments': 1410464.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 1902.0, 'restricted_stock': 315068.0, 'from_poi_to_this_person': 305.0, 'fraction_emails_from_poi': 0.8288043478260869}, 35L: {'salary': 262788.0, 'to_messages': 0.0, 'name': 'DIMICHELE RICHARD G', 'long_term_incentive': 694862.0, 'bonus': 1000000.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 8317782.0, 'expenses': 35812.0, 'exercised_stock_options': 8191755.0, 'from_messages': 0.0, 'other': 374689.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 2368151.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 126027.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 36L: {'salary': 221003.0, 'to_messages': 176.0, 'name': 'DODSON KEITH', 'long_term_incentive': 0.0, 'bonus': 70000.0, 'fraction_emails_to_poi': 0.01675977653631285, 'total_stock_value': 0.0, 'expenses': 28164.0, 'exercised_stock_options': 0.0, 'from_messages': 14.0, 'other': 774.0, 'from_this_person_to_poi': 3.0, 'poi': False, 'total_payments': 319941.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 114.0, 'restricted_stock': 0.0, 'from_poi_to_this_person': 10.0, 'fraction_emails_from_poi': 0.4166666666666667}, 37L: {'salary': 278601.0, 'to_messages': 865.0, 'name': 'DONAHUE JR JEFFREY M', 'long_term_incentive': 0.0, 'bonus': 800000.0, 'fraction_emails_to_poi': 0.012557077625570776, 'total_stock_value': 1080988.0, 'expenses': 96268.0, 'exercised_stock_options': 765920.0, 'from_messages': 22.0, 'other': 891.0, 'from_this_person_to_poi': 11.0, 'poi': False, 'total_payments': 875760.0, 'deferred_income': -300000.0, 'shared_receipt_with_poi': 772.0, 'restricted_stock': 315068.0, 'from_poi_to_this_person': 188.0, 'fraction_emails_from_poi': 0.8952380952380953}, 38L: {'salary': 0.0, 'to_messages': 0.0, 'name': 'DUNCAN JOHN H', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 371750.0, 'expenses': 0.0, 'exercised_stock_options': 371750.0, 'from_messages': 0.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 77492.0, 'deferred_income': -25000.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 0.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 39L: {'salary': 210692.0, 'to_messages': 904.0, 'name': 'DURAN WILLIAM D', 'long_term_incentive': 1105218.0, 'bonus': 750000.0, 'fraction_emails_to_poi': 0.0033076074972436605, 'total_stock_value': 1640910.0, 'expenses': 25785.0, 'exercised_stock_options': 1451869.0, 'from_messages': 12.0, 'other': 1568.0, 'from_this_person_to_poi': 3.0, 'poi': False, 'total_payments': 2093263.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 599.0, 'restricted_stock': 189041.0, 'from_poi_to_this_person': 106.0, 'fraction_emails_from_poi': 0.8983050847457628}, 40L: {'salary': 182245.0, 'to_messages': 0.0, 'name': 'ECHOLS JOHN B', 'long_term_incentive': 2234774.0, 'bonus': 200000.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 1008941.0, 'expenses': 21530.0, 'exercised_stock_options': 601438.0, 'from_messages': 0.0, 'other': 53775.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 2692324.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 407503.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 41L: {'salary': 170941.0, 'to_messages': 0.0, 'name': 'ELLIOTT STEVEN', 'long_term_incentive': 0.0, 'bonus': 350000.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 6678735.0, 'expenses': 78552.0, 'exercised_stock_options': 4890344.0, 'from_messages': 0.0, 'other': 12961.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 211725.0, 'deferred_income': -400729.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 1788391.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 42L: {'salary': 304588.0, 'to_messages': 1755.0, 'name': 'FALLON JAMES B', 'long_term_incentive': 374347.0, 'bonus': 2500000.0, 'fraction_emails_to_poi': 0.020647321428571428, 'total_stock_value': 2332399.0, 'expenses': 95924.0, 'exercised_stock_options': 940257.0, 'from_messages': 75.0, 'other': 401481.0, 'from_this_person_to_poi': 37.0, 'poi': False, 'total_payments': 3676340.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 1604.0, 'restricted_stock': 1392142.0, 'from_poi_to_this_person': 42.0, 'fraction_emails_from_poi': 0.358974358974359}, 43L: {'salary': 440698.0, 'to_messages': 0.0, 'name': 'FASTOW ANDREW S', 'long_term_incentive': 1736055.0, 'bonus': 1300000.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 1794412.0, 'expenses': 55921.0, 'exercised_stock_options': 0.0, 'from_messages': 0.0, 'other': 277464.0, 'from_this_person_to_poi': 0.0, 'poi': True, 'total_payments': 2424083.0, 'deferred_income': -1386055.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 1794412.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 44L: {'salary': 199157.0, 'to_messages': 936.0, 'name': 'FITZGERALD JAY L', 'long_term_incentive': 556416.0, 'bonus': 350000.0, 'fraction_emails_to_poi': 0.00847457627118644, 'total_stock_value': 1621236.0, 'expenses': 23870.0, 'exercised_stock_options': 664461.0, 'from_messages': 16.0, 'other': 285414.0, 'from_this_person_to_poi': 8.0, 'poi': False, 'total_payments': 1414857.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 723.0, 'restricted_stock': 956775.0, 'from_poi_to_this_person': 1.0, 'fraction_emails_from_poi': 0.058823529411764705}, 45L: {'salary': 0.0, 'to_messages': 517.0, 'name': 'FOWLER PEGGY', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 1884748.0, 'expenses': 0.0, 'exercised_stock_options': 1324578.0, 'from_messages': 36.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 0.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 10.0, 'restricted_stock': 560170.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 46L: {'salary': 0.0, 'to_messages': 57.0, 'name': 'FOY JOE', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 343434.0, 'expenses': 0.0, 'exercised_stock_options': 343434.0, 'from_messages': 13.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 181755.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 2.0, 'restricted_stock': 0.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 47L: {'salary': 1060932.0, 'to_messages': 3275.0, 'name': 'FREVERT MARK A', 'long_term_incentive': 1617011.0, 'bonus': 2000000.0, 'fraction_emails_to_poi': 0.001828710758914965, 'total_stock_value': 14622185.0, 'expenses': 86987.0, 'exercised_stock_options': 10433518.0, 'from_messages': 21.0, 'other': 7427621.0, 'from_this_person_to_poi': 6.0, 'poi': False, 'total_payments': 17252530.0, 'deferred_income': -3367011.0, 'shared_receipt_with_poi': 2979.0, 'restricted_stock': 4188667.0, 'from_poi_to_this_person': 242.0, 'fraction_emails_from_poi': 0.9201520912547528}, 48L: {'salary': 0.0, 'to_messages': 0.0, 'name': 'FUGH JOHN L', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 176378.0, 'expenses': 0.0, 'exercised_stock_options': 176378.0, 'from_messages': 0.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 50591.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 0.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 49L: {'salary': 192008.0, 'to_messages': 0.0, 'name': 'GAHN ROBERT S', 'long_term_incentive': 0.0, 'bonus': 509870.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 318607.0, 'expenses': 50080.0, 'exercised_stock_options': 83237.0, 'from_messages': 0.0, 'other': 76547.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 900585.0, 'deferred_income': -1042.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 235370.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 50L: {'salary': 231946.0, 'to_messages': 209.0, 'name': 'GARLAND C KEVIN', 'long_term_incentive': 375304.0, 'bonus': 850000.0, 'fraction_emails_to_poi': 0.11440677966101695, 'total_stock_value': 896153.0, 'expenses': 48405.0, 'exercised_stock_options': 636246.0, 'from_messages': 44.0, 'other': 60814.0, 'from_this_person_to_poi': 27.0, 'poi': False, 'total_payments': 1566469.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 178.0, 'restricted_stock': 259907.0, 'from_poi_to_this_person': 10.0, 'fraction_emails_from_poi': 0.18518518518518517}, 51L: {'salary': 0.0, 'to_messages': 0.0, 'name': 'GATHMANN WILLIAM D', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 1945360.0, 'expenses': 0.0, 'exercised_stock_options': 1753766.0, 'from_messages': 0.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 0.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 264013.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 52L: {'salary': 0.0, 'to_messages': 169.0, 'name': 'GIBBS DANA R', 'long_term_incentive': 461912.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 2218275.0, 'expenses': 0.0, 'exercised_stock_options': 2218275.0, 'from_messages': 12.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 966522.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 23.0, 'restricted_stock': 0.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 53L: {'salary': 0.0, 'to_messages': 0.0, 'name': 'GILLIS JOHN', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 85641.0, 'expenses': 0.0, 'exercised_stock_options': 9803.0, 'from_messages': 0.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 0.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 75838.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 54L: {'salary': 274975.0, 'to_messages': 873.0, 'name': 'GLISAN JR BEN F', 'long_term_incentive': 71023.0, 'bonus': 600000.0, 'fraction_emails_to_poi': 0.006825938566552901, 'total_stock_value': 778546.0, 'expenses': 125978.0, 'exercised_stock_options': 384728.0, 'from_messages': 16.0, 'other': 200308.0, 'from_this_person_to_poi': 6.0, 'poi': True, 'total_payments': 1272284.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 874.0, 'restricted_stock': 393818.0, 'from_poi_to_this_person': 52.0, 'fraction_emails_from_poi': 0.7647058823529411}, 55L: {'salary': 272880.0, 'to_messages': 0.0, 'name': 'GOLD JOSEPH', 'long_term_incentive': 304805.0, 'bonus': 750000.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 877611.0, 'expenses': 0.0, 'exercised_stock_options': 436515.0, 'from_messages': 0.0, 'other': 819288.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 2146973.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 441096.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 56L: {'salary': 0.0, 'to_messages': 0.0, 'name': 'GRAMM WENDY L', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 0.0, 'expenses': 0.0, 'exercised_stock_options': 0.0, 'from_messages': 0.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 119292.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 0.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 57L: {'salary': 6615.0, 'to_messages': 0.0, 'name': 'GRAY RODNEY', 'long_term_incentive': 365625.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 0.0, 'expenses': 0.0, 'exercised_stock_options': 0.0, 'from_messages': 0.0, 'other': 680833.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 1146658.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 0.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 58L: {'salary': 374125.0, 'to_messages': 4009.0, 'name': 'HAEDICKE MARK E', 'long_term_incentive': 983346.0, 'bonus': 1150000.0, 'fraction_emails_to_poi': 0.014987714987714987, 'total_stock_value': 803094.0, 'expenses': 76169.0, 'exercised_stock_options': 608750.0, 'from_messages': 1941.0, 'other': 52382.0, 'from_this_person_to_poi': 61.0, 'poi': False, 'total_payments': 3859065.0, 'deferred_income': -934484.0, 'shared_receipt_with_poi': 1847.0, 'restricted_stock': 524169.0, 'from_poi_to_this_person': 180.0, 'fraction_emails_from_poi': 0.08486562942008487}, 59L: {'salary': 243293.0, 'to_messages': 1045.0, 'name': 'HANNON KEVIN P', 'long_term_incentive': 1617011.0, 'bonus': 1500000.0, 'fraction_emails_to_poi': 0.019699812382739212, 'total_stock_value': 6391065.0, 'expenses': 34039.0, 'exercised_stock_options': 5538001.0, 'from_messages': 32.0, 'other': 11350.0, 'from_this_person_to_poi': 21.0, 'poi': True, 'total_payments': 288682.0, 'deferred_income': -3117011.0, 'shared_receipt_with_poi': 1035.0, 'restricted_stock': 853064.0, 'from_poi_to_this_person': 32.0, 'fraction_emails_from_poi': 0.5}, 60L: {'salary': 0.0, 'to_messages': 573.0, 'name': 'HAUG DAVID L', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.01206896551724138, 'total_stock_value': 2217299.0, 'expenses': 475.0, 'exercised_stock_options': 0.0, 'from_messages': 19.0, 'other': 0.0, 'from_this_person_to_poi': 7.0, 'poi': False, 'total_payments': 475.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 471.0, 'restricted_stock': 2217299.0, 'from_poi_to_this_person': 4.0, 'fraction_emails_from_poi': 0.17391304347826086}, 61L: {'salary': 0.0, 'to_messages': 504.0, 'name': 'HAYES ROBERT E', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 151418.0, 'expenses': 0.0, 'exercised_stock_options': 0.0, 'from_messages': 12.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 7961.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 50.0, 'restricted_stock': 151418.0, 'from_poi_to_this_person': 16.0, 'fraction_emails_from_poi': 0.5714285714285714}, 62L: {'salary': 0.0, 'to_messages': 2649.0, 'name': 'HAYSLETT RODERICK J', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.014142165984369185, 'total_stock_value': 346663.0, 'expenses': 0.0, 'exercised_stock_options': 0.0, 'from_messages': 1061.0, 'other': 0.0, 'from_this_person_to_poi': 38.0, 'poi': False, 'total_payments': 0.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 571.0, 'restricted_stock': 346663.0, 'from_poi_to_this_person': 35.0, 'fraction_emails_from_poi': 0.03193430656934307}, 63L: {'salary': 262663.0, 'to_messages': 0.0, 'name': 'HERMANN ROBERT J', 'long_term_incentive': 150000.0, 'bonus': 700000.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 668132.0, 'expenses': 48357.0, 'exercised_stock_options': 187500.0, 'from_messages': 0.0, 'other': 416441.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 1297461.0, 'deferred_income': -280000.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 480632.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 64L: {'salary': 211788.0, 'to_messages': 1320.0, 'name': 'HICKERSON GARY J', 'long_term_incentive': 69223.0, 'bonus': 1700000.0, 'fraction_emails_to_poi': 0.000757002271006813, 'total_stock_value': 441096.0, 'expenses': 98849.0, 'exercised_stock_options': 0.0, 'from_messages': 27.0, 'other': 1936.0, 'from_this_person_to_poi': 1.0, 'poi': False, 'total_payments': 2081796.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 900.0, 'restricted_stock': 441096.0, 'from_poi_to_this_person': 40.0, 'fraction_emails_from_poi': 0.5970149253731343}, 65L: {'salary': 0.0, 'to_messages': 0.0, 'name': 'HIRKO JOSEPH', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 30766064.0, 'expenses': 77978.0, 'exercised_stock_options': 30766064.0, 'from_messages': 0.0, 'other': 2856.0, 'from_this_person_to_poi': 0.0, 'poi': True, 'total_payments': 91093.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 0.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 66L: {'salary': 0.0, 'to_messages': 2350.0, 'name': 'HORTON STANLEY C', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.006342494714587738, 'total_stock_value': 7256648.0, 'expenses': 0.0, 'exercised_stock_options': 5210569.0, 'from_messages': 1073.0, 'other': 0.0, 'from_this_person_to_poi': 15.0, 'poi': False, 'total_payments': 3131860.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 1074.0, 'restricted_stock': 2046079.0, 'from_poi_to_this_person': 44.0, 'fraction_emails_from_poi': 0.03939122649955237}, 67L: {'salary': 0.0, 'to_messages': 719.0, 'name': 'HUGHES JAMES A', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.006906077348066298, 'total_stock_value': 1118394.0, 'expenses': 0.0, 'exercised_stock_options': 754966.0, 'from_messages': 34.0, 'other': 0.0, 'from_this_person_to_poi': 5.0, 'poi': False, 'total_payments': 0.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 589.0, 'restricted_stock': 363428.0, 'from_poi_to_this_person': 35.0, 'fraction_emails_from_poi': 0.5072463768115942}, 68L: {'salary': 130724.0, 'to_messages': 128.0, 'name': 'HUMPHREY GENE E', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.11724137931034483, 'total_stock_value': 2282768.0, 'expenses': 4994.0, 'exercised_stock_options': 2282768.0, 'from_messages': 17.0, 'other': 0.0, 'from_this_person_to_poi': 17.0, 'poi': False, 'total_payments': 3100224.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 119.0, 'restricted_stock': 0.0, 'from_poi_to_this_person': 10.0, 'fraction_emails_from_poi': 0.37037037037037035}, 69L: {'salary': 85274.0, 'to_messages': 496.0, 'name': 'IZZO LAWRENCE L', 'long_term_incentive': 312500.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.00998003992015968, 'total_stock_value': 5819980.0, 'expenses': 28093.0, 'exercised_stock_options': 2165172.0, 'from_messages': 19.0, 'other': 1553729.0, 'from_this_person_to_poi': 5.0, 'poi': False, 'total_payments': 1979596.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 437.0, 'restricted_stock': 3654808.0, 'from_poi_to_this_person': 28.0, 'fraction_emails_from_poi': 0.5957446808510638}, 70L: {'salary': 288558.0, 'to_messages': 258.0, 'name': 'JACKSON CHARLENE R', 'long_term_incentive': 0.0, 'bonus': 250000.0, 'fraction_emails_to_poi': 0.06859205776173286, 'total_stock_value': 725735.0, 'expenses': 10181.0, 'exercised_stock_options': 185063.0, 'from_messages': 56.0, 'other': 2435.0, 'from_this_person_to_poi': 19.0, 'poi': False, 'total_payments': 551174.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 117.0, 'restricted_stock': 540672.0, 'from_poi_to_this_person': 25.0, 'fraction_emails_from_poi': 0.30864197530864196}, 71L: {'salary': 0.0, 'to_messages': 0.0, 'name': 'JAEDICKE ROBERT', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 431750.0, 'expenses': 0.0, 'exercised_stock_options': 431750.0, 'from_messages': 0.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 83750.0, 'deferred_income': -25000.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 44093.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 72L: {'salary': 275101.0, 'to_messages': 4607.0, 'name': 'KAMINSKI WINCENTY J', 'long_term_incentive': 323466.0, 'bonus': 400000.0, 'fraction_emails_to_poi': 0.035789033068229385, 'total_stock_value': 976037.0, 'expenses': 83585.0, 'exercised_stock_options': 850010.0, 'from_messages': 14368.0, 'other': 4669.0, 'from_this_person_to_poi': 171.0, 'poi': False, 'total_payments': 1086821.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 583.0, 'restricted_stock': 126027.0, 'from_poi_to_this_person': 41.0, 'fraction_emails_from_poi': 0.0028454438198348255}, 73L: {'salary': 404338.0, 'to_messages': 12754.0, 'name': 'KEAN STEVEN J', 'long_term_incentive': 300000.0, 'bonus': 1000000.0, 'fraction_emails_to_poi': 0.029449813560611826, 'total_stock_value': 6153642.0, 'expenses': 41953.0, 'exercised_stock_options': 2022048.0, 'from_messages': 6759.0, 'other': 1231.0, 'from_this_person_to_poi': 387.0, 'poi': False, 'total_payments': 1747522.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 3639.0, 'restricted_stock': 4131594.0, 'from_poi_to_this_person': 140.0, 'fraction_emails_from_poi': 0.02029279605739962}, 74L: {'salary': 174246.0, 'to_messages': 0.0, 'name': 'KISHKILL JOSEPH G', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 1034346.0, 'expenses': 116335.0, 'exercised_stock_options': 0.0, 'from_messages': 0.0, 'other': 465357.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 704896.0, 'deferred_income': -51042.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 1034346.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 75L: {'salary': 271442.0, 'to_messages': 8305.0, 'name': 'KITCHEN LOUISE', 'long_term_incentive': 0.0, 'bonus': 3100000.0, 'fraction_emails_to_poi': 0.022826214848805742, 'total_stock_value': 547143.0, 'expenses': 5774.0, 'exercised_stock_options': 81042.0, 'from_messages': 1728.0, 'other': 93925.0, 'from_this_person_to_poi': 194.0, 'poi': False, 'total_payments': 3471141.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 3669.0, 'restricted_stock': 466101.0, 'from_poi_to_this_person': 251.0, 'fraction_emails_from_poi': 0.12683173319858515}, 76L: {'salary': 309946.0, 'to_messages': 2374.0, 'name': 'KOENIG MARK E', 'long_term_incentive': 300000.0, 'bonus': 700000.0, 'fraction_emails_to_poi': 0.006278777731268313, 'total_stock_value': 1920055.0, 'expenses': 127017.0, 'exercised_stock_options': 671737.0, 'from_messages': 61.0, 'other': 150458.0, 'from_this_person_to_poi': 15.0, 'poi': True, 'total_payments': 1587421.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 2271.0, 'restricted_stock': 1248318.0, 'from_poi_to_this_person': 53.0, 'fraction_emails_from_poi': 0.4649122807017544}, 77L: {'salary': 224305.0, 'to_messages': 0.0, 'name': 'KOPPER MICHAEL J', 'long_term_incentive': 602671.0, 'bonus': 800000.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 985032.0, 'expenses': 118134.0, 'exercised_stock_options': 0.0, 'from_messages': 0.0, 'other': 907502.0, 'from_this_person_to_poi': 0.0, 'poi': True, 'total_payments': 2652612.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 985032.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 78L: {'salary': 339288.0, 'to_messages': 7259.0, 'name': 'LAVORATO JOHN J', 'long_term_incentive': 2035380.0, 'bonus': 8000000.0, 'fraction_emails_to_poi': 0.05358539765319426, 'total_stock_value': 5167144.0, 'expenses': 49537.0, 'exercised_stock_options': 4158995.0, 'from_messages': 2585.0, 'other': 1552.0, 'from_this_person_to_poi': 411.0, 'poi': False, 'total_payments': 10425757.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 3962.0, 'restricted_stock': 1008149.0, 'from_poi_to_this_person': 528.0, 'fraction_emails_from_poi': 0.1696113074204947}, 79L: {'salary': 1072321.0, 'to_messages': 4273.0, 'name': 'LAY KENNETH L', 'long_term_incentive': 3600000.0, 'bonus': 7000000.0, 'fraction_emails_to_poi': 0.0037304733038004195, 'total_stock_value': 49110078.0, 'expenses': 99832.0, 'exercised_stock_options': 34348384.0, 'from_messages': 36.0, 'other': 10359729.0, 'from_this_person_to_poi': 16.0, 'poi': True, 'total_payments': 103559793.0, 'deferred_income': -300000.0, 'shared_receipt_with_poi': 2411.0, 'restricted_stock': 14761694.0, 'from_poi_to_this_person': 123.0, 'fraction_emails_from_poi': 0.7735849056603774}, 80L: {'salary': 273746.0, 'to_messages': 2822.0, 'name': 'LEFF DANIEL P', 'long_term_incentive': 1387399.0, 'bonus': 1000000.0, 'fraction_emails_to_poi': 0.004936530324400564, 'total_stock_value': 360528.0, 'expenses': 0.0, 'exercised_stock_options': 0.0, 'from_messages': 63.0, 'other': 3083.0, 'from_this_person_to_poi': 14.0, 'poi': False, 'total_payments': 2664228.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 2672.0, 'restricted_stock': 360528.0, 'from_poi_to_this_person': 67.0, 'fraction_emails_from_poi': 0.5153846153846153}, 81L: {'salary': 0.0, 'to_messages': 0.0, 'name': 'LEMAISTRE CHARLES', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 412878.0, 'expenses': 0.0, 'exercised_stock_options': 412878.0, 'from_messages': 0.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 87492.0, 'deferred_income': -25000.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 0.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 82L: {'salary': 0.0, 'to_messages': 952.0, 'name': 'LEWIS RICHARD', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 850477.0, 'expenses': 0.0, 'exercised_stock_options': 850477.0, 'from_messages': 26.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 0.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 739.0, 'restricted_stock': 0.0, 'from_poi_to_this_person': 10.0, 'fraction_emails_from_poi': 0.2777777777777778}, 83L: {'salary': 236457.0, 'to_messages': 0.0, 'name': 'LINDHOLM TOD A', 'long_term_incentive': 175000.0, 'bonus': 200000.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 3064208.0, 'expenses': 57727.0, 'exercised_stock_options': 2549361.0, 'from_messages': 0.0, 'other': 2630.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 875889.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 514847.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 84L: {'salary': 0.0, 'to_messages': 0.0, 'name': 'LOWRY CHARLES P', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 372205.0, 'expenses': 0.0, 'exercised_stock_options': 372205.0, 'from_messages': 0.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 0.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 153686.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 85L: {'salary': 349487.0, 'to_messages': 1522.0, 'name': 'MARTIN AMANDA K', 'long_term_incentive': 5145434.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 2070306.0, 'expenses': 8211.0, 'exercised_stock_options': 2070306.0, 'from_messages': 230.0, 'other': 2818454.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 8407016.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 477.0, 'restricted_stock': 0.0, 'from_poi_to_this_person': 8.0, 'fraction_emails_from_poi': 0.03361344537815126}, 86L: {'salary': 0.0, 'to_messages': 1433.0, 'name': 'MCCARTY DANNY J', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0013937282229965157, 'total_stock_value': 758931.0, 'expenses': 0.0, 'exercised_stock_options': 664375.0, 'from_messages': 215.0, 'other': 0.0, 'from_this_person_to_poi': 2.0, 'poi': False, 'total_payments': 0.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 508.0, 'restricted_stock': 94556.0, 'from_poi_to_this_person': 25.0, 'fraction_emails_from_poi': 0.10416666666666667}, 87L: {'salary': 263413.0, 'to_messages': 1744.0, 'name': 'MCCLELLAN GEORGE', 'long_term_incentive': 0.0, 'bonus': 900000.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 947861.0, 'expenses': 228763.0, 'exercised_stock_options': 506765.0, 'from_messages': 49.0, 'other': 51587.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 1318763.0, 'deferred_income': -125000.0, 'shared_receipt_with_poi': 1469.0, 'restricted_stock': 441096.0, 'from_poi_to_this_person': 52.0, 'fraction_emails_from_poi': 0.5148514851485149}, 88L: {'salary': 365038.0, 'to_messages': 3329.0, 'name': 'MCCONNELL MICHAEL S', 'long_term_incentive': 554422.0, 'bonus': 1100000.0, 'fraction_emails_to_poi': 0.05506670451319898, 'total_stock_value': 3101279.0, 'expenses': 81364.0, 'exercised_stock_options': 1623010.0, 'from_messages': 2742.0, 'other': 540.0, 'from_this_person_to_poi': 194.0, 'poi': False, 'total_payments': 2101364.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 2189.0, 'restricted_stock': 1478269.0, 'from_poi_to_this_person': 92.0, 'fraction_emails_from_poi': 0.032462949894142556}, 89L: {'salary': 0.0, 'to_messages': 894.0, 'name': 'MCDONALD REBECCA', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0011173184357541898, 'total_stock_value': 1691366.0, 'expenses': 0.0, 'exercised_stock_options': 757301.0, 'from_messages': 13.0, 'other': 0.0, 'from_this_person_to_poi': 1.0, 'poi': False, 'total_payments': 0.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 720.0, 'restricted_stock': 934065.0, 'from_poi_to_this_person': 54.0, 'fraction_emails_from_poi': 0.8059701492537313}, 90L: {'salary': 370448.0, 'to_messages': 2355.0, 'name': 'MCMAHON JEFFREY', 'long_term_incentive': 694862.0, 'bonus': 2600000.0, 'fraction_emails_to_poi': 0.010919781604367913, 'total_stock_value': 1662855.0, 'expenses': 137108.0, 'exercised_stock_options': 1104054.0, 'from_messages': 48.0, 'other': 297353.0, 'from_this_person_to_poi': 26.0, 'poi': False, 'total_payments': 4099771.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 2228.0, 'restricted_stock': 558801.0, 'from_poi_to_this_person': 58.0, 'fraction_emails_from_poi': 0.5471698113207547}, 91L: {'salary': 0.0, 'to_messages': 0.0, 'name': 'MENDELSOHN JOHN', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 0.0, 'expenses': 148.0, 'exercised_stock_options': 0.0, 'from_messages': 0.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 148.0, 'deferred_income': -103750.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 0.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 92L: {'salary': 365788.0, 'to_messages': 807.0, 'name': 'METTS MARK', 'long_term_incentive': 0.0, 'bonus': 600000.0, 'fraction_emails_to_poi': 0.0012376237623762376, 'total_stock_value': 585062.0, 'expenses': 94299.0, 'exercised_stock_options': 0.0, 'from_messages': 29.0, 'other': 1740.0, 'from_this_person_to_poi': 1.0, 'poi': False, 'total_payments': 1061827.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 702.0, 'restricted_stock': 585062.0, 'from_poi_to_this_person': 38.0, 'fraction_emails_from_poi': 0.5671641791044776}, 93L: {'salary': 0.0, 'to_messages': 0.0, 'name': 'MEYER JEROME J', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 0.0, 'expenses': 2151.0, 'exercised_stock_options': 0.0, 'from_messages': 0.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 2151.0, 'deferred_income': -38346.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 0.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 94L: {'salary': 0.0, 'to_messages': 232.0, 'name': 'MEYER ROCKFORD G', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 955873.0, 'expenses': 0.0, 'exercised_stock_options': 493489.0, 'from_messages': 28.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 1848227.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 22.0, 'restricted_stock': 462384.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 95L: {'salary': 0.0, 'to_messages': 672.0, 'name': 'MORAN MICHAEL P', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 221141.0, 'expenses': 0.0, 'exercised_stock_options': 59539.0, 'from_messages': 19.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 0.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 127.0, 'restricted_stock': 161602.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 96L: {'salary': 267093.0, 'to_messages': 0.0, 'name': 'MORDAUNT KRISTINA M', 'long_term_incentive': 0.0, 'bonus': 325000.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 208510.0, 'expenses': 35018.0, 'exercised_stock_options': 0.0, 'from_messages': 0.0, 'other': 1411.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 628522.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 208510.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 97L: {'salary': 251654.0, 'to_messages': 136.0, 'name': 'MULLER MARK S', 'long_term_incentive': 1725545.0, 'bonus': 1100000.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 1416848.0, 'expenses': 0.0, 'exercised_stock_options': 1056320.0, 'from_messages': 16.0, 'other': 947.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 3202070.0, 'deferred_income': -719000.0, 'shared_receipt_with_poi': 114.0, 'restricted_stock': 360528.0, 'from_poi_to_this_person': 12.0, 'fraction_emails_from_poi': 0.42857142857142855}, 98L: {'salary': 229284.0, 'to_messages': 2192.0, 'name': 'MURRAY JULIA H', 'long_term_incentive': 125000.0, 'bonus': 400000.0, 'fraction_emails_to_poi': 0.0009115770282588879, 'total_stock_value': 597461.0, 'expenses': 57580.0, 'exercised_stock_options': 400478.0, 'from_messages': 45.0, 'other': 330.0, 'from_this_person_to_poi': 2.0, 'poi': False, 'total_payments': 812194.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 395.0, 'restricted_stock': 196983.0, 'from_poi_to_this_person': 11.0, 'fraction_emails_from_poi': 0.19642857142857142}, 99L: {'salary': 0.0, 'to_messages': 0.0, 'name': 'NOLES JAMES L', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 368705.0, 'expenses': 0.0, 'exercised_stock_options': 0.0, 'from_messages': 0.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 774401.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 463261.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 100L: {'salary': 329078.0, 'to_messages': 1184.0, 'name': 'OLSON CINDY K', 'long_term_incentive': 100000.0, 'bonus': 750000.0, 'fraction_emails_to_poi': 0.012510425354462052, 'total_stock_value': 2606763.0, 'expenses': 63791.0, 'exercised_stock_options': 1637034.0, 'from_messages': 52.0, 'other': 972.0, 'from_this_person_to_poi': 15.0, 'poi': False, 'total_payments': 1321557.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 856.0, 'restricted_stock': 969729.0, 'from_poi_to_this_person': 20.0, 'fraction_emails_from_poi': 0.2777777777777778}, 101L: {'salary': 94941.0, 'to_messages': 0.0, 'name': 'OVERDYKE JR JERE C', 'long_term_incentive': 135836.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 7307594.0, 'expenses': 18834.0, 'exercised_stock_options': 5266578.0, 'from_messages': 0.0, 'other': 176.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 249787.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 2041016.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 102L: {'salary': 261879.0, 'to_messages': 0.0, 'name': 'PAI LOU L', 'long_term_incentive': 0.0, 'bonus': 1000000.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 23817930.0, 'expenses': 32047.0, 'exercised_stock_options': 15364167.0, 'from_messages': 0.0, 'other': 1829457.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 3123383.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 8453763.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 103L: {'salary': 0.0, 'to_messages': 0.0, 'name': 'PEREIRA PAULO V. FERRAZ', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 0.0, 'expenses': 27942.0, 'exercised_stock_options': 0.0, 'from_messages': 0.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 27942.0, 'deferred_income': -101250.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 0.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 104L: {'salary': 655037.0, 'to_messages': 898.0, 'name': 'PICKERING MARK R', 'long_term_incentive': 0.0, 'bonus': 300000.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 28798.0, 'expenses': 31653.0, 'exercised_stock_options': 28798.0, 'from_messages': 67.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 1386690.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 728.0, 'restricted_stock': 0.0, 'from_poi_to_this_person': 7.0, 'fraction_emails_from_poi': 0.0945945945945946}, 105L: {'salary': 197091.0, 'to_messages': 1238.0, 'name': 'PIPER GREGORY F', 'long_term_incentive': 0.0, 'bonus': 400000.0, 'fraction_emails_to_poi': 0.03732503888024884, 'total_stock_value': 880290.0, 'expenses': 43057.0, 'exercised_stock_options': 880290.0, 'from_messages': 222.0, 'other': 778.0, 'from_this_person_to_poi': 48.0, 'poi': False, 'total_payments': 1737629.0, 'deferred_income': -33333.0, 'shared_receipt_with_poi': 742.0, 'restricted_stock': 409554.0, 'from_poi_to_this_person': 61.0, 'fraction_emails_from_poi': 0.21554770318021202}, 106L: {'salary': 0.0, 'to_messages': 58.0, 'name': 'PIRO JIM', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.01694915254237288, 'total_stock_value': 47304.0, 'expenses': 0.0, 'exercised_stock_options': 0.0, 'from_messages': 16.0, 'other': 0.0, 'from_this_person_to_poi': 1.0, 'poi': False, 'total_payments': 0.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 3.0, 'restricted_stock': 47304.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 107L: {'salary': 0.0, 'to_messages': 653.0, 'name': 'POWERS WILLIAM', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 0.0, 'expenses': 0.0, 'exercised_stock_options': 0.0, 'from_messages': 26.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 0.0, 'deferred_income': -17500.0, 'shared_receipt_with_poi': 12.0, 'restricted_stock': 0.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 108L: {'salary': 0.0, 'to_messages': 0.0, 'name': 'PRENTICE JAMES', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 1095040.0, 'expenses': 0.0, 'exercised_stock_options': 886231.0, 'from_messages': 0.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 564348.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 208809.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 109L: {'salary': 96840.0, 'to_messages': 1671.0, 'name': 'REDMOND BRIAN L', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.028488372093023257, 'total_stock_value': 7890324.0, 'expenses': 14689.0, 'exercised_stock_options': 7509039.0, 'from_messages': 221.0, 'other': 0.0, 'from_this_person_to_poi': 49.0, 'poi': False, 'total_payments': 111529.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 1063.0, 'restricted_stock': 381285.0, 'from_poi_to_this_person': 204.0, 'fraction_emails_from_poi': 0.48}, 110L: {'salary': 76399.0, 'to_messages': 0.0, 'name': 'REYNOLDS LAWRENCE', 'long_term_incentive': 156250.0, 'bonus': 100000.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 4221891.0, 'expenses': 8409.0, 'exercised_stock_options': 4160672.0, 'from_messages': 0.0, 'other': 202052.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 394475.0, 'deferred_income': -200000.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 201483.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 111L: {'salary': 420636.0, 'to_messages': 905.0, 'name': 'RICE KENNETH D', 'long_term_incentive': 1617011.0, 'bonus': 1750000.0, 'fraction_emails_to_poi': 0.0044004400440044, 'total_stock_value': 22542539.0, 'expenses': 46950.0, 'exercised_stock_options': 19794175.0, 'from_messages': 18.0, 'other': 174839.0, 'from_this_person_to_poi': 4.0, 'poi': True, 'total_payments': 505050.0, 'deferred_income': -3504386.0, 'shared_receipt_with_poi': 864.0, 'restricted_stock': 2748364.0, 'from_poi_to_this_person': 42.0, 'fraction_emails_from_poi': 0.7}, 112L: {'salary': 249201.0, 'to_messages': 1328.0, 'name': 'RIEKER PAULA H', 'long_term_incentive': 0.0, 'bonus': 700000.0, 'fraction_emails_to_poi': 0.03488372093023256, 'total_stock_value': 1918887.0, 'expenses': 33271.0, 'exercised_stock_options': 1635238.0, 'from_messages': 82.0, 'other': 1950.0, 'from_this_person_to_poi': 48.0, 'poi': True, 'total_payments': 1099100.0, 'deferred_income': -100000.0, 'shared_receipt_with_poi': 1258.0, 'restricted_stock': 283649.0, 'from_poi_to_this_person': 35.0, 'fraction_emails_from_poi': 0.29914529914529914}, 113L: {'salary': 0.0, 'to_messages': 0.0, 'name': 'SAVAGE FRANK', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 0.0, 'expenses': 0.0, 'exercised_stock_options': 0.0, 'from_messages': 0.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 3750.0, 'deferred_income': -121284.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 0.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 114L: {'salary': 0.0, 'to_messages': 0.0, 'name': 'SCRIMSHAW MATTHEW', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 759557.0, 'expenses': 0.0, 'exercised_stock_options': 759557.0, 'from_messages': 0.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 0.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 0.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 115L: {'salary': 304110.0, 'to_messages': 3221.0, 'name': 'SHANKMAN JEFFREY A', 'long_term_incentive': 554422.0, 'bonus': 2000000.0, 'fraction_emails_to_poi': 0.025121065375302662, 'total_stock_value': 2072035.0, 'expenses': 178979.0, 'exercised_stock_options': 1441898.0, 'from_messages': 2681.0, 'other': 1191.0, 'from_this_person_to_poi': 83.0, 'poi': False, 'total_payments': 3038702.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 1730.0, 'restricted_stock': 630137.0, 'from_poi_to_this_person': 94.0, 'fraction_emails_from_poi': 0.033873873873873875}, 116L: {'salary': 269076.0, 'to_messages': 15149.0, 'name': 'SHAPIRO RICHARD S', 'long_term_incentive': 0.0, 'bonus': 650000.0, 'fraction_emails_to_poi': 0.004272380701985014, 'total_stock_value': 987001.0, 'expenses': 137767.0, 'exercised_stock_options': 607837.0, 'from_messages': 1215.0, 'other': 705.0, 'from_this_person_to_poi': 65.0, 'poi': False, 'total_payments': 1057548.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 4527.0, 'restricted_stock': 379164.0, 'from_poi_to_this_person': 74.0, 'fraction_emails_from_poi': 0.057408844065166796}, 117L: {'salary': 248146.0, 'to_messages': 3136.0, 'name': 'SHARP VICTORIA T', 'long_term_incentive': 422158.0, 'bonus': 600000.0, 'fraction_emails_to_poi': 0.0019096117122851686, 'total_stock_value': 494136.0, 'expenses': 116337.0, 'exercised_stock_options': 281073.0, 'from_messages': 136.0, 'other': 2401.0, 'from_this_person_to_poi': 6.0, 'poi': False, 'total_payments': 1576511.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 2477.0, 'restricted_stock': 213063.0, 'from_poi_to_this_person': 24.0, 'fraction_emails_from_poi': 0.15}, 118L: {'salary': 211844.0, 'to_messages': 225.0, 'name': 'SHELBY REX', 'long_term_incentive': 0.0, 'bonus': 200000.0, 'fraction_emails_to_poi': 0.058577405857740586, 'total_stock_value': 2493616.0, 'expenses': 22884.0, 'exercised_stock_options': 1624396.0, 'from_messages': 39.0, 'other': 1573324.0, 'from_this_person_to_poi': 14.0, 'poi': True, 'total_payments': 2003885.0, 'deferred_income': -4167.0, 'shared_receipt_with_poi': 91.0, 'restricted_stock': 869220.0, 'from_poi_to_this_person': 13.0, 'fraction_emails_from_poi': 0.25}, 119L: {'salary': 0.0, 'to_messages': 613.0, 'name': 'SHERRICK JEFFREY B', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.028526148969889066, 'total_stock_value': 1832468.0, 'expenses': 0.0, 'exercised_stock_options': 1426469.0, 'from_messages': 25.0, 'other': 0.0, 'from_this_person_to_poi': 18.0, 'poi': False, 'total_payments': 0.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 583.0, 'restricted_stock': 405999.0, 'from_poi_to_this_person': 39.0, 'fraction_emails_from_poi': 0.609375}, 120L: {'salary': 428780.0, 'to_messages': 3187.0, 'name': 'SHERRIFF JOHN R', 'long_term_incentive': 554422.0, 'bonus': 1500000.0, 'fraction_emails_to_poi': 0.0071651090342679125, 'total_stock_value': 3128982.0, 'expenses': 0.0, 'exercised_stock_options': 1835558.0, 'from_messages': 92.0, 'other': 1852186.0, 'from_this_person_to_poi': 23.0, 'poi': False, 'total_payments': 4335388.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 2103.0, 'restricted_stock': 1293424.0, 'from_poi_to_this_person': 28.0, 'fraction_emails_from_poi': 0.23333333333333334}, 121L: {'salary': 1111258.0, 'to_messages': 3627.0, 'name': 'SKILLING JEFFREY K', 'long_term_incentive': 1920000.0, 'bonus': 5600000.0, 'fraction_emails_to_poi': 0.008203445447087777, 'total_stock_value': 26093672.0, 'expenses': 29336.0, 'exercised_stock_options': 19250000.0, 'from_messages': 108.0, 'other': 22122.0, 'from_this_person_to_poi': 30.0, 'poi': True, 'total_payments': 8682716.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 2042.0, 'restricted_stock': 6843672.0, 'from_poi_to_this_person': 88.0, 'fraction_emails_from_poi': 0.4489795918367347}, 122L: {'salary': 239502.0, 'to_messages': 0.0, 'name': 'STABLER FRANK', 'long_term_incentive': 0.0, 'bonus': 500000.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 511734.0, 'expenses': 16514.0, 'exercised_stock_options': 0.0, 'from_messages': 0.0, 'other': 356071.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 1112087.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 511734.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 123L: {'salary': 162779.0, 'to_messages': 0.0, 'name': 'SULLIVAN-SHAKLOVITZ COLLEEN', 'long_term_incentive': 554422.0, 'bonus': 100000.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 1362375.0, 'expenses': 0.0, 'exercised_stock_options': 1362375.0, 'from_messages': 0.0, 'other': 162.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 999356.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 0.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 124L: {'salary': 257486.0, 'to_messages': 2647.0, 'name': 'SUNDE MARTIN', 'long_term_incentive': 476451.0, 'bonus': 700000.0, 'fraction_emails_to_poi': 0.004887218045112782, 'total_stock_value': 698920.0, 'expenses': 0.0, 'exercised_stock_options': 0.0, 'from_messages': 38.0, 'other': 111122.0, 'from_this_person_to_poi': 13.0, 'poi': False, 'total_payments': 1545059.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 2565.0, 'restricted_stock': 698920.0, 'from_poi_to_this_person': 37.0, 'fraction_emails_from_poi': 0.49333333333333335}, 125L: {'salary': 265214.0, 'to_messages': 533.0, 'name': 'TAYLOR MITCHELL S', 'long_term_incentive': 0.0, 'bonus': 600000.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 3745048.0, 'expenses': 0.0, 'exercised_stock_options': 3181250.0, 'from_messages': 29.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 1092663.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 300.0, 'restricted_stock': 563798.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 126L: {'salary': 0.0, 'to_messages': 0.0, 'name': 'THE TRAVEL AGENCY IN THE PARK', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 0.0, 'expenses': 0.0, 'exercised_stock_options': 0.0, 'from_messages': 0.0, 'other': 362096.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 362096.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 0.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 127L: {'salary': 222093.0, 'to_messages': 266.0, 'name': 'THORN TERENCE H', 'long_term_incentive': 200000.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 4817796.0, 'expenses': 46145.0, 'exercised_stock_options': 4452476.0, 'from_messages': 41.0, 'other': 426629.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 911453.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 73.0, 'restricted_stock': 365320.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 128L: {'salary': 247338.0, 'to_messages': 460.0, 'name': 'TILNEY ELIZABETH A', 'long_term_incentive': 275000.0, 'bonus': 300000.0, 'fraction_emails_to_poi': 0.02335456475583864, 'total_stock_value': 1168042.0, 'expenses': 0.0, 'exercised_stock_options': 591250.0, 'from_messages': 19.0, 'other': 152055.0, 'from_this_person_to_poi': 11.0, 'poi': False, 'total_payments': 399393.0, 'deferred_income': -575000.0, 'shared_receipt_with_poi': 379.0, 'restricted_stock': 576792.0, 'from_poi_to_this_person': 10.0, 'fraction_emails_from_poi': 0.3448275862068966}, 129L: {'salary': 288589.0, 'to_messages': 111.0, 'name': 'UMANOFF ADAM S', 'long_term_incentive': 0.0, 'bonus': 788750.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 0.0, 'expenses': 53122.0, 'exercised_stock_options': 0.0, 'from_messages': 18.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 1130461.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 41.0, 'restricted_stock': 0.0, 'from_poi_to_this_person': 12.0, 'fraction_emails_from_poi': 0.4}, 130L: {'salary': 0.0, 'to_messages': 0.0, 'name': 'URQUHART JOHN A', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 0.0, 'expenses': 228656.0, 'exercised_stock_options': 0.0, 'from_messages': 0.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 228656.0, 'deferred_income': -36666.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 0.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 131L: {'salary': 0.0, 'to_messages': 0.0, 'name': 'WAKEHAM JOHN', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 0.0, 'expenses': 103773.0, 'exercised_stock_options': 0.0, 'from_messages': 0.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 213071.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 0.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 132L: {'salary': 357091.0, 'to_messages': 671.0, 'name': 'WALLS JR ROBERT H', 'long_term_incentive': 540751.0, 'bonus': 850000.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 5898997.0, 'expenses': 50936.0, 'exercised_stock_options': 4346544.0, 'from_messages': 146.0, 'other': 2.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 1798780.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 215.0, 'restricted_stock': 1552453.0, 'from_poi_to_this_person': 17.0, 'fraction_emails_from_poi': 0.10429447852760736}, 133L: {'salary': 0.0, 'to_messages': 0.0, 'name': 'WALTERS GARETH W', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 1030329.0, 'expenses': 33785.0, 'exercised_stock_options': 1030329.0, 'from_messages': 0.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 87410.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 0.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 134L: {'salary': 259996.0, 'to_messages': 400.0, 'name': 'WASAFF GEORGE', 'long_term_incentive': 200000.0, 'bonus': 325000.0, 'fraction_emails_to_poi': 0.0171990171990172, 'total_stock_value': 2056427.0, 'expenses': 0.0, 'exercised_stock_options': 1668260.0, 'from_messages': 30.0, 'other': 1425.0, 'from_this_person_to_poi': 7.0, 'poi': False, 'total_payments': 1034395.0, 'deferred_income': -583325.0, 'shared_receipt_with_poi': 337.0, 'restricted_stock': 388167.0, 'from_poi_to_this_person': 22.0, 'fraction_emails_from_poi': 0.4230769230769231}, 135L: {'salary': 63744.0, 'to_messages': 0.0, 'name': 'WESTFAHL RICHARD K', 'long_term_incentive': 256191.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 384930.0, 'expenses': 51870.0, 'exercised_stock_options': 0.0, 'from_messages': 0.0, 'other': 401130.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 762135.0, 'deferred_income': -10800.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 384930.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 136L: {'salary': 0.0, 'to_messages': 0.0, 'name': 'WHALEY DAVID A', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 98718.0, 'expenses': 0.0, 'exercised_stock_options': 98718.0, 'from_messages': 0.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 0.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 0.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 137L: {'salary': 510364.0, 'to_messages': 6019.0, 'name': 'WHALLEY LAWRENCE G', 'long_term_incentive': 808346.0, 'bonus': 3000000.0, 'fraction_emails_to_poi': 0.003971537315902697, 'total_stock_value': 6079137.0, 'expenses': 57838.0, 'exercised_stock_options': 3282960.0, 'from_messages': 556.0, 'other': 301026.0, 'from_this_person_to_poi': 24.0, 'poi': False, 'total_payments': 4677574.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 3920.0, 'restricted_stock': 2796177.0, 'from_poi_to_this_person': 186.0, 'fraction_emails_from_poi': 0.25067385444743934}, 138L: {'salary': 317543.0, 'to_messages': 0.0, 'name': 'WHITE JR THOMAS E', 'long_term_incentive': 0.0, 'bonus': 450000.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 15144123.0, 'expenses': 81353.0, 'exercised_stock_options': 1297049.0, 'from_messages': 0.0, 'other': 1085463.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 1934359.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 13847074.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 139L: {'salary': 0.0, 'to_messages': 0.0, 'name': 'WINOKUR JR. HERBERT S', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 0.0, 'expenses': 1413.0, 'exercised_stock_options': 0.0, 'from_messages': 0.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 84992.0, 'deferred_income': -25000.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 0.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 140L: {'salary': 0.0, 'to_messages': 0.0, 'name': 'WODRASKA JOHN', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 0.0, 'expenses': 0.0, 'exercised_stock_options': 0.0, 'from_messages': 0.0, 'other': 189583.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 189583.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 0.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 141L: {'salary': 0.0, 'to_messages': 0.0, 'name': 'WROBEL BRUCE', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 139130.0, 'expenses': 0.0, 'exercised_stock_options': 139130.0, 'from_messages': 0.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 0.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 0.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 142L: {'salary': 158403.0, 'to_messages': 0.0, 'name': 'YEAGER F SCOTT', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 11884758.0, 'expenses': 53947.0, 'exercised_stock_options': 8308552.0, 'from_messages': 0.0, 'other': 147950.0, 'from_this_person_to_poi': 0.0, 'poi': True, 'total_payments': 360300.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 3576206.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}, 143L: {'salary': 0.0, 'to_messages': 0.0, 'name': 'YEAP SOON', 'long_term_incentive': 0.0, 'bonus': 0.0, 'fraction_emails_to_poi': 0.0, 'total_stock_value': 192758.0, 'expenses': 55097.0, 'exercised_stock_options': 192758.0, 'from_messages': 0.0, 'other': 0.0, 'from_this_person_to_poi': 0.0, 'poi': False, 'total_payments': 55097.0, 'deferred_income': 0.0, 'shared_receipt_with_poi': 0.0, 'restricted_stock': 0.0, 'from_poi_to_this_person': 0.0, 'fraction_emails_from_poi': 0.0}}
    


    <Figure size 432x288 with 0 Axes>



```python
%run tester.py
```

    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=None, n_neighbors=5, p=2,
               weights='uniform')
    	Accuracy: 0.88465	Precision: 0.71121	Recall: 0.29183	F1: 0.41385	F2: 0.33085
    	Total predictions: 43000	True positives: 1751	False positives:  711	False negatives: 4249	True negatives: 36289
    
    

<a id='wrapup'></a>
### Project Wrap Up
</br>

- I believe that the algorithm and feature set chosen has performed well enough in our testing and implementation despite this being a smaller dataset.  It would have been helpful if the dataset had been larger (who wouldn't want more data!) but you must work with what you are given.


```python

```
