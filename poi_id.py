#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import pandas as pd

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','bonus','fraction_emails_to_poi','exercised_stock_options']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    enron_dict = pickle.load(data_file)


### Remove outliers
enron_dict.pop('LOCKHART EUGENE E',0)
enron_dict.pop('TOTAL',0)

### Converting to dataframe
enron_df = pd.DataFrame(enron_dict)
enron_df = enron_df.transpose()

cols = ['salary', 'to_messages', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus','restricted_stock_deferred',
        'deferred_income', 'total_stock_value', 'expenses', 'from_poi_to_this_person', 'exercised_stock_options',
        'from_messages', 'other', 'from_this_person_to_poi', 'long_term_incentive', 'shared_receipt_with_poi',
        'restricted_stock','director_fees']
enron_df[cols] = enron_df[cols].apply(pd.to_numeric, errors='coerce', axis=1)
enron_df = enron_df.reset_index()
enron_df.rename(columns={"index": "name"}, inplace=True)

### Removing columns due to missing values (NaNs)
enron_df.drop(['email_address', 'deferral_payments', 'loan_advances',
               'restricted_stock_deferred', 'director_fees'], axis = 1, inplace = True)

### Added a feature listing the fraction of emails to and from POIs
### Also fills in 'NaN' spaces
enron_df['fraction_emails_to_poi'] = enron_df['from_this_person_to_poi'].fillna(0.0)/ (enron_df['to_messages'].fillna(0.0) + enron_df['from_this_person_to_poi'].fillna(0.0))

enron_df['fraction_emails_from_poi'] = enron_df['from_poi_to_this_person'].fillna(0.0)/ (enron_df['from_messages'].fillna(0.0) + enron_df['from_poi_to_this_person'].fillna(0.0))

### Replacing NaN values with 0.0
enron_df.fillna(0.0, inplace=True)

### Converting dataframe back to dictionary for processing
enron_df.set_index('name')
enron_dict = enron_df.to_dict('index')

print(enron_dict) 
### Store to my_dataset for easy export below.
my_dataset = enron_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Please name your classifier clf for easy export below.

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = featureFormat(enron_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, 
                                                                            test_size = 0.33, random_state = 42)
    
 
clf = KNeighborsClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)