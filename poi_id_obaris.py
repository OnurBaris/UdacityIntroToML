#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'total_payments', 'bonus', 'total_stock_value', 'expenses',\
                 'exercised_stock_options', 'long_term_incentive', 'restricted_stock', 'to_messages', 'from_messages'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
del data_dict['TOTAL']
del data_dict['CHRISTODOULOU DIOMEDES']
del data_dict['CLINE KENNETH W']
del data_dict['BROWN MICHAEL']
del data_dict['CORDES WILLIAM R']
del data_dict['FOWLER PEGGY']
del data_dict['FOY JOE']
del data_dict['GATHMANN WILLIAM D']
del data_dict['GILLIS JOHN']
del data_dict['HORTON STANLEY C']
del data_dict['HUGHES JAMES A']
del data_dict['PRENTICE JAMES']
del data_dict['WROBEL BRUCE']
del data_dict['GIBBS DANA R']
del data_dict['PIRO JIM']
del data_dict['POWERS WILLIAM']
del data_dict['WINOKUR JR. HERBERT S']
del data_dict['WODRASKA JOHN']
del data_dict['THE TRAVEL AGENCY IN THE PARK']
del data_dict['BERBERIAN DAVID']
del data_dict['BERGSIEKER RICHARD P']
del data_dict['BHATNAGAR SANJAY']
del data_dict['BIBI PHILIPPE A']
del data_dict['BLACHMAN JEREMY M']
del data_dict['BELFER ROBERT']
### Task 3: Create new feature(s)
nan_count=0
for i in data_dict:
        data_dict[i]['total_payment_bonus_ratio']=str(float(data_dict[i]['total_payments'])/float(data_dict[i]['bonus']))
        data_dict[i]['exercised_retricted_stock_ratio']=str(float(data_dict[i]['exercised_stock_options'])/float(data_dict[i]['restricted_stock']))
        for a in data_dict[i]:
            if data_dict[i][a]=='NaN' or data_dict[i][a]=='nan':
                nan_count+=1
                data_dict[i][a]=0 
            else:
                continue
features_list.extend(['total_payment_bonus_ratio','exercised_retricted_stock_ratio'])

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.feature_selection import SelectKBest, f_classif
feat_select=SelectKBest(f_classif, k=7).fit(features, labels)
plt.bar(features_list[1:], feat_select.scores_, width=.2,
        label=r'Scores', color='darkorange', edgecolor='black')
features=SelectKBest(f_classif, k=7).fit_transform(features, labels) 

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import f1_score, average_precision_score, recall_score, precision_score
clf = GaussianNB()
clf_SVC = SVC()
clf_LinearSVC = LinearSVC()
clf_GPC = GaussianProcessClassifier()
clf_tree = DecisionTreeClassifier()
clf_RFC = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=42)
clf_AdaB = AdaBoostClassifier()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

clf.fit(features_train, labels_train)
clf_SVC.fit(features_train, labels_train)
clf_LinearSVC.fit(features_train, labels_train)
clf_GPC.fit(features_train, labels_train)
clf_tree.fit(features_train, labels_train)
clf_RFC.fit(features_train, labels_train)
clf_AdaB.fit(features_train, labels_train)
Models = [clf, clf_SVC, clf_LinearSVC, clf_GPC, clf_tree, clf_RFC, clf_AdaB]
y_true=[]
y_pred=[]
y_false=[]
y_falsepred=[]
i=0
y = clf_RFC.predict(features_test)
while i<len(labels_test):
    if labels_test[i]==True:
        y_true.append(labels_test[i])
        y_pred.append(y[i])
    else:
        y_false.append(labels_test[i])
        y_falsepred.append(y[i])
    i+=1

print precision_score(y_true, y_pred), recall_score(y_true, y_pred)
print precision_score(y_false, y_falsepred), recall_score(y_false, y_falsepred)

print 'GaussianNB: ' + str(clf.score(features_test,labels_test)) + ' ' + str(f1_score(labels_test, clf.predict(features_test), average='weighted'))
print 'Support Vector Machine: ' + str(clf_SVC.score(features_test,labels_test)) + ' ' + str(f1_score(labels_test, clf_SVC.predict(features_test),average='weighted'))
print 'Linear Support Vector Machine: ' + str(clf_LinearSVC.score(features_test,labels_test)) + ' ' + str(f1_score(labels_test, clf_LinearSVC.predict(features_test),average='weighted'))
print 'Gaussian Process: ' + str(clf_GPC.score(features_test,labels_test)) + ' ' + str(f1_score(labels_test, clf_GPC.predict(features_test),average='weighted'))
print 'Decision Tree: ' + str(clf_tree.score(features_test,labels_test)) + ' ' + str(f1_score(labels_test, clf_tree.predict(features_test),average='weighted'))
print 'Random Forest: ' + str(clf_RFC.score(features_test,labels_test)) + ' ' + str(f1_score(labels_test, clf_RFC.predict(features_test),average='weighted'))
print 'Ada Boost: ' + str(clf_AdaB.score(features_test,labels_test)) + ' ' + str(f1_score(labels_test, clf_AdaB.predict(features_test),average='weighted'))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)