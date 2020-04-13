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

x1=[]
x2=[]
y1=[]
y2=[]
z1=[]
z2=[]
a1=[]
a2=[]
b1=[]
b2=[]

print data_dict['CHRISTODOULOU DIOMEDES']['poi']
print data_dict['CLINE KENNETH W']['poi']
print data_dict['BROWN MICHAEL']['poi']
print data_dict['CORDES WILLIAM R']['poi']
print data_dict['FOWLER PEGGY']['poi']
print data_dict['FOY JOE']['poi']
print data_dict['GATHMANN WILLIAM D']['poi']
print data_dict['GILLIS JOHN']['poi']
print data_dict['HORTON STANLEY C']['poi']
print data_dict['HUGHES JAMES A']['poi']
print data_dict['PRENTICE JAMES']['poi']
print data_dict['WROBEL BRUCE']['poi']
print data_dict['GIBBS DANA R']['poi']
print data_dict['PIRO JIM']['poi']
print data_dict['POWERS WILLIAM']['poi']

"""for i in data_dict:
        if data_dict[i]['poi']==True:
            x1.append(float(data_dict[i]['total_payments'])/float(data_dict[i]['bonus']))
            y1.append(float(data_dict[i]['salary']))
            z1.append(float(data_dict[i]['exercised_stock_options'])/float(data_dict[i]['total_stock_value']))
            a1.append(float(data_dict[i]['long_term_incentive']))
            b1.append(float(data_dict[i]['to_messages']))
        if data_dict[i]['poi']==False:
            x2.append(float(data_dict[i]['total_payments'])/float(data_dict[i]['bonus']))
            y2.append(float(data_dict[i]['salary']))
            z2.append(float(data_dict[i]['exercised_stock_options'])/float(data_dict[i]['total_stock_value']))
            a2.append(float(data_dict[i]['long_term_incentive']))
            b2.append(float(data_dict[i]['to_messages']))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x1, y1, z1, c='red')
ax.scatter(x2, y2, z2, c='blue')
plt.show()
plt.clf()

fig2=plt.scatter(a1, b1, c='red')
fig2=plt.scatter(a2, b2, c='blue')
plt.show()"""


"""from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(0.15)
features=selector.fit_transform(features)
print(features.shape)"""

### Task 2: Remove outliers
data_dict.pop('TOTAL')
### Task 3: Create new feature(s)
for i in data_dict:
        data_dict[i]['total_payment_bonus_ratio']=str(float(data_dict[i]['total_payments'])/float(data_dict[i]['bonus']))
        data_dict[i]['exercised_total_stock_ratio']=str(float(data_dict[i]['exercised_stock_options'])/float(data_dict[i]['total_stock_value']))
features_list.extend(['total_payment_bonus_ratio','exercised_total_stock_ratio'])
features_list.remove('total_payments')
features_list.remove('bonus')
features_list.remove('exercised_stock_options')
features_list.remove('total_stock_value')

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)