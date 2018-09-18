import pandas

## Set up for dataset
features = ['checking account balance', 'duration', 'credit history',
            'purpose', 'amount', 'savings', 'employment', 'installment',
            'marital status', 'other debtors', 'residence time',
            'property', 'age', 'other installments', 'housing', 'credits',
            'job', 'persons', 'phone', 'foreign']
target = 'repaid'
df = pandas.read_csv('../../data/credit/german.data', sep=' ',
                     names=features+[target])
import matplotlib.pyplot as plt
numerical_features = ['duration', 'age', 'residence time', 'installment', 'amount', 'duration', 'persons', 'credits']
quantitative_features = list(filter(lambda x: x not in numerical_features, features))
X = pandas.get_dummies(df, columns=quantitative_features, drop_first=True)
encoded_features = list(filter(lambda x: x != target, X.columns))

## Test function
def test_decision_maker(X_test, y_test, interest_rate, decision_maker):
    n_test_examples = len(X_test)
    utility = 0

    ## Example test function - this is only an unbiased test if the data has not been seen in training
    accuracy_matrix = [[0, 0], [0, 0]]
    for t in range(n_test_examples):
        action = decision_maker.get_best_action(X_test.iloc[t])
        good_loan = y_test.iloc[t] # assume the labels are correct
        duration = X_test['duration'].iloc[t]
        amount = X_test['amount'].iloc[t]
        # If we don't grant the loan then nothing happens
        accuracy_matrix[action][good_loan % 2] += 1
        if (action==1):
            if (good_loan == 1):
                utility += amount*(pow(1 + interest_rate, duration) - 1)
            else:
                utility -= amount
    #print(accuracy_matrix)
    return utility


## Main code


### Setup model
import random_banker # this is a random banker
decision_maker = random_banker.RandomBanker()
interest_rate = 0.0165

### Do a number of preliminary tests by splitting the data in parts
from sklearn.model_selection import train_test_split
n_tests = 100
utility = 0
for iter in range(n_tests):
    X_train, X_test, y_train, y_test = train_test_split(X[encoded_features], X[target], test_size=0.2)
    decision_maker.set_interest_rate(interest_rate)
    decision_maker.fit(X_train, y_train)
    utility += test_decision_maker(X_test, y_test, interest_rate, decision_maker)

print(utility / n_tests)

import aleksaw_banker
decision_maker = aleksaw_banker.NameBanker()
print("Start My Banker")
utility = 0
for iter in range(n_tests):
    X_train, X_test, y_train, y_test = train_test_split(X[encoded_features], X[target], test_size=0.2)
    decision_maker.set_interest_rate(interest_rate)
    decision_maker.fit(X_train, y_train)
    utility += test_decision_maker(X_test, y_test, interest_rate, decision_maker)
    # print("My Banker run {}: {}".format(iter+1, utility))

print("My Banker: {}".format(utility / n_tests))






"""import pandas

## Set up for dataset
features = ['checking account balance', 'duration', 'credit history',
            'purpose', 'amount', 'savings', 'employment', 'installment',
            'marital status', 'other debtors', 'residence time',
            'property', 'age', 'other installments', 'housing', 'credits',
            'job', 'persons', 'phone', 'foreign']
target = 'repaid'
df = pandas.read_csv('../../data/credit/german.data', sep=' ',
                     names=features+[target])
import matplotlib.pyplot as plt
numerical_features = ['duration', 'age', 'residence time', 'installment', 'amount', 'duration', 'persons', 'credits']
quantitative_features = list(filter(lambda x: x not in numerical_features, features))
X = pandas.get_dummies(df, columns=quantitative_features, drop_first=True)
encoded_features = list(filter(lambda x: x != target, X.columns))

## Test function
def test_decision_maker(X_test, y_test, interest_rate, decision_maker):
    n_test_examples = len(X_test)
    utility = 0
    loans_granted = 0
    ## Example test function - this is not an unbiased test as it uses the training data directly. Adapt as necessary
    for t in range(n_test_examples):
        action = decision_maker.get_best_action(X_test.iloc[t])
        good_loan = (y_test.iloc[t] == 1) # assume the labels are correct
        duration = X_test['duration'].iloc[t]
        amount = X_test['amount'].iloc[t]
        # If we don't grant the loan then nothing happens
        if (action==1):
            loans_granted += 1
            if (good_loan):
                utility -= amount
            else:
                utility += amount*((1 + interest_rate)**duration - 1)
    print("Loans granted: {:.1f}%".format(100*loans_granted/n_test_examples))
    return utility


## Main code


### Setup model
#import logistic_banker
#decision_maker = logistic_banker.LogisticBanker()
"""
"""import reference_banker
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(16, 4, 2), random_state=1)
bagging = BaggingClassifier(KNeighborsClassifier(), n_estimators=10)
knn = KNeighborsClassifier()
logistic = linear_model.LogisticRegression()
"""
"""#decision_maker = reference_banker.ReferenceBanker(mlp)
import random_banker
decision_maker = random_banker.RandomBanker()
interest_rate = 0.005

### Do a number of preliminary tests by splitting the data in parts
from sklearn.model_selection import train_test_split
n_tests = 10
print("Start Random Banker")
utility = 0
for iter in range(n_tests):
    X_train, X_test, y_train, y_test = train_test_split(X[encoded_features], X[target], test_size=0.2)
    decision_maker.set_interest_rate(interest_rate)
    decision_maker.fit(X_train, y_train)
    utility += test_decision_maker(X_test, y_test, interest_rate, decision_maker)
    print("Random Banker run {}: {}".format(iter+1, utility))

print("Random Banker: {}".format(utility / n_tests))

import aleksaw_banker
decision_maker = aleksaw_banker.NameBanker()
print("Start My Banker")
utility = 0
for iter in range(n_tests):
    X_train, X_test, y_train, y_test = train_test_split(X[encoded_features], X[target], test_size=0.2)
    decision_maker.set_interest_rate(interest_rate)
    decision_maker.fit(X_train, y_train)
    utility += test_decision_maker(X_test, y_test, interest_rate, decision_maker)
    print("My Banker run {}: {}".format(iter+1, utility))

print("My Banker: {}".format(utility / n_tests))"""
