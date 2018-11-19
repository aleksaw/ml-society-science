# -*- Mode: python -*-
# A simple reference recommender
#
#
# This is a medical scenario with historical data.
#
# General functions
#
# - set_reward
#
# There is a set of functions for dealing with historical data:
#
# - fit_data
# - fit_treatment_outcome
# - estimate_utiltiy
#
# There is a set of functions for online decision making
#
# - predict_proba
# - recommend
# - observe

from recommender import Recommender

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

class ImprovedRecommender(Recommender):

    #################################
    # Initialise
    #
    # Set the recommender with a default number of actions and outcomes.  This is
    # because the number of actions in historical data can be
    # different from the ones that you can take with your policy.
    def __init__(self, n_actions, n_outcomes, quick_fits=False,
                 refit_trigger=1.1):
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.reward = self._default_reward
        self.scaler = StandardScaler()
        self.quick_fits = quick_fits
        self.refit_trigger = 1.1

    ##################################
    # Fit a model from patient data.
    #
    # This will generally speaking be an
    # unsupervised model. Anything from a Gaussian mixture model to a
    # neural network is a valid choice.  However, you can give special
    # meaning to different parts of the data, and use a supervised
    # model instead.
    def fit_data(self, data):
        # Could we use word2vec here to capture similarity?
        # http://mccormickml.com/2018/06/15/applying-word2vec-to-recommenders-and-advertising/
        print("Preprocessing data")
        return None


    ## Fit a model from patient data, actions and their effects
    ## Here we assume that the outcome is a direct function of data and actions
    ## This model can then be used in estimate_utility(), predict_proba() and recommend()
    def fit_treatment_outcome(self, data, actions, outcomes, quick=False):
        #print("Fitting treatment outcomes")
        self.X = self.scaler.fit_transform(data)
        self.A = actions
        self.Y = outcomes
        self.model = [None for _ in range(self.n_actions)]
        self._fit_models(quick=quick)
        return None

    ## Fit a model to the data we have
    def _fit_models(self, quick = False):
        self.data_in_model = self.X.shape[0]
        # Set defaults and triggerpoints
        quick_trigger = 10
        quick_c = 10^8
        acc_rr_threshold = 0.2
        FiveFold_threshold = 200
        fits = 200 #n_repeats * n_folds
        Cs = np.logspace(-1,6,30)

        def acc_rr(y_pred, y_te):
            tp = (y_pred&yte).sum()
            fp = ((y_pred==1)&(y_te==0)).sum()
            fn = ((y_pred==0)&(y_te==1)).sum()
            # If there are no 1's in either array we get no result, but then it's a match.
            return (tp / (tp+fp+fn)) if (tp+fp+fn)>0 else 1

        class DummyClassifier:
            def __init__(self, probabilities):
                self.p = probabilities
            def predict_proba():
                return p

        A = self.A
        Y = self.Y
        for a in range(self.n_actions):
            X_A = self.X[(A==a)[:,0],:]
            Y_A = Y[(A==a)[:,0],].ravel()

            if Y_A.sum() == 0:
                # We can't fit anything with only one class in data
                ps = np.append([1], np.zeros(self.n_outcomes-1))
                self.model[a] = DummyClassifier(ps)
                continue

            if Y_A.sum() < quick_trigger:
                # We're likely to run into trouble fitting classifier in
                # KFold, so we drop that.
                quick = True

            if quick:
                self.model[a] = LogisticRegression(C=quick_c, solver='sag').fit(X_A, Y_A)
                continue

            if Y_A.sum() < acc_rr_threshold * len(Y_A):
                measure = acc_rr
            else:
                measure = accuracy_score

            n_folds = 10 if len(Y_A) > FiveFold_threshold else 5
            n_repeats = round(fits / n_folds)
            accuracies = np.zeros(len(Cs))
            for i, c in enumerate(Cs):
                accuracy = np.zeros(n_folds * n_repeats)
                j = 0
                rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats)
                for train, test in rskf.split(X=X_A, y=Y_A, groups=Y_A):
                    Xtr, ytr = (X_A[train,], Y_A[train])
                    Xte, yte = (X_A[test,], Y_A[test])
                    y_pred = LogisticRegression(C=c, solver='sag').fit(Xtr, ytr).predict(Xte)
                    accuracy[j] = measure(y_pred, yte)
                    j += 1
                accuracies[i] = np.mean(accuracy)

            # I'll go with highest accuracy. We could have gone most parsimonious
            # within 1se f.ex, but test indicate less regularization is better.
            best_c = Cs[np.argmax(accuracies)]
            self.model[a] = LogisticRegression(C=best_c, solver='sag').fit(X_A, Y_A)

        return None


    # Return a distribution of effects for a given person's data and a specific treatment.
    # This should be an numpy.array of length self.n_outcomes
    def predict_proba(self, data, treatment):
        data = self.scaler.transform(data)
        return self.model[treatment].predict_proba(data)

    # Return a distribution of recommendations for a specific user datum
    # This should a numpy array of size equal to self.n_actions, summing up to 1
    def get_action_probabilities(self, user_data):
        Er = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            p = self.predict_proba(user_data, a)
            r = self.reward(a, np.arange(self.n_outcomes))
            Er[a] = (p * r).sum()
            #Er[a] = np.dot(p, r)
        return np.exp(Er) / (np.exp(Er)).sum()


    # Return recommendations for a specific user datum
    # This should be an integer in range(self.n_actions)
    def recommend(self, user_data):
        #print("Recommending")
        return np.argmax(self.get_action_probabilities(user_data))
        # The above rule might give the best results, but it also will probably
        # never assign placebo treatment given the data we have now.
        # This is an option that reduces that problem
        #return np.random.choice(np.arange(self.n_actions), p=self.get_action_probabilities(user_data))

    # Observe the effect of an action. This is an opportunity for you
    # to refit your models, to take the new information into account.
    def observe(self, user, action, outcome):
        self.X = np.append(self.X, self.scaler.transform(user.reshape(1,-1)))
        self.A = np.append(self.A, actions)
        self.Y = np.append(self.Y, outcome)
        # Check how much we have increased out dataset and update models if necessary
        if self.X.shape[0] > self.refit_trigger * self.data_in_model:
            self._fit_models(quick=self.quick_fits)
        return None
