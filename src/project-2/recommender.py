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

from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np

class Recommender:

    #################################
    # Initialise
    #
    # Set the recommender with a default number of actions and outcomes.  This is
    # because the number of actions in historical data can be
    # different from the ones that you can take with your policy.
    def __init__(self, n_actions, n_outcomes):
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.reward = self._default_reward

    ## By default, the reward is just equal to the outcome, as the actions play no role.
    def _default_reward(self, action, outcome):
        return outcome

    # Set the reward function r(a, y)
    def set_reward(self, reward):
        self.reward = reward

    ##################################
    # Fit a model from patient data.
    #
    # This will generally speaking be an
    # unsupervised model. Anything from a Gaussian mixture model to a
    # neural network is a valid choice.  However, you can give special
    # meaning to different parts of the data, and use a supervised
    # model instead.
    def fit_data(self, data):
        print("Preprocessing data")
        self.X = data
        return None


    ## Fit a model from patient data, actions and their effects
    ## Here we assume that the outcome is a direct function of data and actions
    ## This model can then be used in estimate_utility(), predict_proba() and recommend()
    def fit_treatment_outcome(self, data, actions, outcomes, quick=False):
        print("Fitting treatment outcomes")
        self.X = data
        self.A = actions
        self.Y = outcomes
        return None


    ## Estimate the utility of a specific policy from historical data (data, actions, outcome),
    ## where utility is the expected reward of the policy.
    ##
    ## If policy is not given, simply use the average reward of the observed actions and outcomes.
    ##
    ## If a policy is given, then you can either use importance
    ## sampling, or use the model you have fitted from historical data
    ## to get an estimate of the utility.
    ##
    ## The policy should be a recommender that implements get_action_probability()
    def estimate_utility(self, data, actions, outcomes, policy=None,
                         n_splits=10, n_repeats=5, quick=True):
        def utility(a, y):
            return np.sum(self.reward(a, y)) / len(a)

        if policy==None:
            return utility(actions.values, outcomes.values)

        EU = 0
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
        split_sensitive = outcomes+self.n_outcomes*actions
        for train, test in rskf.split(data, outcomes, split_sensitive):
            Xtr, Atr, Ytr = (data.iloc[train,:], actions.iloc[train,:], outcomes.iloc[train,:])
            Xte, Ate, Yte = (data.iloc[test,:], actions.iloc[test,:], outcomes.iloc[test,:])
            policy.fit_treatment_outcome(Xtr, Atr, Ytr, quick=quick)
            A_hat = np.array([policy.recommend(Xte.iloc[i:i+1,:]) for i in range(len(Xte))])
            # We only know the outcome in the cases where our recommendation
            # equals the historical action
            mask = (Ate.values.ravel()==A_hat)
            EU += utility(Ate.iloc[mask].values, Yte.iloc[mask].values)
        return EU/(n_splits*n_repeats)

    # Return a distribution of effects for a given person's data and a specific treatment.
    # This should be an numpy.array of length self.n_outcomes
    def predict_proba(self, data, treatment):
        return np.ones(self.n_outcomes) / self.n_outcomes

    # Return a distribution of recommendations for a specific user datum
    # This should a numpy array of size equal to self.n_actions, summing up to 1
    def get_action_probabilities(self, user_data):
        return np.ones(self.n_actions) / self.n_actions

    # Return recommendations for a specific user datum
    # This should be an integer in range(self.n_actions)
    def recommend(self, user_data):
        return np.random.choice(np.arange(self.n_actions))

    # Observe the effect of an action. This is an opportunity for you
    # to refit your models, to take the new information into account.
    def observe(self, user, action, outcome):
        self.data = np.append(self.data)
        self.actions = self.actions.append(actions)
        self.outcomes = self.outcomes.append(outcome)
        return None
