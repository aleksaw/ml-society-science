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

class HomeopathicRecommender(Recommender):

    # Return a distribution of effects for a given person's data and a specific treatment.
    # This should be an numpy.array of length self.n_outcomes
    def predict_proba(self, data, treatment):
        ret = np.zeros(self.n_outcomes)
        ret[0] = 1
        return ret

    # Return recommendations for a specific user datum
    # This should be an integer in range(self.n_actions)
    def recommend(self, user_data):
        return 0
