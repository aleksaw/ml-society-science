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

import numpy as np

class HistoricalRecommender2(Recommender):

    # Return recommendations for a specific user datum
    # This should be an integer in range(self.n_actions)
    def recommend(self, user_data, exploring=0):
        #print("Recommending")
        R = 1*(np.random.uniform() < user_data[128] * 0.4  + user_data[129] * 0.5);
        self.obs_R = np.append(self.obs_R, R)
        return R
