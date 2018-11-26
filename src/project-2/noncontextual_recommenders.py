""" 3 Non-contextual recommenders

Random Recommender
Optimistic Recommender
Homeopathic Recommender
"""

from recommender import Recommender

import numpy as np

class RandomRecommender(Recommender):
    """ A random recommender making entirely random recommendations """

    def predict_proba(self, data: np.ndarray, treatment: int) -> np.ndarray:
        """ Predict outcome probabilities

        Return a distribution of effects for a given person's data and a
        specific treatment. The random recommender is predicting equal
        probability of each outcome for all actions.

        Parameters
        ----------
        user : np.ndarray(n_features)
            Features of the user for whom to predict
        treatment: int
            The actions to take

        Returns
        ---------
        np.ndarray(n_outcomes)
            The probability of each outcome given the data and the treatment
        """
        return numpy.ones(self.n_outcomes) / self.n_outcomes

    def get_action_probabilities(self, user_data: np.ndarray) -> np.ndarray:
        """ Probabilities of each action being the best action

        Return a distribution of recommendations for a specific user datum.
        The random recommender gives equal probability to each action being
        the best

        Parameters
        ----------
        user_data : np.ndarray(n_features)
            Features of the user for whom to predict

        Returns
        ---------
        np.ndarray(n_actions)
            The probability of each action being the best, summing to 1
        """
        #print("Recommending")
        return np.ones(self.n_actions) / self.n_actions;

    def recommend(self, user_data: np.ndarray, exploring: float=0.1) -> int:
        """ The policy's recommendation for a specific user

        The random recommender will randomly select an action regardless of
        exporation setting.

        Parameters
        ----------
        user_data : np.ndarray(n_features)
            Features of the user for whom to recommend
        exploring : float
            The probability of exploring vs exploiting. Range [0,1]

        Returns
        ---------
        int
            The recommended action, in range[0, n_actions]
        """
        R = np.random.choice(self.n_actions, p = self.get_action_probabilities(user_data))
        self.obs_R = np.append(self.obs_R, R)
        self.obs_Explore = np.append(self.obs_Explore, 1)
        return R


class OptimisticRecommender(Recommender):
    """ An optimistic recommender always recommending treatment """

    def predict_proba(self, data: np.ndarray, treatment: int) -> np.ndarray:
        """ Predict outcome probabilities

        Return a distribution of effects for a given person's data and a
        specific treatment. The optimistic recommender is predicting
        probability 1 to treatment and 0 to all other actions.

        Parameters
        ----------
        user : np.ndarray(n_features)
            Features of the user for whom to predict
        treatment: int
            The actions to take

        Returns
        ---------
        np.ndarray(n_outcomes)
            The probability of each outcome given the data and the treatment
        """
        ret = np.zeros(self.n_outcomes)
        ret[1] = 1
        return ret

    def recommend(self, user_data: np.ndarray, exploring: float=0.1) -> int:
        """ The policy's recommendation for a specific user

        The optimistic recommender will always select treatment unless
        exploring.

        Parameters
        ----------
        user_data : np.ndarray(n_features)
            Features of the user for whom to recommend
        exploring : float
            The probability of exploring vs exploiting. Range [0,1]

        Returns
        ---------
        int
            The recommended action, in range[0, n_actions]
        """
        if exploring > np.random.random():
            min_p = exploring/self.n_actions
            p  = np.minimum(self.get_action_probabilities(user_data), min_p)
            p /= p.sum()
            p = np.ones(self.n_actions) / self.n_actions
            R = np.random.choice(np.arange(self.n_actions), p=p)
            self.obs_Explore = np.append(self.obs_Explore, 1)
        else:
            R = 1
            self.obs_Explore = np.append(self.obs_Explore, 0)
        self.obs_R = np.append(self.obs_R, R)
        return R


class HomeopathicRecommender(Recommender):
    """ An homeopathic recommender always recommending placebo """

    def predict_proba(self, data: np.ndarray, treatment: int) -> np.ndarray:
        """ Predict outcome probabilities

        Return a distribution of effects for a given person's data and a
        specific treatment. The homeopathic recommender is predicting
        probability 1 to placebo and 0 to all other actions.

        Parameters
        ----------
        user : np.ndarray(n_features)
            Features of the user for whom to predict
        treatment: int
            The actions to take

        Returns
        ---------
        np.ndarray(n_outcomes)
            The probability of each outcome given the data and the treatment
        """
        ret = np.zeros(self.n_outcomes)
        ret[0] = 1
        return ret

    def recommend(self, user_data: np.ndarray, exploring: float=0.1) -> int:
        """ The policy's recommendation for a specific user

        The homeopathic recommender will always select placebo unless
        exploring.

        Parameters
        ----------
        user_data : np.ndarray(n_features)
            Features of the user for whom to recommend
        exploring : float
            The probability of exploring vs exploiting. Range [0,1]

        Returns
        ---------
        int
            The recommended action, in range[0, n_actions]
        """
        if exploring > np.random.random():
            min_p = exploring/self.n_actions
            p  = np.minimum(self.get_action_probabilities(user_data), min_p)
            p /= p.sum()
            p = np.ones(self.n_actions) / self.n_actions
            R = np.random.choice(np.arange(self.n_actions), p=p)
            self.obs_Explore = np.append(self.obs_Explore, 1)
        else:
            R = 0
            self.obs_Explore = np.append(self.obs_Explore, 0)
        self.obs_R = np.append(self.obs_R, R)
        return R
