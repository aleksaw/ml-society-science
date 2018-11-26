""" A base recommender to build other recommenders on

This is a medical scenario with historical data.


General functions
- set_reward

There is a set of functions for dealing with historical data:
- fit_data
- fit_treatment_outcome
- estimate_utiltiy

There is a set of functions for online decision making
- predict_proba
- recommend
- observe

There is an analysis function
- final_analysis
"""

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix
import numpy as np
from typing import Callable

from tqdm import tqdm # For progress bar during estimation

class Recommender:
    """ Base class to extend for other recommenders """

    def __init__(self, n_actions: int, n_outcomes: int) -> "Recommender":
        """ Initialize the class

        Set the recommender with a default number of actions and outcomes.
        This is because the number of actions in historical data can be
        different from the ones that you can take with your policy.

        Parameters
        ----------
        n_actions : int
            Number of possible actions to take
        n_outcomes : int
            Number of possible outcomes

        Returns
        -------
        Recommender
            Initialized object
        """
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.reward = self._default_reward
        self._set_extreme_rewards()

    def _default_reward(self, action: int, outcome: int) -> int:
        """ Default reward

        By default, the reward is just equal to the outcome, as the actions
        play no role.

        Parameters
        ----------
        action : int
            action taken
        outcome : int
            outcome observed

        Returns
        -------
        int
            reward gained
        """
        return outcome


    def _set_extreme_rewards(self):
        """ Set the minimum and maximum possible rewards """
        self.max_reward, self.min_reward = -1e100, 1e100
        for a in range(self.n_actions):
            for o in range(self.n_outcomes):
                r = self.reward(a, o)
                if r < self.min_reward:
                    self.min_reward = r
                if r > self.max_reward:
                    self.max_reward = r



    def set_reward(self, reward: Callable[[int, int], float]):
        """ Set the reward function r(a, y)

        Parameters
        ----------
        reward : Callable[int, int]
            A new reward function to replace the default
        """
        self.reward = reward
        self._set_extreme_rewards()

    def fit_data(self, data: np.ndarray) -> None:
        """ Fit a model from patient data.

        This will generally speaking be an unsupervised model. Anything from
        a Gaussian mixture model to a neural network is a valid choice.
        However, you can give special meaning to different parts of the data,
        and use a supervised model instead.

        Parameters
        ----------
        data : np.ndarray(n_observation, n_features)
            The historical data to find pattern in
        """
        #print("Preprocessing data")
        self.X = data
        return None


    def fit_treatment_outcome(self, data: np.ndarray, actions: np.ndarray,
                              outcomes: np.ndarray, quick: bool=True) -> None:
        """ Fit a model from patient data, actions and their effects

        Here we assume that the outcome is a direct function of data and
        actions. This model can then be used in estimate_utility(),
        predict_proba() and recommend()

        Parameters
        ----------
        data : np.ndarray(n_observation, n_features)
            The historical data to find pattern in
        actions: np.ndarray(n_observations)
            The actions taken by historical recommender
        outcomes: np.ndarray(n_observations)
            The outcomes of those actions
        quick: bool
            Whether or not to fit quickly. Not always possible.
            Defaults to True
        """
        #print("Fitting treatment outcomes")
        self.X = data
        self.A = actions.ravel()
        self.Y = outcomes.ravel()

        # Initialize structures to store observations for future analysis
        self.obs_Explore = np.zeros(0, dtype=np.int)
        self.obs_R = np.zeros(0, dtype=np.int)
        self.obs_X = np.zeros((0, data.shape[1]))
        self.obs_A = np.zeros(0, dtype=np.int)
        self.obs_Y = np.zeros(0, dtype=np.int)

        return None


    def estimate_utility(self, data: np.ndarray, actions: np.ndarray,
                         outcomes: np.ndarray, policy: "Recommender"=None,
                         n_splits: int=10, n_repeats: int=5, quick: bool=True,
                         output: bool=False) -> float:
        """ Estimate the mean utility of a policy from historical data

        Estimate the mean utility of a specific policy from historical data
        (data, actions, outcome), where utility is the expected reward of the
        policy. We use the mean in order to be able to compare performance on
        differently sized samples.

        If policy is not given, simply use the average reward of the observed
        actions and outcomes.

        If a policy is given, then you can either use importance sampling, or
        use the model you have fitted from historical data to get an estimate
        of the utility.

        The policy should be a recommender that implements recommend()

        Parameters
        ----------
        data : np.ndarray(n_observation, n_features)
            The historical data to find pattern in
        actions: np.ndarray(n_observations)
            The actions taken by historical recommender
        outcomes: np.ndarray(n_observations)
            The outcomes of those actions
        policy: Recommender
            The policy to evaluate against the historical data
            Defaults to None
        n_splits: int
            The number of splits to use in KFold estimation
            Defaults to 10
        n_repeats: int
            The number of times to perform KFold estimation
            Defaults to 5
        quick: bool
            For quick fitting of the policy
            Defaults to True
        output: bool
            Whether or not to show detailed output behind the estimate
            Defaults to False

        Returns
        ---------
        float
            The estimated utility
        """
        def utility(a, y):
            return np.sum(self.reward(a, y)) / len(a)

        if policy==None:
            return utility(actions, outcomes)

        EU = 0
        acc = 0
        confusion = np.zeros((self.n_actions, self.n_actions))
        treated = 0
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
        # Create a new array that codes combinations of actions and outcomes
        # to use when splitting so we get a fairly even split
        split_sensitive = outcomes+self.n_outcomes*actions
        for train, test in tqdm(rskf.split(data, outcomes, split_sensitive)):
            Xtr, Atr, Ytr = (data[train,:], actions[train,:], outcomes[train,:])
            Xte, Ate, Yte = (data[test,:], actions[test,:], outcomes[test,:])
            policy.fit_treatment_outcome(Xtr, Atr, Ytr, quick=quick)
            A_hat = np.array([policy.recommend(Xte[i,:]) for i in range(len(Xte))])

            acc += sum((Ate.ravel()==A_hat)) / len(A_hat)
            treated += sum(A_hat==1) / len(A_hat)
            confusion += confusion_matrix(Ate.ravel(), A_hat)
            # Find the probability for each action from the policy
            p_a = np.bincount(A_hat) / len(A_hat)
            for a in range(self.n_actions):
                # We only know the outcome in the cases where our recommendation
                # equals the historical action, so we only use the data points
                # where historical and predicted actions are the same.
                mask = (Ate.ravel()==a) & (A_hat==a)
                if len(Ate[mask]) > 0:
                    # We assume we will have the same success rate of healing
                    # when treating those the historical policy didn't.
                    EU += utility(Ate[mask], Yte[mask]) * p_a[a]
        if output:
            print(f"Accuracy A vs historical A: {acc/(n_splits*n_repeats):.3f}")
            print(f"Treated: {treated/(n_splits*n_repeats):.3f}")
            print(f"Confusion: {confusion/(n_splits*n_repeats)}")

        return EU/(n_splits*n_repeats)


    def predict_proba(self, user: np.ndarray, treatment: int) -> np.ndarray:
        """ Predict outcome probabilities

        Return a distribution of effects for a given person's data and a
        specific treatment.

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
        return np.ones(self.n_outcomes) / self.n_outcomes


    def get_action_probabilities(self, user_data: np.ndarray) -> np.ndarray:
        """ Probabilities of each action being the best action

        Return a distribution of recommendations for a specific user datum.
        This is done by calculating for each action the probability of
        each outcome, then using that to calculate the expected reward and
        finally transforming it to a distribution through
        $\frac{e^{E[r]}}{\sum_A e^{E[r]}}$.

        Parameters
        ----------
        user_data : np.ndarray(n_features)
            Features of the user for whom to predict

        Returns
        ---------
        np.ndarray(n_actions)
            The probability of each action being the best, summing to 1
        """
        Er = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            p = self.predict_proba(user_data, a)
            r = self.reward(a, np.arange(self.n_outcomes))
            Er[a] = np.dot(p, r)
        Er_frac = (Er - self.min_reward) / (self.max_reward - self.min_reward)
        return Er_frac / Er_frac.sum()


    def recommend(self, user_data: np.ndarray, exploring: float=0.1,
                  strategy: str='default') -> int:
        """ The policy's recommendation for a specific user

        Return recommendation for a specific user datum. Exploring tries
        non-optimal recommendations according to current knowledge in order
        to get more knowledge and hope for better recommendations in the future.
        When we are exploting we choose the most probable action.

        Parameters
        ----------
        user_data : np.ndarray(n_features)
            Features of the user for whom to recommend
        exploring : float
            The probability of exploring vs exploiting. Range [0,1]
            Defaults to 0.1
        strategy : String
            exploration strategy. 'default' chooses the default for the
            specific recommender. 'epsilon' is available for all and chooses
            epsilon-greedy. 'prob' is available for all and chooses to explore
            more often those more likely to be good, but doesn't take into
            account the uncertainty of the estimate. 'thompson' is only not
            available for all, and explores those that *might* be good.
            'backinduction' is also not universally available. It attempts to
            look into the future by exploring possible future outcomes and
            selects the outcome with the best expected future outcome. This is
            computationally heavy!
            Defaults to 'default'
        Returns
        ---------
        int
            The recommended action, in range[0, n_actions]
        """
        #print("Recommending")
        if exploring > np.random.random():
            if strategy == 'default': strategy = 'prob'
            if strategy == 'prob':
                min_p = 0.1/self.n_actions
                p  = np.minimum(self.get_action_probabilities(user_data), min_p)
                p /= p.sum()
            elif strategy == 'epsilon':
                p = np.ones(self.n_actions) / self.n_actions
            else :
                print("No valid exploration strategy, exploiting!")
                R =  np.argmax(self.get_action_probabilities(user_data))
            R = np.random.choice(np.arange(self.n_actions), p=p)
            self.obs_Explore = np.append(self.obs_Explore, 1)
        else:
            R =  np.argmax(self.get_action_probabilities(user_data))
            self.obs_Explore = np.append(self.obs_Explore, 0)

        self.obs_R = np.append(self.obs_R, R)
        return R



    def observe(self, user: np.ndarray, action: int, outcome: int) -> None:
        """ Observe the effect of an action.

        This is an opportunity for the recommender to refit the models, to
        take the new information into account.

        Parameters
        ----------
        user : np.ndarray(n_features)
            Features of the user
        action : int
            Action taken
        outcome: int
            Outcome observed
        """
        self.X = np.append(self.X, user.reshape(1,-1), axis=0)
        self.A = np.append(self.A, action)
        self.Y = np.append(self.Y, outcome)
        self._store_observation(user, action, outcome)
        return None

    def _store_observation(self, user: np.ndarray, action: int,
                           outcome: int) -> None:
        """ Store the observation for final analysis

        Parameters
        ----------
        user : np.ndarray(n_features)
            Features of the user
        action : int
            Action taken
        outcome: int
            Outcome observed
        """
        self.obs_X = np.append(self.obs_X, user.reshape(1,-1))
        self.obs_A = np.append(self.obs_A, action)
        self.obs_Y = np.append(self.obs_Y, outcome)


    def final_analysis(self) -> None:
        """ Perform final analysis on the policy

        After all the data has been obtained, do a final analysis. This can
        consist of a number of things:
        1   Recommending a specific fixed treatment policy
        2.  Suggesting looking at specific genes more closely
        3.  Showing whether or not the new treatment might be better than the
            old, and by how much.
        4.  Outputting an estimate of the advantage of gene-targeting
            treatments versus the best fixed treatment
        """
        if (self.obs_R == self.obs_A).sum() < len(self.obs_R):
            # These should all be equal, if they aren't I don't know where the
            # data is coming from
            print("Compromised data")
        else:
            actioncounts = np.bincount(self.obs_A, minlength=self.n_actions)
            print("Actions by count: {}".format(actioncounts))
            success_by_action = np.zeros(self.n_actions)
            for i in range(len(self.obs_A)):
                success_by_action[self.obs_A[i]] += self.obs_Y[i]
            print("Successrate by action: {}".format(success_by_action / actioncounts))
            penalties = np.zeros(self.n_actions); penalties[1:] += 0.1
            mean_r = (success_by_action-penalties*actioncounts) / actioncounts
            print("Mean reward by action {}".format(mean_r))
            print("Matrix of actions vs outcomes, outcome is row, action is column")
            print(confusion_matrix(self.obs_Y, self.obs_A)[:self.n_outcomes, :self.n_actions])
            # Store for each action taken whether it was done exploratory or exploitatively
            print("Actions by exploration")
            a_by_explr = np.zeros(self.n_actions)
            for a, e in zip(self.obs_R, self.obs_Explore):
                a_by_explr[a] += e
            print(a_by_explr)
        return None
