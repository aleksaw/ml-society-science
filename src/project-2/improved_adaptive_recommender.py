""" Improved Adaptive Recommender, combining approaches for optimal learning """

from recommender import Recommender

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np

class ImprovedAdaptiveRecommender(Recommender):
    def __init__(self, n_actions: int, n_outcomes: int, quick_fits: bool=True,
                 refit_trigger: float=1.1) -> "Recommender":
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
        quick_fits : bool
            Whether to fit the models quickly or slowly.
            Defaults to True
        refit_trigger : float
            Increase over previous data in model before we refit.
            Defaults to 1.1

        Returns
        -------
        Recommender
            Initialized object
        """
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.reward = self._default_reward
        self.quick_fits = quick_fits
        self.refit_trigger = 1.1

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
        super().fit_treatment_outcome(data, actions, outcomes)
        self.model = [self._fit_model(a, quick=quick)
                      for a in range(self.n_actions)]



    def _fit_model(self, action: int, quick: bool=True) -> 'Classifier':
        """ Fit a model from patient data for a specific action

        We use pre-stored patient data to fit a suitable classifier for each
        action.
        If we have too few observations of each outcome we can't fit a
        Logistic Classifier, so we use one that supports is.

        Parameters
        ----------
        action: int
            The action to fit a model for
        quick: bool
            Whether or not to fit quickly.
            Defaults to True

        Returns
        ----------
        Classifier: A classifier for the action
        """
        a = action

        # Set defaults and triggerpoints
        # Number of observations in each class needed to trigger Logistic
        bandit_logistic_trigger = 1
        # Number of observations needed to fit slowly if quick==False
        quick_trigger = 10
        # Regularization parameter to use in quick fitting
        quick_c = 100000000.

        # Settings for slow-fit, selecting amount of regularization
        # Ratio of smallest outcome class to all observations below which
        # we need to use custom accuracy score.
        acc_rr_threshold = 0.2
        # Number of observations needed to use 10-fold CV
        FiveFold_threshold = 200
        fits = 200 #n_repeats * n_folds
        # Regularization parameters to choose from
        Cs = np.logspace(-1, 6, 30)

        # Custom accuracy score for unbalanced outcome classes
        def acc_rr(y_pred, y_te):
            tp = (y_pred&yte).sum()
            fp = ((y_pred==1)&(y_te==0)).sum()
            fn = ((y_pred==0)&(y_te==1)).sum()
            # If there are no 1's in either array we get no result, but then it's a match.
            return (tp / (tp+fp+fn)) if (tp+fp+fn)>0 else 1


        X_A = self.X[(self.A==a),:]
        Y_A = self.Y[(self.A==a)]
        smallest_class_of_outcomes = np.min(np.bincount(Y_A, minlength=self.n_outcomes))

        if smallest_class_of_outcomes == 0:
            # We can't fit Logistic with only one class in data
            return BanditClassifier(self.n_outcomes).fit(X_A, Y_A)

        if smallest_class_of_outcomes < quick_trigger:
            # We're likely to run into trouble fitting classifier in
            # KFold, so we drop that.
            quick = True

        if quick:
            clf = LogisticClassifier(refit_trigger=self.refit_trigger,
                                     C=quick_c, solver='lbfgs', max_iter=2000)
            return clf.fit(X_A, Y_A)

        if smallest_class_of_outcomes < acc_rr_threshold * len(Y_A):
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
                clf = LogisticRegression(C=c, solver='lbfgs', max_iter=2000)
                y_pred = clf.fit(Xtr, ytr).predict(Xte)
                accuracy[j] = measure(y_pred, yte)
                j += 1
            accuracies[i] = np.mean(accuracy)

        # I'll go with highest accuracy. We could have gone most parsimonious
        # within 1se f.ex, but test indicate less regularization is better.
        best_c = Cs[np.argmax(accuracies)]
        clf = LogisticClassifier(refit_trigger=self.refit_trigger, C=best_c,
                                 solver='lbfgs', max_iter=2000)
        return clf.fit(X_A, Y_A)


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
        return self.model[treatment].predict_proba(user.reshape(1,-1))

    def sample_proba(self, user: np.ndarray, treatment: int) -> np.ndarray:
        """ Sample from the distribution of outcome probabilities

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
        return self.model[treatment].sample_proba(user.reshape(1,-1))

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
            account the uncertainty of the estimate. 'thompson' explores those
            that *might* be good. 'backinduction' attempts to look into the
            future by exploring possible future outcomes and selects the
            outcome with the best expected future outcome. This is
            computationally heavy!
            Defaults to 'default', which for this recopmmender is 'thompson'

        Returns
        ---------
        int
            The recommended action, in range[0, n_actions]
        """
        #print("Recommending")
        if exploring > np.random.random():
            if strategy == 'default': strategy = 'thompson'
            if strategy == 'thompson':
                # This is the same as exploiting get_action_probabilities except
                # with sample_proba instead of predict_proba
                Er = np.zeros(self.n_actions)
                for a in range(self.n_actions):
                    p = self.sample_proba(user_data, a)
                    r = self.reward(a, np.arange(self.n_outcomes))
                    Er[a] = np.dot(p, r)
                Er_frac = (Er - self.min_reward) / (self.max_reward - self.min_reward)
                R =  np.argmax(Er_frac / Er_frac.sum())
            elif strategy == 'backinduction':
                # Set parameters
                # The depth of the backwards induction. I.e. how far into the
                # future to try to look. Beware, complexity is exponential.
                # Looking deep will take long.
                updates_per_level = self.n_actions * self.n_outcomes
                max_updates = 300
                # upl**depth=updates<max_updates => depth<log(mu)/log(upl)
                depth = np.floor(np.log(max_updates) / np.log(updates_per_level))
                # The horizon to maximize for. This will influence the value of
                # exploring vs exploiting.
                horizon = 100

                # Define some useful functions
                def update_priors(priors, action, outcome):
                    ret = []
                    for a, p in enumerate(priors):
                        if action == a:
                            ret.append(p.update_for_backwards_induction(user_data, outcome))
                        else:
                            ret.append(p)
                    return ret

                def backwards_induction(priors, depth, T):
                    EU = np.zeros(len(priors))
                    for a, p in enumerate(priors):
                        prob = p.predict_proba(user_data.reshape(1,-1))[1]
                        EU[a] = self.reward(a, prob)
                        if depth > 0:
                            induce = lambda outcome: \
                                        backwards_induction( \
                                            update_priors(priors, a, outcome), \
                                            depth - 1, T - 1)
                            (r1, a1), (r0, a0) = (induce(o) for o in (1, 0))
                            EU[a] += prob * r1
                            EU[a] += (1-prob) * r0
                        else:
                            EU[a] = prob * T
                    return EU.max(), np.argmax(EU) # Expected Utility, Action

                # Get results from induction and make recommendation
                EU, R = backwards_induction(self.model, depth, horizon)
            else:
                return super().recommend(user_data, exploring=1.0, strategy=strategy)
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
        # Store observations for final analysis
        super()._store_observation(user, action, outcome)
        self.model[action].update_fit(user, outcome)
        # Check how much we have increased out dataset and update models if necessary
        if getattr(self.model[action], 'non_contextual', False): # BanditClassifier
            smallest_class_of_outcomes = np.min(np.bincount(self.Y[self.A==action], minlength=self.n_outcomes))
            if smallest_class_of_outcomes > 0:
                self.model[action] = self._fit_model(action, quick=self.quick_fits)

        return None

    def final_analysis(self, quiet: bool=False) -> None:
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
        if not quiet:
            model_is_NC = [getattr(self.model[a], 'non_contextual', False)
                           for a in range(self.n_actions)]
            lrs = self.n_actions - sum(model_is_NC)
            print("Logistic Recommenders: {}".format(lrs))
        return super().final_analysis(quiet=quiet)



class LogisticClassifier:
    def __init__(self, refit_trigger=1.1, **kwargs):
        self.clf = LogisticRegression(**kwargs)
        self.refit_trigger = refit_trigger
        self.kwargs = kwargs
        self.boots = 100

    def fit(self, X, y):
        self.clf.fit(X,y)
        self.data_in_fit = X.shape[0]
        self.data_for_model = self.data_in_fit
        self.X = X
        self.y = y

        if X.shape[0] < X.shape[1]: # X.T@V@X will be singular
            self.bigdata = False
            self.bootclf = [None for _ in range(self.boots)]
            for i in range(self.boots):
                zeros = (y==0).sum();            ones = (y==1).sum()
                p_zero = 1*(y==0).ravel()/zeros; p_one = 1*(y==1).ravel()/ones
                sample0 = np.random.choice(range(X.shape[0]), p=p_zero, size=zeros)
                sample1 = np.random.choice(range(X.shape[0]), p=p_one, size=ones)
                sampleX = np.append(X[sample0,:], X[sample1,:], axis=0)
                sampley = np.append(y[sample0], y[sample1], axis=0)
                self.bootclf[i] = LogisticRegression(**self.kwargs).fit(sampleX, sampley)
        else:
            self.bigdata = True
            self.beta = np.append(self.clf.intercept_, self.clf.coef_)
            X_design = np.append(np.ones((X.shape[0],1)), X, axis=1)
            self.cov = np.linalg.inv(X_design.T @ X_design)

        return self

    def update_fit(self, user, outcome):
        self.data_for_model += 1
        self.X = np.append(self.X, user.reshape(1,-1), axis=0)
        self.y = np.append(self.y, outcome)
        if self.data_for_model > self.data_in_fit * self.refit_trigger:
            self.fit(self.X, self.y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X).ravel()

    def sample_proba(self, X):
        if self.bigdata:
            X = np.append([1], X)
            Xb = np.dot(X, self.beta)
            std = np.sqrt(X @ self.cov @ X.T)
            # If a is too big exp(a) gives nan
            a = min(100, np.random.normal(Xb, std))
            # This now only supports 2 outcomes.
            p1 = np.exp(a) / (1+np.exp(a))
            return np.array([(1-p1), p1])
        else:
            model = np.random.choice(range(self.boots))
            return self.bootclf[model].predict_proba(X).ravel()

    def update_for_backwards_induction(self, user, outcome):
        if self.data_for_model > self.data_in_fit * self.refit_trigger:
            clf =  LogisticClassifier(refit_trigger=self.refit_trigger, **self.kwargs)
            return clf.fit(self.X, self.y)
        else:
            return self



class BanditClassifier:
    def __init__(self, n_outcomes, alphas=np.empty(0)):
        self.non_contextual = True # Used to identify classifier
        self.n_outcomes = n_outcomes
        if len(alphas) == 0:
            self.alphas = np.ones(n_outcomes)
        else:
            self.alphas = alphas

    def fit(self, data, outcomes):
        self.data_in_fit = data.shape[0]
        self.data_for_model = self.data_in_fit
        self.alphas += np.bincount(outcomes, minlength=self.n_outcomes)
        return self

    def update_fit(self, user, outcome):
        self.alphas[outcome] += 1

    def predict_proba(self, data):
        return self.alphas / self.alphas.sum()

    def sample_proba(self, X):
        return np.random.dirichlet(self.alphas)

    def update_for_backwards_induction(self, user, outcome):
        alpha = self.alphas
        alpha[outcome] += 1
        return BanditClassifier(self.n_outcomes, alpha)
