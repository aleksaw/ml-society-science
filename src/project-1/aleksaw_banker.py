import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats

class NameBanker:
    def __init__(self):
        # Instance variables
        # Data
        self.X = np.array([[0]])
        self.y = np.array([0])
        # Scalers
        self.scaler = StandardScaler()
        self.pca = PCA()
        # Classifiers
        # self.classifiers
        # Parameters
        self.parameters_set = False
        self.lr_components = 10
        self.nn_alpha = 1e-4
        self.nn_hidden_components = (70)
        self.knn_k = 40
        self.method_weights = np.array([0,0,1])#np.ones(3) / 3
        # Interest rate
        self.rate = 0.005
        # Interesting X indices (perhaps these ought to be passed as arguments)
        self.ix_amount = 4
        self.ix_duration = 1

    def __choose_parameters(self, print_progress=False):
        if print_progress: print("Choosing parameters")
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.25)
        scaler = StandardScaler()
        X_test_unscaled = X_test
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
        n_features = np.shape(self.X)[1]

        def utility(loan, rate, time, outcome):
            return loan * (-1 if outcome==2 else (1+rate)**time - 1)
        def expected_utility(loan, rate, time, prob):
            return loan * (prob * (1+rate)**time - 1)

        # Neural network
        # Choose alpha
        if print_progress: print("Choosing parameter 1/5")
        n_folds = 10
        alphas = np.logspace(-5, 0, 21)
        accuracies = np.zeros(len(alphas))
        deviations = np.zeros(len(alphas))
        i = 0
        for alpha in alphas:
            accuracy = np.zeros(n_folds)
            j = 0
            for train, test in KFold(n_splits=n_folds).split(X_train):
                classifier = MLPClassifier(solver='lbfgs', alpha=alpha,
                                           hidden_layer_sizes=(70))
                classifier.fit(X_train[train], y_train[train])
                y_pred = classifier.predict(X_train[test])
                accuracy[j] = accuracy_score(y_train[test], y_pred)
                j += 1
            accuracies[i] = np.mean(accuracy)
            deviations[i] = stats.sem(accuracy)
            i += 1

        self.nn_alpha = alphas[np.argmax(accuracies)]
        print(f"Alpha: {self.nn_alpha}")

        #Choose layers
        if print_progress: print("Choosing parameter 2/5")
        n_folds = 10
        hidden_components = [(50), (70), (100),
                             (50, 10), (70, 10), (100, 10),
                             (50, 30), (70, 30), (100, 30),
                             (50, 50), (70, 50), (100, 50),
                             (50, 70), (70, 70), (100, 70),
                             (50, 30, 10), (70, 30, 10), (100, 30, 10),
                             (16, 4, 2)]
        accuracies = np.zeros(len(hidden_components))
        deviations = np.zeros(len(hidden_components))
        i = 0
        for components in hidden_components:
            accuracy = np.zeros(n_folds)
            j = 0
            for train, test in KFold(n_splits=n_folds).split(X_train):
                classifier = MLPClassifier(solver='lbfgs', alpha=self.nn_alpha,
                                           hidden_layer_sizes=components)
                classifier.fit(X_train[train], y_train[train])
                y_pred = classifier.predict(X_train[test])
                accuracy[j] = accuracy_score(y_train[test], y_pred)
                j += 1
            accuracies[i] = np.mean(accuracy)
            deviations[i] = stats.sem(accuracy)
            i += 1

        self.nn_hidden_components = hidden_components[np.argmax(accuracies)]
        print(f"Layers: {self.nn_hidden_components}")

        # Logistic regression
        if print_progress: print("Choosing parameter 3/5")
        n_bootstraps = 100
        n_folds = 10
        accuracies = np.zeros(n_features-1)
        deviations = np.zeros(n_features-1)
        pca = PCA()
        pca.fit(X_train)
        X_train_pca = pca.transform(X_train)
        for components in range(1, n_features):
            accuracy = np.zeros(n_bootstraps * n_folds)
            i = 0
            for train, test in KFold(n_splits=n_folds).split(X_train_pca):
                for B in range(n_bootstraps):
                    Xtr, ytr = resample(X_train_pca[train, :components],
                                        y_train[train])
                    Xte, yte = resample(X_train_pca[test, :components],
                                        y_train[test])
                    y_pred = LogisticRegression().fit(Xtr, ytr).predict(Xte)
                    accuracy[i] = accuracy_score(y_pred, yte)
                    i += 1
            accuracies[components-1] = np.mean(accuracy)
            deviations[components-1] = stats.sem(accuracy)

        self.lr_components = np.argmax(accuracies-deviations)
        print(f"Components: {self.lr_components}")

        # K-Nearest Neighbours
        if print_progress: print("Choosing parameter 4/5")
        n_folds = 10
        # Ks from 1 to the maximum number of point in the cv training set
        # This could be a huge number, perhaps we should limit it somehow?
        # ks = np.arange(1, int(((n_folds-1)/n_folds)*len(X_train)))
        # Perhaps it's better to choose differently. Perhaps not every k
        # in the interval, and perhaps we don't need the upper half?
        n_tr_points = int(((n_folds-1)/n_folds)*len(X_train))
        # Start at 1 (10**0), end at n_tr_point/2 (10**np.log10(int(n_tr_point/2)))
        ks = np.logspace(0, np.log10(int(n_tr_points/2)), 50)
        accuracies = np.zeros(len(ks))
        deviations = np.zeros(len(ks))
        n_bootstraps = 100
        j = 0
        for k in ks:
            k = int(k)
            accuracy = np.zeros(n_folds*n_bootstraps)
            i = 0
            for train, test in KFold(n_splits=n_folds).split(X_train):
                for B in range(n_bootstraps):
                    Xtr, ytr = resample(X_train[train], y_train[train])
                    Xte, yte = resample(X_train[test], y_train[test])
                    y_pred = KNeighborsClassifier(n_neighbors=k).fit(Xtr, ytr).predict(Xte)
                    accuracy[i] = accuracy_score(y_pred, yte)
                    i += 1
            accuracies[j] = np.mean(accuracy)
            deviations[j] = stats.sem(accuracy)
            j += 1

        self.knn_k = int(ks[np.argmax(accuracies-deviations)])
        print(f"k: {self.knn_k}")

        # Compare methods
        if print_progress: print("Choosing parameter 5/5")
        classifiers = [MLPClassifier(solver='lbfgs', alpha=self.nn_alpha,
                            hidden_layer_sizes=self.nn_hidden_components).fit(X_train, y_train),
                       LogisticRegression().fit(X_train_pca[:, :self.lr_components], y_train),
                       KNeighborsClassifier(n_neighbors=self.knn_k).fit(X_train, y_train)]

        probabilities = np.zeros((len(X_test), len(classifiers)),
                                 dtype=np.float64)
        for i in range(len(X_test)):
            for j in range(len(classifiers)):
                if j == 1:
                    Xs = pca.transform(X_test[i].reshape(1, -1))[:, :self.lr_components]
                else:
                    Xs = X_test[i].reshape(1, -1)
                probabilities[i, j] = classifiers[j].predict_proba(Xs)[:, y_test[i]-1]

        max_loan = X_test_unscaled[:,self.ix_amount].max()

        n_folds = 10
        n_ways_of_weighing = 10
        weights = np.zeros((n_ways_of_weighing, len(classifiers)))
        weighing_utility = np.zeros((n_folds, n_ways_of_weighing))
        n = 0
        for train, test in KFold(n_splits=n_folds).split(X_test):
            # Fit weights
            weights[0, :] = np.ones(len(classifiers))/len(classifiers)
            weights[1, :] = np.ones(len(classifiers))/len(classifiers)
            weights[2, :] = np.ones(len(classifiers))/len(classifiers)
            weights[3, :] = np.ones(len(classifiers))/len(classifiers)
            weights[4, :] = np.ones(len(classifiers))/len(classifiers)
            weights[5, :] = np.zeros(len(classifiers))
            weights[6, :] = np.ones(len(classifiers))/len(classifiers)
            weights[7, :] = np.array([1, 0, 0])
            weights[8, :] = np.array([0, 1, 0])
            weights[9, :] = np.array([0, 0, 1])

            for i in train:
                loan, time = X_test_unscaled[i, (self.ix_amount, self.ix_duration)]
                loan /= max_loan
                prob = probabilities[i, :] if y_test[i] == 1 else 1-probabilities[i, :]
                util = np.array([utility(loan, self.rate, time, y_test[i])
                                 if eu > 0 else 0
                                 for eu in expected_utility(loan, self.rate, time, prob)])

                weights[0, :] *= probabilities[i, :]
                weights[0, :] /= np.sum(weights[0, :])
                weights[1, :] *= 1 + util
                weights[1, :] /= np.sum(weights[1, :])
                weights[2, :] *= 2 + util
                weights[2, :] /= np.sum(weights[2, :])
                weights[3, :] *= 3 + util
                weights[3, :] /= np.sum(weights[3, :])
                weights[4, :] *= 4 + util
                weights[4, :] /= np.sum(weights[4, :])
                weights[5, :] += util

            # Adjust away possible negative numbers (if you lose money, you get 0 weight)
            weights[5, :] -= min(0, np.min(weights[5, :]))
            weights[5, :] /= np.sum(weights[5, :])

            # Test performance
            for i in test:
                loan, time = X_test_unscaled[i, (self.ix_amount, self.ix_duration)]
                prob = probabilities[i, :] if y_test[i] == 1 else 1-probabilities[i, :]
                prob = prob @ weights.T
                weighing_utility[n, :] += np.array([utility(loan, self.rate, time, y_test[i])
                                                    if eu > 0 else 0
                                                    for eu in expected_utility(loan, self.rate, time, prob)])

            n += 1
        self.method_weights = weights[np.argmax(np.mean(weighing_utility, axis=0)
                                                - stats.sem(weighing_utility))]
        if print_progress: print(f"Weights: {self.method_weights}")
        self.parameters_set = True
        if print_progress: print("Parameters chosen")

    # Fit the model to the data.  You can use any model you like to do
    # the fit, however you should be able to predict all class
    # probabilities
    def fit(self, X, y, set_params=False):
        # self.parameters_set = True
        # Convert X and y to array
        self.ix_amount = X.columns.get_loc('amount')
        self.ix_duration = X.columns.get_loc('duration')
        self.X = X.values
        self.y = y.values.ravel()
        if set_params or not self.parameters_set:
            self.__choose_parameters(True)
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        self.pca.fit(X)
        X_pca = self.pca.transform(X)[:, :self.lr_components]

        self.classifiers = [MLPClassifier(solver='lbfgs', alpha=self.nn_alpha, \
                                          hidden_layer_sizes=self.nn_hidden_components).\
                                          fit(X, y),
                            LogisticRegression().fit(X_pca, y),
                            KNeighborsClassifier(n_neighbors=self.knn_k).fit(X, y)]

    # set the interest rate
    def set_interest_rate(self, rate, refit=False):
        self.rate = rate
        if refit:
            self.fit(self.X, self.y)
        return

    # Predict the probability of repayment for a specific person with data x
    def predict_proba(self, x):
        x = self.scaler.transform(x.reshape(1, -1))
        x_pca = self.pca.transform(x)[:, :self.lr_components]
        mlp_prob = self.classifiers[0].predict_proba(x)
        log_prob = self.classifiers[1].predict_proba(x_pca)
        knn_prob = self.classifiers[2].predict_proba(x)
        return (mlp_prob[0,0] * self.method_weights[0] +
                log_prob[0,0] * self.method_weights[1] +
                knn_prob[0,0] * self.method_weights[2])

    # THe expected utility of granting the loan or not. Here there are two actions:
    # action = 0 do not grant the loan
    # action = 1 grant the loan
    #
    # Make sure that you extract the length_of_loan from the
    # 2nd attribute of x. Then the return if the loan is paid off to you is amount_of_loan*(1 + rate)^length_of_loan
    # The return if the loan is not paid off is -amount_of_loan.
    #def expected_utility(loan, rate, time, prob):
    #    return loan * (prob * (1+rate)**time - 1)

    def expected_utility(self, x, action):
        if action == 1:
            t = x[self.ix_duration]
            l = x[self.ix_amount]
            p = self.predict_proba(x)
            return l * (p * (1+self.rate)**t - 1)
        else:
            return 0
    # Return the best action. This is normally the one that maximises expected utility.
    # However, you are allowed to deviate from this if you can justify the reason.
    def get_best_action(self, x):
        # Convert Dataframe to array
        x = x.values
        return 1 if self.expected_utility(x,1) >= self.expected_utility(x,0) else 0
