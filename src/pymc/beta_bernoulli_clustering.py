# See https://docs.pymc.io/notebooks/getting_started.html for pymc documentation

import pymc3 as mc
import numpy as np
import matplotlib.pyplot as plt
import theano
import itertools

model = mc.Model()

n_users = 8
n_movies = 4
ratings = np.zeros([n_users, n_movies]);
for u in range(n_users):
   ## Generate data from two clusters with equal probability
    if (np.random.uniform() < 0.5):
        mean = np.array([0.2, 0.3, 0.4, 0.9]);
    else:
        mean = np.array([0.8, 0.7, 0.2, 0.5]);

    ## Fill in data for the user. Currently no missing data allowed.
    for m in range(n_movies):
        if (np.random.uniform() < mean[m]):
            ratings[u,m] = 1
        else:
            ratings[u,m] = 0

# setup model
model = mc.Model()
n_clusters = 2
with model:
    # We assume a finite number of possible values for the clusters
    # The prior on the multinomial cluster distribution is a Dirichlet
    zeta = mc.Dirichlet('zeta', a=np.ones(n_clusters))

    # Latent cluster of each user
    # A single sample from a multinomial is called 'categorial' in pymc
    clusters = [mc.Categorical('cluster_{}'.format(i),  p=zeta) for i in range(n_users)]

    # The model assumption is that every movie's rating is sampled from the appropriate cluster
    # We define a C*M matrix of theta parameters for this
    theta = mc.Beta('theta', alpha =1, beta = 1, shape=(n_clusters, n_movies))


# Tried to use a custom likelihood but failed
#    def rating_dist(theta, cluster, ratings):
#        log_likelihood = 0
#        print ("users: ", n_users, "movies:", n_movies)
#        for u in range(n_users):
#            for m in range(n_movies):
#                assigned_theta = theta[cluster[u], m]
#                if (ratings[u,m] == 1):
#                    log_likelihood += log(assigned_theta)
#                else:
#                    log_likelihood += log(1 - assigned_theta)
#        return log_likelihood
#
#    rating_loglike = mc.DensityDist('rating_loglike', rating_dist, observed={'theta':theta, 'cluster':cluster, 'ratings':ratings})


## Generate samples
with model:
    step1 = mc.NUTS()
    tr = mc.sample(10000, step=[step1])

mc.traceplot(tr)
plt.show()
"""
# See https://docs.pymc.io/notebooks/getting_started.html for pymc documentation

import pymc3 as mc
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as tt
from theano.compile.ops import as_op
theano.config.optimizer='fast_compile'
theano.config.exception_verbosity='high'
theano.config.compute_test_value = 'warn'

model = mc.Model()

n_users = 8
n_movies = 4
ratings = np.zeros([n_users, n_movies]);
for u in range(n_users):
   ## Generate data from two clusters with equal probability
    if (np.random.uniform() < 0.5):
        mean = np.array([0.2, 0.3, 0.4, 0.9]);
    else:
        mean = np.array([0.8, 0.7, 0.2, 0.5]);

    ## Fill in data for the user. Currently no missing data allowed.
    for m in range(n_movies):
        if (np.random.uniform() < mean[m]):
            ratings[u,m] = 1
        else:
            ratings[u,m] = 0

# setup model
model = mc.Model()
n_clusters = 2

#@as_op(itypes=[tt.bscalar, tt.bscalar, tt.dmatrix, tt.lvector, tt.dmatrix], otypes=[tt.dscalar])

with model:
    # We assume a finite number of possible values for the clusters
    # The prior on the multinomial cluster distribution is a Dirichlet
    zeta = mc.Dirichlet('zeta', a=np.ones(n_clusters), shape=n_clusters)

    # Latent cluster of each user
    # A single sample from a multinomial is called 'categorial' in pymc
    cluster = mc.Categorical('cluster',  p=zeta, shape=n_users)

    # The model assumption is that every movie's rating is sampled from the appropriate cluster
    # We define a C*M matrix of theta parameters for this
    theta = mc.Beta('theta', alpha =1, beta = 1, shape=(n_clusters, n_movies))

    def rating_dist(n_users, n_movies, theta, cluster, ratings):
        log_likelihood = 0
        print ("users: ", n_users, "movies:", n_movies)
        #for u in range(n_users):
            #for m in range(n_movies):
        for u in range(int(n_users.value)):
            for m in range(int(n_movies.value)):
                #print(theano.pp(cluster))
                #print(cluster[u].type)
                #print(cluster[u].eval({cluster:ratings[:,m].eval()}))
                #print(theano.pp(cluster[u].type))
                #print(type(theta))
                assigned_theta = theta[int(round(cluster.eval({zeta:0})[u])), m]
                #assigned_theta = theta[cluster[u], m]
                if (ratings[u,m] == 1):
                    log_likelihood += tt.log(assigned_theta)
                else:
                    log_likelihood += tt.log(1 - assigned_theta)
        return log_likelihood

    #def logp(r):
        #return rating_dist(tt.as_tensor_variable(n_users), tt.as_tensor_variable(n_movies), theta, cluster, r)

    #rating_loglike = mc.DensityDist('rating_loglike', logp, observed=ratings)
    rating_loglike = mc.DensityDist('rating_loglike', rating_dist, observed={'n_users': n_users, 'n_movies': n_movies, 'theta': theta, 'cluster': cluster, 'ratings': ratings})


## Generate samples
with model:
    step1 = mc.NUTS(vars=[zeta, theta])
    step2 = mc.ElemWiseCategorical(vars=[cluster], values=[0,1,2])
    tr = mc.sample(10000, step=[step1, step2])

mc.traceplot(tr)
plt.show()
"""
