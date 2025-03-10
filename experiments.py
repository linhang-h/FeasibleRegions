#!/usr/bin/env python
# coding: utf-8

# In[90]:


from problems.HS100 import HS100, CallableClass
import methods.methods as methods
from typing import Tuple, Union, List, Type
from copy import deepcopy
import pandas as pd
import numpy as np
from numpy import number
from scipy.stats.qmc import LatinHypercube as lhs
from scipy.stats.qmc import scale
import scipy.stats as stats
from smt.surrogate_models import KRG
import random
import itertools


# In[91]:


def sample_sites(problem: dict, n_samples: int, seed=random.randint(1,1000)) -> pd.DataFrame:
    """ Simple wrapper to sample sites
    
    Parameters
    ----------
    problem : dict
        The variables and constraints given

    n_sample: int
        The number of sites to generare

    seed: int
        Seed for the LHS sampling
   
    Returns
    -------
    pd.DataFrame
        The DataFrame of sites
    """

    variables = list(problem["variables"].keys())
    nind = len(variables)
    # Get all the bounds for the variables
    bounds = np.array([problem["variables"][var]["bounds"] for var in variables])
    # Generate the experiment 
    lhs_instance = lhs(
        nind,
        scramble=True,
        strength=1,
        optimization=None,
        seed=seed,
    )
    # Get the experiment
    normalized_array = lhs_instance.random(n_samples)
    # Scale using the bounds
    exp_array = scale(normalized_array, bounds[:, 0], bounds[:, 1])
    # Create the DataFrame
    exp_df = pd.DataFrame(data=exp_array, columns=variables)
    return(exp_df)


# In[92]:


def evaluate_sites(local_eval: CallableClass, exp_data: pd.DataFrame, verbose=False):
    """ evaluates sites passed and returns their constraint violation data

        Parameters
        ----------
        local_eval : CallableClass
            The example evaluator to use
        
        n_sample: int
            The number of sites to generare
   
        Returns
        -------
        pd.DataFrame
            The DataFrame of evaluated experimental sites with constraint violation
    """
    eps = 1e-6
    local_problem = local_eval.problem()
    local_eval(exp_data)
    my_constraint_calculator = methods.ConstraintCalculator(local_problem)
    exp_data['__conviol__'] = my_constraint_calculator(exp_data)
    exp_data['__State__'] = pd.cut(exp_data['__conviol__'], [-np.inf, eps, 100*eps, np.inf], include_lowest=True, labels=['Feasible', 'Nearly Feasible', "Infeasible"]).astype("str")
    feasible_sites = (exp_data['__State__'] == 'Feasible').sum()
    percentage = feasible_sites/len(exp_data) * 100
    if verbose: 
        print(f"Test evaluator {local_eval.name} with {len(local_problem['variables'])} variables {percentage}% feasible sites")
    return (exp_data)


# In[93]:


def experiment_1(local_eval: CallableClass, n_samples: int) -> pd.DataFrame:
    exp_data = sample_sites(local_eval.problem(), n_samples)
    exp_data = evaluate_sites(local_eval, exp_data, verbose=False)
    return exp_data


# In[94]:


hs100 = HS100()
problem = hs100.problem()
nind = len(problem['variables'])
num_sites = 50
experiment_1(hs100, num_sites)


# In[95]:


def experiment_2(local_eval: CallableClass, num_sites_training: int, num_sites_testing : int, verbose=True) -> pd.DataFrame:
    """ Wrapper to create an experiment, evaluate the passed in function, create a surrogate model, evaluate model

        Parameters
        ----------
        local_eval : CallableClass
            The example evaluator to use
        
        n_sample_training: int
            The number of sites to train the sm on

        n_sample_testing: int
            The number of sites to test the sm
   
        Returns
        -------
        pd.DataFrame
            The DataFrame of evaluated experimental sites with constraint violation
    """
    # get training data
    sample_data = sample_sites(local_eval.problem(), num_sites_training)
    exp_data = evaluate_sites(local_eval, sample_data, verbose=False)
    variables = list(local_eval.problem()["variables"].keys())
    xt = exp_data[variables].to_numpy()
    yt = exp_data['__conviol__'].to_numpy()

    # create model
    sm = KRG(theta0=[1e-2], print_global=False)
    sm.set_training_values(xt, yt)
    sm.train()

    # get testing data
    x = sample_sites(local_eval.problem(), num_sites_testing**2).to_numpy()
    y = sm.predict_values(x)
    feasible_points = pd.DataFrame(data=x, columns=variables)
    feasible_points['coviol'] = y
    feasible_points = feasible_points.sort_values(by='coviol').reset_index(drop=True)[:num_sites_testing].drop('coviol', axis=1)
    
    # evaluate model on testing data
    feasible_points = evaluate_sites(local_eval, feasible_points, verbose=verbose)
    return (feasible_points) 
    


# In[96]:


hs100 = HS100()
problem = hs100.problem()
nind = len(problem['variables'])
num_sites = 25
test_data = experiment_2(hs100, num_sites, num_sites)
# print((test_data['__State__'] == 'Feasible').sum())
test_data


# ## Experiment 3
# 1. Generate training data once and train
# 2. Regenerate sites, pass through filter, take the top sites
# 3. Train again, and repeat

# In[97]:


# multi-run training
def experiment_3(local_eval, sites_per_run, runs, verbose=False):
    problem = hs100.problem()
    nind = len(problem['variables'])

    # create model
    sm = KRG(theta0=[1e-2], print_global=False)
    
    for i in range(runs):
        if i == 0:
            # get first run's data
            sample_data = sample_sites(local_eval.problem(), sites_per_run)
            exp_data = evaluate_sites(local_eval, sample_data, verbose=False)
            training_data = exp_data
        else:
            training_data = pd.concat([training_data, exp_data], axis=0).drop_duplicates()
        variables = list(local_eval.problem()["variables"].keys())
        xt = training_data[variables].to_numpy()
        yt = training_data['__conviol__'].to_numpy()
        
        # train model
        sm.set_training_values(xt, yt)
        sm.train()
    
        # get next run's data
        x = sample_sites(local_eval.problem(), sites_per_run**2).to_numpy()
        y = sm.predict_values(x)
        exp_data = pd.DataFrame(data=np.concatenate((x,y), axis=1), columns=variables + ['conviol'])
        exp_data = exp_data.sort_values(by='conviol').reset_index(drop=True)[:sites_per_run].drop('conviol',axis=1)
        # evaluate model on testing data
        exp_data = evaluate_sites(local_eval, exp_data)
    return exp_data


# In[98]:


hs100 = HS100()
problem = hs100.problem()
nind = len(problem['variables'])
num_sites = 20
runs = 3
experiment_3(hs100, num_sites, runs)


# ## Experiment 4
# 1. Train to minimize $\log(C+\epsilon)$
# 3. Filter is now if EI(x) is sufficiently large,
#     $EI(x) = (f^*-\mu_f(x))\Phi\left(\frac{f^*-\mu_f(x)}{\sigma_f(x)}\right)+\sigma_f(x)\phi\left(\frac{f^*-\mu_f(x)}{\sigma_f(x)}\right)$
#    where $\mu_f,\sigma_f, f^*$ are the predicted value, variance, and maximum of the sm, and $\Phi,\phi$ are normal CDF and PDF.
#    
#    Note: Need to choose test sites wisely and tune filter better

# In[99]:


# weigh with variance
def experiment_4(local_eval, num_sites_training, num_sites_testing,verbose=False):
    # get training data
    sample_data = sample_sites(local_eval.problem(), num_sites_training)
    exp_data = evaluate_sites(local_eval, sample_data, verbose=False)
    variables = list(local_eval.problem()["variables"].keys())
    xt = exp_data[variables].to_numpy()
    yt = (-1) * (1 / (exp_data['__conviol__'].to_numpy()+1e-6))
    
    # create model
    sm = KRG(theta0=[1e-2], print_global=False)
    sm.set_training_values(xt, yt)
    sm.train()

    # compute EI(x)
    x = sample_sites(local_eval.problem(), num_sites_testing**2).to_numpy()
    mu = sm.predict_values(x)
    sigma = np.sqrt(sm.predict_variances(x))
    y_min = np.min(mu)
    t = (y_min-mu)/sigma
    EI_x = (y_min-mu)*stats.norm.cdf(t) + sigma*stats.norm.pdf(t)
    feasible_points = pd.DataFrame(data=np.concatenate((x, EI_x), axis=1), columns=variables + ['EI'])
    
    feasible_points = feasible_points.sort_values(by='EI', ascending=False).reset_index(drop=True)[:num_sites_testing] #.drop('EI',axis=1)
    
    # evaluate model on testing data
    feasible_points = evaluate_sites(local_eval, feasible_points, verbose=verbose)
    return (feasible_points) 


# In[100]:


hs100 = HS100()
problem = hs100.problem()
nind = len(problem['variables'])
training_sites = 50
testing_sites = 50
experiment_4(hs100, training_sites, testing_sites)


# # Experiment 5
# We select sites sparsely
# 1. Sort the sites by their score through the model
# 2. Select the next possible site and delete the others that are too close
# 3. Continue until we are left with the appropriate number of sites

# In[117]:


# filter points that are too close
def experiment_5(local_eval, num_sites_training, num_sites_testing,verbose=True):
    # get training data
    sample_data = sample_sites(local_eval.problem(), num_sites_training)
    exp_data = evaluate_sites(local_eval, sample_data, verbose=False)
    variables = list(local_eval.problem()["variables"].keys())
    xt = exp_data[variables].to_numpy()
    yt = (-1)*np.log(exp_data['__conviol__'].to_numpy() + 1e-6)
    
    # create model
    sm = KRG(theta0=[1e-2], print_global=False)
    sm.set_training_values(xt, yt)
    sm.train()

    # filter points
    x = sample_sites(local_eval.problem(), num_sites_testing**2).to_numpy()
    y = sm.predict_values(x)
    exp_data = pd.DataFrame(data=np.concatenate((x,y), axis=1), columns=variables + ['conviol'])
    exp_data = exp_data.sort_values(by='conviol').reset_index().drop('conviol',axis=1).drop('index', axis=1).to_numpy()
    eps = 1 # not sure how to decide this
    feasible_points = np.zeros((1,len(variables)))
        
    while len(feasible_points) < num_sites_testing and len(exp_data) > 0:
        feasible_points = np.append(feasible_points, np.array([exp_data[0]]), axis=0)
        exp_data = exp_data[1:]
        dists = np.linalg.norm(exp_data - feasible_points[-1], axis=1)
        exp_data = exp_data[dists > eps]

    feasible_points = feasible_points[1:]
    # evaluate model on testing data
    feasible_points = pd.DataFrame(data=feasible_points, columns=variables)
    feasible_points = evaluate_sites(local_eval, feasible_points)
    return feasible_points


# In[102]:


hs100 = HS100()
problem = hs100.problem()
nind = len(problem['variables'])
training_sites = 50
testing_sites = 50
experiment_5(hs100, training_sites, testing_sites)


# # Experiment 6
# Combine the other parts together

# In[103]:


# multi-run training
def experiment_6(local_eval, sites_per_run, runs, verbose=False):
    problem = hs100.problem()
    eps = 1e-6
    nind = len(problem['variables'])

    # create model
    sm = KRG(theta0=[1e-2], print_global=False)
    
    for i in range(runs):
        if i == 0:
            # get first run's data
            sample_data = sample_sites(local_eval.problem(), sites_per_run)
            exp_data = evaluate_sites(local_eval, sample_data, verbose=False)
            training_data = exp_data
        else:
            training_data = pd.concat([training_data, exp_data], axis=0).drop_duplicates()
        variables = list(local_eval.problem()["variables"].keys())
        xt = training_data[variables].to_numpy()
        yt = np.log(training_data['__conviol__'].to_numpy() + eps)
        
        # train model
        sm.set_training_values(xt, yt)
        sm.train()

        # compute EI
        x = sample_sites(local_eval.problem(), sites_per_run**2).to_numpy()
        mu = sm.predict_values(x)
        sigma = np.sqrt(sm.predict_variances(x))
        y_max = np.max(mu)
        t = (y_max-mu)/sigma
        EI_x = (y_max-mu)*stats.norm.cdf(t) + sigma*stats.norm.pdf(t)
        feasible_points = pd.DataFrame(data=np.concatenate((x, EI_x), axis=1), columns=variables + ['EI'])
        feasible_points = feasible_points.sort_values(by='EI', ascending=False).reset_index().drop(['EI', 'index'], axis=1)
        exp_data = feasible_points.to_numpy()

        # ensure sparse
        feasible_points = np.zeros((1,len(variables)))
        while len(feasible_points) < sites_per_run and len(exp_data) > 0:
            feasible_points = np.append(feasible_points, np.array([exp_data[0]]), axis=0)
            exp_data = exp_data[1:]
            dists = np.linalg.norm(exp_data - feasible_points[-1], axis=1)
            exp_data = exp_data[dists > eps]
    
        feasible_points = feasible_points[1:]
        # evaluate model on testing data
        feasible_points = pd.DataFrame(data=feasible_points, columns=variables)
        exp_data = evaluate_sites(local_eval, feasible_points)
        
    return exp_data


# In[107]:


hs100 = HS100()
problem = hs100.problem()
nind = len(problem['variables'])
sites_per_run = 20
runs = 5
experiment_6(hs100, sites_per_run, runs)


# # Benchmark experiments against each other

# In[119]:


total_sites = [20, 30, 50, 75, 100]
hs100 = HS100()
problem = hs100.problem()
results = pd.DataFrame(columns=['Random Points','Simple Surrogate Model','Multi-Run','EI','Sparseness', 'All combined'])

for ts in total_sites:
    cur_results = [0]*6
    runs = 5
    sites_per_run = ts//runs
    cur_results[0] = (experiment_1(hs100, ts)['__State__'] == 'Feasible').sum()
    cur_results[1] = (experiment_2(hs100, ts//2, ts//2, verbose=False)['__State__'] == 'Feasible').sum()
    cur_results[2] = (experiment_3(hs100, sites_per_run, runs, verbose=False)['__State__'] == 'Feasible').sum()
    cur_results[3] = (experiment_4(hs100, ts//2, ts//2, verbose=False)['__State__'] == 'Feasible').sum()
    cur_results[4] = (experiment_5(hs100, ts//2, ts//2, verbose=False)['__State__'] == 'Feasible').sum()
    cur_results[5] = (experiment_6(hs100, sites_per_run, runs, verbose=False)['__State__'] == 'Feasible').sum()
    results = pd.concat([results, pd.DataFrame(data=[cur_results], columns=results.columns)], ignore_index=True)
    print(ts)
    print(cur_results)
results = pd.DataFrame(results.to_numpy(),total_sites, results.columns)


# In[121]:


results.plot.bar()


# In[ ]:




