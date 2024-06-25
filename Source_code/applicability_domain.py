#!/usr/bin/env python

from logging import info, warning
from math import ceil
import os
from os.path import exists
import pickle as pkl
from time import time
import cProfile
import re

import numpy as np
import scipy.spatial as spatial
from scipy.stats import mode
from glob import glob
import pandas as pd
from random_forest import RandomForest
import statistics
from ml_tools import MyEncoder
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.neighbors import KernelDensity
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import pairwise_distances as p_dists
from sklearn.tree import DecisionTreeClassifier as clf

from config import Options

#np.warnings.filterwarnings('ignore')

# ID values for system state
NOT_RUN = 0
RUNNING = 1
FINISHED = 2

__version_info__ = (0, 0, 1, 0)
__version__ = "%i.%i.%i.%i" % __version_info__

class cd:
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


class ApplicabilityDomain(object):
    """
    Single property that holds the information required for determining
    the applicability domain
    """
    def __init__(self, options, name):
        self.state = {'data_prep': NOT_RUN, 'trained': NOT_RUN,
                      'threshold_test': NOT_RUN}
        self.opt = options
        self.job_name = name
        self.ranges = None
        self.ti = None
        self.kde = None
        self.max_density = None
        self.cross_data = None
        self.ids = self.data_prep()
        self.state['data_prep'] = FINISHED

    def dump_state(self):
        """Write the AD.state file for the AD training"""
        info(f"Writing state file, {self.job_name}.ads")
        with open(f"{self.job_name}.ads", 'wb') as state_file:
            pkl.dump(self, state_file)


    def data_prep(self, reduction_technique='FactorAnalysis'):
        """Prepare the data for use in applicability domain calculations"""
        cross, self.target = self.opt.get('cross_column'), self.opt.get('target')
        if exists(f'{self.job_name}.rfs'):
            info("Found RandomForest, using its data")
            with open(f'{self.job_name}.rfs', 'rb') as load_rfs:
                u = pkl._Unpickler(load_rfs)
                u.encoding = 'latin1'
                rfs_data = u.load()
            if exists('Prepped_Data.pkl'):
                rfs_data.data = pd.read_pickle('Prepped_Data.pkl')
                with open(f"{self.job_name}.rfs", 'wb') as my_state:
                    pkl.dump(rfs_data, my_state)
            tr_data = self.data_slim(rfs_data, cross=cross, training=True)
            all_data = self.data_slim(rfs_data, cross=cross)

        # Choose dimensionality reduction technique
        if reduction_technique == 'FactorAnalysis':
            reducer = self.factor_analysis(tr_data)
        if reduction_technique == 'PCA':
            reducer = PCA(n_components=0.98)

        reduced_data = reducer.fit_transform(tr_data)
        n_comps = len(reduced_data[0])
        cols = [f'Component_{x + 1}' for x in range(n_comps)]
        self.reduced_data = pd.DataFrame(reduced_data,
                                         index=tr_data.index,
                                         columns=cols)
        self.reducer = reducer

        all_reduced_data = self.reducer.transform(all_data)

        self.all_reduced_data = pd.DataFrame(all_reduced_data,
                                             index=all_data.index,
                                             columns=cols)

        base = rfs_data.data[[cross, 'Target']]
        base['TrTe'] = rfs_data.data_break[0]

        return base

    
    def data_slim(self, data, cross=None, training=False):
        if training:
            base = data.data_break[[0, cross]]
            self.cross_data = base[cross]
            train = base[base[0] == 'Tr']
            data = data.data.loc[train.index]
        else:
            data = data.data
        return data.drop([cross, 'Target', 'Predict', 'Path'], axis=1)


    def factor_analysis(self, tr_data, variability_threshold=0.98):
       # Calculate the number of factors based on the desired explained variance
        total_variance = tr_data.var().sum()
        num_factors = 0
        explained_variance = total_variance
        while explained_variance / total_variance > 1 - variability_threshold:
            num_factors += 1
            fa = FactorAnalysis(n_components=num_factors)
            tr_data_reduced = fa.fit_transform(tr_data)
            explained_variance = fa.noise_variance_.sum()
        return FactorAnalysis(n_components=num_factors)


    def fit(self):
        """Function to fit an applicability domain"""
        # Retrieve domain and distance from self.opt dictionary
        domain = self.opt.get('domain')
        dtype = self.opt.get('distance')
        
        # Filter training data indices
        tr_idx = self.ids[self.ids['TrTe'] == 'Tr'].index
        
        # Extract the training data
        data = self.reduced_data.loc[tr_idx]
        
        # Calculate pairwise distances
        pdists = spatial.distance.pdist(data.values, metric=dtype)
        
        # Convert the distance matrix to a DataFrame
        pdists = pd.DataFrame(
            spatial.distance.squareform(pdists),
            index=data.index,
            columns=data.index
        )

        # Set the pdists diagnol to NaN so they are ignored in calculations
        np.fill_diagonal(pdists.values, np.nan)

        # Check the domain for valid values
        if domain in ['all', 'dknn', 'rdn']:
            try:
                k = self.opt.getint('nearest_neighbour')
            except ValueError:
                # Consistent with d-kNN paper 10.1186/1758-2946-5-27
                k = int(ceil(len(pdists) ** (1 / 3.0)))
        
            # Fit d-kNN
            self.t_i = dknn_fit(pdists, k)
        
            if domain in ['all', 'rdn']:
                # Implemented based on RDN paper 10.1186/s13321-016-0182-y
                trees = self.opt.getint('rdn_trees')
                out_o = self.opt.gettuple('output_order')
        
                # Fit RDN
                self.t_i = rdn_fit(data, self.ids, self.t_i, trees, out_o)
        
            # Calculate all densities
            all_data = self.all_reduced_data

            dists = spatial.distance.cdist(all_data.values, data.values, metric=dtype)

            dist_df = pd.DataFrame(dists, index=all_data.index, columns=data.index)

            dist_df = dist_df.apply(lambda col: col.where((col <= self.t_i.values) & (col != 0)), axis=1)

            self.data_densities = dist_df.count(axis=1)
        
        # Check for 'kde' domain
        if domain in ['all', 'kde']:
            bandwidth = self.opt.getfloat('kde_bandwidth')
        
            # Define bandwidth parameters
            params = {'bandwidth': np.logspace(-1, 1, 20)}
        
            # Perform GridSearchCV for KDE
            grid = GridSearchCV(KernelDensity(metric=dtype), params, cv=3, n_jobs=-1)
            grid.fit(data)
        
            # Get the best KDE estimator
            self.kde = grid.best_estimator_
            info(f"Best KDE bandwidth was found to be {self.kde.bandwidth}")
        
            # Calculate densities using KDE
            densities = pd.Series(self.kde.score_samples(data), index=data.index)
            self.max_density = max(densities)
        
            # Rescale all densities
            all_data = self.all_reduced_data
            densities = pd.Series(self.kde.score_samples(all_data.values), index=all_data.index)
            densities = (densities * 50).round() / 50
#            densities = densities.apply(lambda x: 10 ** (x - self.max_density))
            self.data_densities = densities

        self.data_densities.name = 'Density'
        df = pd.concat([self.cross_data, self.data_densities], axis=1)
        df.to_csv('Density_Information.csv')

    def test_results(self):
        idx = int(self.opt.get('level_idx'))
        vals = pd.read_csv('Performance_Data.csv', index_col=0)
        if idx == '':
            if self.opt.getbool('pareto_points'):
                idx = int(input('What index should be used? '))
                best = vals.loc[idx]['Parameters']
            else:
                best = list(vals.sort_values('Fitness').tail(1)['Parameters'])[0]
        else:
            best = vals.loc[idx]['Parameters']
        levels = [int(x) for x in best.split(',')]
        full_predictions = pd.read_pickle('Full_Predictions.pkl')
        results = []
        for i, level in enumerate(levels):
            if level == 0:
                continue
            results.append(full_predictions[i + 1][level])
        results = pd.concat(results, axis=1)
        full_results = results.apply(find_mode, axis=1)
        order = sorted(full_results.unique(), reverse=True)
        full_results.name = 'Predict'
        all_data = pd.concat([self.ids, full_results, self.data_densities], axis=1)
        thresholds = sorted(all_data['Density'].unique())
        total = all_data.shape[0]
        tr_total = all_data[all_data['TrTe'] == 'Tr'].shape[0]
        te_total = all_data[all_data['TrTe'] == 'Te'].shape[0]
        all_results = []
        for threshold in thresholds:
            in_ad = all_data[all_data['Density'] >= threshold]
            tot_results = calc_ad_results(in_ad, total, order)
            tot_results.insert(0, 'Threshold', threshold)
            tr_results = calc_ad_results(in_ad[in_ad['TrTe'] == 'Tr'], tr_total, order)
            tr_results.columns = ['Train ' + col for col in tr_results.columns]
            te_results = calc_ad_results(in_ad[in_ad['TrTe'] == 'Te'], te_total, order)
            te_results.columns = ['Test ' + col for col in te_results.columns]
            all_results.append(pd.concat([tot_results, tr_results, te_results], axis=1))
        threshold_results = pd.concat(all_results)
        name = self.job_name
        threshold_results.to_csv(f'{name}_threshold.csv')


def calc_ad_results(data, total, order):
    coverage = 100 * data.shape[0] / total
    obs = data['Target']
    exp = data['Predict']
    balanced_accuracy = balanced_accuracy_score(obs, exp)
    if len(order) < 3:
        recall = recall_score(obs, exp, pos_label=0)
        precision = precision_score(obs, exp, pos_label=0)
    else:
        recall = recall_score(obs, exp, labels=order[1:], average='macro')
        precision = precision_score(obs, exp, labels=order[1:], average='macro')
    f1 = 2 * (precision * recall) / (precision + recall)
    cm = confusion_matrix(obs, exp, labels=order).ravel()
    df = pd.DataFrame({'Coverage': [coverage],
                       'Number': [data.shape[0]],
                       'Balanced Accuracy': [balanced_accuracy],
                       'Recall': [recall],
                       'Precision': [precision],
                       'F1': [f1],
                       'Confusion Matrix': [cm]})
    return df

def dknn_fit(pdists, k):
    """
    Fit an applicability domain using the density k-Nearest Neighbors approach.

    Parameters:
    - pdists: DataFrame
        Pairwise distances between data points.
    - k: int
        Number of nearest neighbors to consider.

    Returns:
    - t_i: Series
        Applicability domain threshold for each data point.
    """

    # Calculate the mean distance to k nearest neighbors for each point
    k_dists = pdists.apply(lambda x: x.nsmallest(k).mean())

    # Sort the mean distances and calculate the quartiles
    q1, q3 = np.percentile(k_dists, [25, 75])

    # Calculate the reference value
    refval = q3 + 1.5 * (q3 - q1)

    # Calculate t_i for each data point
    t_i = pdists.apply(lambda x: x[x <= refval].mean())

    # Find the minimum positive t_i value
    min_t = t_i[t_i > 0].min()

    # Replace zero t_i values with half of the minimum positive t_i
    t_i[np.isnan(t_i)] = min_t

    return t_i


def rdn_fit(feats, ids, t_i, n_trees, out_o):
    """Calculate the weighted threshold values based on STD and agreement"""
    if out_o == ():
        out_o = sorted(set(ids.astype('str')))
    endp = sorted(set(ids.astype('str')))
    out_o = sorted(endp, key=out_o.index)
    enc = MyEncoder()
    enc.fit(out_o, ordered=True, reverse=True)
    ids = enc.transform(ids.astype('str'))
    trees = pd.DataFrame(index=feats.index, columns=range(n_trees))
    for i in range(n_trees):
        xtr, _x, ytr, _y = train_test_split(feats, ids, test_size=0.25)
        dt = clf(max_depth=10, max_features="sqrt")
        dt.fit(xtr, ytr)
        trees[i] = dt.predict(feats)
    std = trees.apply(lambda x: 1 - np.std(x, ddof=1), axis=1)
    for col in trees.columns:
        trees[col] = trees[col] == ids
    agree = trees.sum(axis=1) / float(n_trees)
    return t_i * std * agree


def load_applicability_domain(ads_name, main_opts, job_name):
    """
    Load an existing Applicability Domain or create a new one.

    Parameters:
    - ads_name (str): Name of the Applicability Domain state file.
    - main_opts (Options): Main options for the program.
    - job_name (str): Name of the job.

    Returns:
    - a_domain (ApplicabilityDomain): Loaded or newly created Applicability Domain.
    """
    if exists(ads_name):
        info(f"Found previous state: loading ApplicabilityDomain from {ads_name}")
        try:
            with open(ads_name, 'rb') as load_ads:
                a_domain = pkl.load(load_ads)
        except Exception as e:
            warning(f"Failed to load ApplicabilityDomain: {e}")
            a_domain = ApplicabilityDomain(main_opts, job_name)
            a_domain.dump_state()
    else:
        info("Starting a new ApplicabilityDomain training...")
        a_domain = ApplicabilityDomain(main_opts, job_name)
        a_domain.dump_state()
    return a_domain


def find_mode(x):
    """Find the mode"""
    try:
        return statistics.mode(x)
    except statistics.StatisticsError:
        return stats.mode(x)[0][0]


def ad_work(main_opts, job_name):
    info(f"Calculating AD for {job_name}")

    if exists(f'{job_name}_threshold.csv'):
        info("Already calculated AD")
        return

    ads_name = f"{job_name}.ads"
    a_domain = load_applicability_domain(ads_name, main_opts, job_name)

    if a_domain.state['trained'] != FINISHED:
        info("ApplicabilityDomain not fit, fitting now...")
        a_domain.fit()
        a_domain.dump_state()
    if a_domain.state['threshold_test'] != FINISHED:
        info("Testing results of ApplicabilityDomain")
        a_domain.test_results()
        a_domain.dump_state()


def main():
    """
    Main program for Applicability Domain calculation.
    """
    main_opts = Options(code='AD')
    info(f"Starting Applicability Domain {__version__}")
    if glob('*.rfs') != []:
        job_name = main_opts.get('job_name')
        ad_work(main_opts, job_name)
    else:
        for directory in glob('*/*/'):
            with cd(directory):
                job_name = '_'.join(os.getcwd().split('/')[-3:])
                ad_work(main_opts, job_name)


if __name__ in '__main__':
    main()
