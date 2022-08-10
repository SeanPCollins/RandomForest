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
import pandas as pd
from random_forest import RandomForest
from ml_tools import MyEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import pairwise_distances as p_dists
from sklearn.tree import DecisionTreeClassifier as clf

from config import Options

np.warnings.filterwarnings('ignore')

# ID values for system state
NOT_RUN = 0
RUNNING = 1
FINISHED = 2

__version_info__ = (0, 0, 1, 0)
__version__ = "%i.%i.%i.%i" % __version_info__

class ApplicabilityDomain(object):
    """
    Single property that holds the information required for determining
    the applicability domain
    """
    def __init__(self, options, name):
        self.state = {'data_prep': NOT_RUN, 'trained': NOT_RUN}
        self.opt = options
        self.job_name = name
        self.ranges = None
        self.targets = None
        self.ti = None
        self.kde = None
        self.max_density = None
        self.cross_data = None
        self.ids, self.data = self.data_prep()
        self.state['data_prep'] = FINISHED

    def dump_state(self):
        """Write the AD.state file for the AD training"""
        info("Writing state file, {}.ads".format(self.job_name))
        my_state = open("{}.ads".format(self.job_name), 'wb')
        pkl.dump(self, my_state)
        my_state.close()

    def data_prep(self):
        """Prepare the data for use in applicability domain calculations"""
        cross, self.target = self.opt.get('cross_column'), self.opt.get('target')
        if exists('{}.rfs'.format(self.job_name)):
            info("Found RandomForest, using its data")
            load_rfs = open(self.job_name + '.rfs', 'rb')
#            try:
#                rfs_data = pkl.load(load_rfs)
#            except UnicodeDecodeError:
            u = pkl._Unpickler(load_rfs)
            u.encoding = 'latin1'
            rfs_data = u.load()
            load_rfs.close()
            if exists('Prepped_Data.pkl'):
                rfs_data.data = pd.read_pickle('Prepped_Data.pkl')
                my_state = open("{}.rfs".format(self.job_name), 'wb')
                pkl.dump(rfs_data, my_state)
                my_state.close()
            tr = rfs_data.data_break
            self.cross_data = tr[cross]
            tr = tr[tr[0] == 'Tr'][[0, cross]]
            tr_data = tr.merge(rfs_data.data, on=cross, how='left')
            tr_data = tr_data.drop([0, cross, 'Target', 'Predict', 'Path'], axis=1)
            self.feats = tr_data.columns
        self.pca, others = PCA(n_components=3).fit(tr_data), rfs_data.data[self.feats] 
        pca_data = self.pca_transform(others)
        base = rfs_data.data[[cross, 'Target']]
        base['TrTe'] = rfs_data.data_break[0]
        return base, pca_data

    def pca_transform(self, feats):
        """Transform feature set based on fitted PCA"""
        idx = feats.index
        feats = self.pca.transform(feats)
        n_comp = self.pca.n_components
        cols = ['PCA_{}'.format(i + 1) for i in range(n_comp)]
        return pd.DataFrame(feats, index=idx, columns=cols)

    def fit(self):
        """Function to fit an applicability domain"""
        domain, dtype = self.opt.get('domain'), self.opt.get('distance')
        tr_idx = self.ids[self.ids['TrTe'] == 'Tr'].index
        data = self.data[self.data.index.isin(tr_idx)]
        pdists = spatial.distance.pdist(data.values, metric=dtype)
        pdists = pd.DataFrame(spatial.distance.squareform(pdists),
                              index=data.index, columns=data.index)
        if domain in ['all', 'dknn', 'rdn']:
            try:
                k = self.opt.getint('nearest_neighbour')
            except ValueError:
                #Consistent with d-kNN paper 10.1186/1758-2946-5-27
                k = int(ceil(len(pdists) ** (1 / 3.0)))
            self.t_i = dknn_fit(pdists, k)
            if domain in ['all', 'rdn']:
                #Implemented based on RDN paper 10.1186/s13321-016-0182-y
                trees = self.opt.getint('rdn_trees')
                out_o = self.opt.gettuple('output_order')
                self.t_i = rdn_fit(data, self.targets, self.t_i, trees, out_o)
            densities = pdists.apply(lambda x: len(x[x <= self.t_i]) - 1)
            densities = densities / float(len(data))
        if domain in ['all', 'kde']:
            bandwidth = self.opt.getfloat('kde_bandwidth')
            params = {'bandwidth': np.logspace(-1, 1, 20)}
            grid = GridSearchCV(KernelDensity(metric=dtype), params, cv=3,
                                n_jobs=-1)
            grid.fit(data)
            self.kde = grid.best_estimator_
            info("Best KDE bandwidth was found to be {}".format(
                 self.kde.bandwidth))
            densities = pd.Series(self.kde.score_samples(data),
                                  index=data.index)
            self.max_density = max(densities)
            densities = densities.apply(lambda x: 10 ** (x - self.max_density))
        densities.name = 'Density'
        densities = pd.DataFrame({x.name: x for x in [self.cross_data, densities]})
        densities.to_csv('Train_densities.csv')


    def test(self):
        """Test a set of chemicals for how much they are in the AD"""
        idx, target = self.opt.get('test_id'), self.opt.get('target')
        dtype, domain = self.opt.get('distance'), self.opt.get('domain')
        if '' in [idx, target]:
            idx, target = None, None
        data = pd.read_csv(self.opt.get('test_csv') + '.csv', index_col=0).astype('str')
        data = data.dropna(how='any', axis=0)
        idxs = data[idx]
        dropping = list(self.opt.gettuple('ignore_cols'))
        feats = data[self.feats]
        test_data = self.pca_transform(feats)
        trte_column = self.opt.get('trte_column')
        train_idx = self.ids[self.ids[trte_column] == 'Tr'].index
        train_data = self.data[self.data.index.isin(train_idx)]
        cdists = spatial.distance.cdist(test_data, train_data, metric=dtype)
        cdist = pd.DataFrame(cdists, index=test_data.index, columns=train_data.index).T
        if domain in ['all', 'dknn', 'rdn']:
            densities = cdist.apply(lambda x: len(x[x <= self.t_i]))
            densities = densities / float(len(train_data))
        if domain in ['all', 'kde']:
            densities = pd.Series(self.kde.score_samples(test_data), index=test_data.index)
            densities = densities.apply(lambda x: 10 ** (x - self.max_density))
        densities.name = 'Density'
        densities = pd.DataFrame({x.name: x for x in [idxs, densities]})
        #TODO add in function to label if something is in training/testing
        densities.to_csv('Full_densities.csv')
        exit()

def dknn_fit(pdists, k):
    """Fit an applicability domain using the density k-Nearest Neighbours
    approach. Taken from doi: 10.1186/1758-2946-5-27"""
    k_dists = pdists.apply(lambda x: x.sort_values().iloc[1:k + 1].mean())
    sort_k = list(k_dists.sort_values())
    quartile = 0.25 * len(sort_k)
    q1, q3 = sort_k[int(quartile - 1)], sort_k[int(3 * quartile - 1)]
    refval = q3 + 1.5 * (q3 - q1)
    t_i = pdists.apply(lambda x: x[x <= refval].mean())
    min_t = min(t_i[t_i > 0])
    t_i.replace(to_replace=0.0, value=min_t / 2.0, inplace=True)
    return t_i


def rdn_fit(feats, targets, t_i, n_trees, out_o):
    """Calculate the weighted threshold values based on STD and agreement"""
    if out_o == ():
        out_o = sorted(set(targets.astype('str')))
    endp = sorted(set(targets.astype('str')))
    out_o = sorted(endp, key=out_o.index)
    enc = MyEncoder()
    enc.fit(out_o, ordered=True, reverse=True)
    targets = enc.transform(targets.astype('str'))
    trees = pd.DataFrame(index=feats.index, columns=range(n_trees))
    for i in range(n_trees):
        xtr, _x, ytr, _y = train_test_split(feats, targets, test_size=0.25)
        dt = clf(max_depth=10, max_features="sqrt")
        dt.fit(xtr, ytr)
        trees[i] = dt.predict(feats)
    std = trees.apply(lambda x: 1 - np.std(x, ddof=1), axis=1)
    for col in trees.columns:
        trees[col] = trees[col] == targets
    agree = trees.sum(axis=1) / float(n_trees)
    return t_i * std * agree


def feature_selection(data, target):
    """Select important features using different algorithms"""
    data = data.dropna(how='all', axis=1)
    y = data[target]
    features = data.drop([target], axis=1)
    var = features.var(axis=0)
    features = features[var[var > 0.005].index]
    classes = features.sum()
    similar = {x: list(classes[classes == x].index) for x in classes}
    possible_sim = {x: y for (x, y) in similar.items() if len(y) > 1}
    for val in possible_sim.values():
        base = abs(features[val].corr())
        base = base.mask(np.equal(*np.indices(base.shape)))
        for col in base:
            if col not in features:
                continue
            sim = list(base[col][base[col] > 0.98].index)
            if sim != []:
                features.drop(sim, axis=1, inplace=True)
    return features


def main():
    """Main program"""
    main_opts = Options(code='AD')
    info("Starting Applicability Domain {}".format(__version__))
    job_name = main_opts.get('job_name')
    ads_name = '{}.ads'.format(job_name)
    if exists(ads_name):
        info("Found previous state: loading ApplicabilityDomain...")
        load_ads = open(ads_name, 'rb')
        a_domain = pkl.load(load_ads)
        load_ads.close()
    else:
        info("Starting a new ApplicabilityDomain training...")
        a_domain = ApplicabilityDomain(main_opts, job_name)
        a_domain.dump_state()
    info("Succesfully Loaded Data")
    if a_domain.state['trained'] != FINISHED:
        info("ApplicabilityDomain not fit, fitting now...")
        a_domain.fit()
#        a_domain.state['trained'] = FINISHED
        a_domain.dump_state()
    exit()
    info("Determining chemicals in the applicability domain")
    a_domain.test()


if __name__ in '__main__':
    main()
