#!/usr/bin/env python
"""
Title:        scLearn
Author:       Sean Collins
Date:         2019 06 05
Description:  Suite of machine learning algorithms
Usage:        %Call modules from other programs
"""

import array
from collections import Counter, defaultdict
import itertools
from logging import info
from operator import itemgetter
import pickle as pkl
import random
from time import time

import numpy as np
import pandas as pd
import sklearn as skl
import sklearn.metrics as sklm
from sklearn.metrics import matthews_corrcoef
from sklearn.utils.multiclass import unique_labels
#skl.warnings.filterwarnings('ignore')

__version_info__ = (0, 7, 1, 0)
__version__ = "%i.%i.%i.%i" % __version_info__


class DecisionTree(object):
    """
    Class to hold information on a decision tree
    """
    def __init__(self, data, job_name='Training', cross='CERAPP_ID',
                 max_feats=None, scale=True, out_o=(), enc=None):
        """Initialize a Decision Tree instance"""
        self.data = data.reset_index(drop=True)
        self.job_name = job_name
        self.cross = cross
        self.enc = enc
        feats = list(data.columns[4:])
        self.class_idx = self.build_class_idx(feats)
        feats = self.class_idx.columns.levels[0]
        if max_feats is None:
            self.max_feats = int((len(feats) + 0.5) ** 0.5)
        else:
            self.max_feats = int(max_feats * len(feats) + 0.5)
        targets = dict(Counter(data['Target']))
        self.scales = {x: 1 for x in targets.keys()}
        if scale:
            targets = dict(Counter(data['Target']))
            top = float(max(targets.values()))
            self.scales = {x: top / float(y) for (x, y) in targets.items()}
        self.rules = {}
        self.labels = {}
        self.score = None
        self.train_results = None
        self.out_o = tuple(unique_labels(self.data['Target']))
        if out_o != ():
            self.out_o = tuple(sorted(self.out_o, key=list(out_o).index))
        self.tscore = 0

    def clean_up(self):
        del self.data
        del self.class_idx
        del self.score
        del self.max_feats

    def dump_state(self):
        """Write the Decision Tree to a state file"""
        my_state = open("DT_{}.state".format(self.job_name), 'wb')
        pkl.dump(self, my_state)
        my_state.close()

    def build_class_idx(self, feats):
        """Bin and index all data"""
        try:
            return pd.read_pickle('Class_idx.pkl')
        except IOError:
            pass
        temp, index = self.data.apply(bin_classes), []
        for feat, clas in zip(feats, temp[4:]):
            for val in clas[1:]:
                index.append((feat, val))
        cols = pd.MultiIndex.from_tuples(index)
        class_idx = pd.DataFrame(0, index=self.data.index, columns=cols)
        for feat in feats:
            col = self.data[feat]
            try:
                base = class_idx[feat].copy()
            except KeyError:
                continue
            classes = base.columns
            for val in classes:
                right = col[col >= val].index
                conds = [col.index.isin(right)]
                base[val] = np.select(conds, [1])
            class_idx[feat] = base
        class_idx = class_idx.astype('int8')
        return class_idx

    def fit(self, max_depth=5, metric='mcc'):
        """Fit the data, all data is used for fitting"""
        self.depth, start = 0, time()
        if metric == 'Gini':
            res = dict(Counter(self.data['Target']))
            total = float(sum(res[x] * self.scales[x] for x in res.keys()))
            self.score = 1 - sum([(res[x] * self.scales[x] /
                                   total) ** 2 for x in res.keys()])
            self.scale_total = total
        else:
            con_mat = sklm.confusion_matrix(self.data['Target'],
                                            self.data['Predict'],
                                            labels=self.out_o)
            self.score = 1 - self.calc_score(con_mat, metric)
        stime = start
        output = open('{}.out'.format(self.job_name), 'w')
        output.write("Depth  Nodes    Score    DTime (s)  TTime (s)\n")
        output.write("=====  =====  =========  =========  =========\n")
        info("Depth  Nodes    Score    DTime (s)  TTime (s)")
        info("=====  =====  =========  =========  =========")
        spc = '%-6s%6s%11.6f%11.1f%10.1f'
        while self.score != 0:
            dtime = time() - stime
            nodes = set(self.data['Path'])
            dnodes = len([x for x in nodes if len(x) == self.depth])
            output.write(spc % (self.depth, dnodes, self.score, dtime,
                                time() - start) + '\n')
            output.flush()
            info(spc % (self.depth, dnodes, self.score, dtime, time() - start))
            stime = time()
            if self.depth == max_depth:
                break
            attempts = 1
            self.get_splits(metric)
            if self.depth == 0 and set(self.data['Path']) == set(['']):
                attempts += 1
                while set(self.data['Path']) == set(['']):
                    self.get_splits(metric)
                    if attempts == 30:
                        return
            self.depth += 1
            ends = set(self.data['Path'].apply(lambda x: x[-1]))
            if ends.difference(set(['L', 'R'])) == set():
                break
        results = self.data[[self.cross, 'Target', 'Predict']]
        results.to_csv('Prediction.csv')
        actual = self.enc.inverse_transform(self.out_o)
        tree_res = pd.DataFrame(0, index=actual, columns=actual)
        for class1 in self.out_o:
            base = results[results['Target'] == class1]
            target = self.enc.inverse_transform([class1])[0]
            for k, v in dict(Counter(base['Predict'])).items():
                predict = self.enc.inverse_transform([k])[0]
                tree_res[predict][target] = v
        tree_res.to_csv(self.job_name + '_cm.csv')
        output.flush()

    def get_splits(self, metric):
        """Split the decision tree for that level"""
        feats = list(self.class_idx.columns.levels[0])
        score = self.score
        for path in sorted(set(self.data['Path'])):
            if any(x in path for x in ['R', 'L']):
                continue
            cols = sorted(random.sample(feats, self.max_feats))

            other = self.data[self.data['Path'] != path]
            o_tar = other['Target']
            o_pred = other['Predict']
            n_idxs = self.data[self.data['Path'] == path].index
            node = self.class_idx[self.class_idx.index.isin(n_idxs)][cols]
            node_tar = self.data[self.data.index.isin(n_idxs)]['Target']
            if len(node) <= 0.01 * len(self.data) or len(dict(Counter(node_tar))) == 1:
                terminal = path[::-1].title()[::-1]
                end = node.index
                self.labels[terminal] = self.labels[path]
                self.data.loc[self.data.index.isin(end), 'Path'] = terminal
                continue
            if metric == 'Gini':
                gin_ave = score * self.scale_total
                splits = node.apply(self.gini_index, score=gin_ave)
            else:
                self.tscore = 0
                out_o = self.out_o
                con_mat = [[0] * len(out_o)] * len(out_o)
                try:
                    con_mat = sklm.confusion_matrix(o_tar, o_pred, labels=out_o)
                    num, denom = 0, 0
                    for i, (j, k) in enumerate(zip(out_o, con_mat)):
                        num += self.scales[j] * k[i]
                        denom += float(self.scales[j] * sum(k))
                    o_scores = (num, denom)
                except ValueError:
                    con_mat = np.array([[0] * len(out_o)] * len(out_o))
                    o_scores = (0, 0)
                splits = node.apply(self.test_split, metric=metric,
                                    node_tar=node_tar, o_scores=o_scores,
                                    bcon_mat=con_mat)
            splits = pd.DataFrame(splits.str.split(' ').tolist(),
                                  columns=['sign', 'score'], index=splits.index)
            splits['score'] = pd.to_numeric(splits['score'])
            best = splits.sort_values('score', ascending=True).head(1)
            best = best.reset_index().squeeze()
            if best['score'] / score > 0.995:
                terminal = path[::-1].title()[::-1]
                end = node.index
                self.data.loc[self.data.index.isin(end), 'Path'] = terminal
                self.labels[terminal] = self.labels[path]
                continue
            name = [best['level_0'], best['level_1']]
            self.rules[path] = '{} {} {}'.format(name[0], best['sign'],
                                                 name[1])
            right = set(node[node[name[0]][name[1]] == 1].index)
            left = list(set(node.index).difference(right))
            right = list(right)
            if best['sign'] == '<':
                right, left = left, right
            self.labels[path + 'r'] = self.set_label(node_tar, right)
            self.labels[path + 'l'] = self.set_label(node_tar, left)
            self.data.iloc[right, self.data.columns.get_loc('Path')] += 'r'
            self.data.iloc[left, self.data.columns.get_loc('Path')] += 'l'
            self.data['Predict'] = self.data['Path'].apply(lambda x: self.labels[x])
            self.score = best['score']
            score = best['score']


    def set_label(self, node, group):
        """Assign label based off scaled value and output order
        Label assignment favors weighted amount in group, and those
        being equal, the position in output order, with later output
        order being favored"""
        res = dict(Counter(node[node.index.isin(group)]))
        res = {x: res[x] * self.scales[x] for x in res.keys()}
        lab = [k for k, v in res.items() if v == max(res.values())]
        mx = [self.out_o.index(k) for k in lab]
        return lab[mx.index(min(mx))]

    def index_data(self, col, classes, feats):
        """Index the data based on available splits"""
        feat = col.name
        if feat in ['Target', 'Predict', 'Path', self.cross]:
            return
        try:
            idx = feats.index(feat)
        except ValueError:
            return
        classes = classes[idx]
        for val in classes[1:]:
            right = set(col[col >= val].index)
            left = set(col.index).difference(right)
            conds = [col.index.isin(right), col.index.isin(left)]
            self.class_idx[feat][val] = np.select(conds, [1, 0])

    def gini_index(self, col, score):
        """Calculate the improvement of GINI purity based on a split"""
        node = self.data[self.data.index.isin(col.index)]['Target']
        res, scale_total = dict(Counter(node)), self.scale_total
        NT = float(sum(res[x] * self.scales[x] for x in res.keys()))
        imp = 1 - sum([(res[x] * self.scales[x] /
                        NT) ** 2 for x in res.keys()])
        right, left = col[col == 1].index, col[col == 0].index
        rres = dict(Counter(node[col.index.isin(right)]))
        lres = dict(Counter(node[col.index.isin(left)]))
        ntr = float(sum(rres[x] * self.scales[x] for x in rres.keys()))
        rscore = 1 - sum([(rres[x] * self.scales[x] /
                           ntr) ** 2 for x in rres.keys()])
        ntl = float(sum(lres[x] * self.scales[x] for x in lres.keys()))
        lscore = 1 - sum([(lres[x] * self.scales[x] /
                           ntl) ** 2 for x in lres.keys()])
        child_ave = ntr * rscore + ntl * lscore
        score = (score - imp * NT + child_ave) / scale_total
        return '>= {}'.format(score)

    def test_split(self, col, metric, node_tar, o_scores, bcon_mat):
        """Calculate metric based on data feature turning on or off"""
        start = time()
        if sum(col) in [0, len(col)]:
            return '>= {}'.format(self.score)
        conds, scales = [col.isin([0]), col.isin([1])], self.scales
        left, right = list(node_tar[conds[0]]), list(node_tar[conds[1]])
        lres = dict(zip(*np.unique(left, return_counts=True)))
        rres = dict(zip(*np.unique(right, return_counts=True)))
        lres = {k: v * scales[k] for k, v in lres.items()}
        rres = {k: v * scales[k] for k, v in rres.items()}
        lmax, rmax = max(lres.values()), max(rres.values())
        llab = [k for k, v in lres.items() if v == lmax][0]
        rlab = [k for k, v in rres.items() if v == rmax][0]
        choice = [llab, rlab]
        preds = np.select(conds, choice)
        con_mat = sklm.confusion_matrix(list(node_tar), preds, labels=self.out_o)
        if metric != 'BalAcc':
            con_mat += bcon_mat
        score = self.calc_score(con_mat, metric, o_score=o_scores)
        return '>= ' + str(1 - score)

    def predict(self, preds, results=False, depth=False):
        """Predict the outcome of decision tree"""
        rls = self.rules
        if not depth:
            depth = 999
        preds.reset_index(drop=True, inplace=True)
        preds['Predict'] = pd.Series([''] * len(preds), index=preds.index)
        for path in sorted(rls, key=len):
            if len(path) + 1 == depth:
                 break
            rule = rls[path]
            feat, cond, val = rule.split()
            node = preds[preds['Path'] == path]
            right = node[node[feat] >= float(val)].index
            left = node[node[feat] < float(val)].index
            if cond == '<':
                right, left = left, right
            preds.iloc[right, preds.columns.get_loc('Path')] += 'r'
            preds.iloc[left, preds.columns.get_loc('Path')] += 'l'
        preds['Predict'] = preds['Path'].apply(lambda x: self.labels[x])
        try:
            return preds[[self.cross, 'Predict']]
        except KeyError:
            return preds['Predict']

    def calc_score(self, con_mat, metric, o_score=(0, 0)):
        """Calculate a metric to use to fit the DecisionTree"""
        scales, labels, num, denom = self.scales, self.out_o, 0, 0
        if metric == 'BalAcc':
            for i, (j, k) in enumerate(zip(labels, con_mat)):
                num += scales[j] * k[i]
                denom += float(scales[j] * sum(k))
            return (num + o_score[0]) / (denom + o_score[1])
        true, pred = [], []
#        if self.enc.reverse and len(labels) == 2:
#            labels = labels[::-1]
        for i, k in enumerate(con_mat):
            target = labels[i]
            for j, l in enumerate(k):
                predict = labels[j]
                true += [target] * l
                pred += [predict] * l
        if metric == 'mcc':
            return matthews_corrcoef(true, pred)
        if metric[:2].lower() == 'f_':
            fbeta = float(metric.split('f_')[-1])
            return sklm.fbeta_score(true, pred, fbeta, average='macro')


class MyEncoder(object):
    """
    Class built to encode data for use in machine learning models. Based on the
    sclearn class, but allows proper sorting when encoding data. This is useful
    for the scipy stats module where modes are calculated using the smallest
    value.
    """
    def __init__(self):
        """Initialize the class, classes and encoding are stored"""
        self.classes = np.array([])
        self.encoding = {}
        self.ordered = False
        self.reverse = False

    def fit(self, y, ordered=False, reverse=False):
        """Fit the data for encoding
        y: Data to fit the encoding to
        ordered: Keep the data in the same ordering as what was given
        reverse: Reverse the order of data, useful for scipy.stats.mode
        """
        base = list(set(y))
        if ordered:
            base = sorted(base, key=y.index, reverse=reverse)
        self.classes = np.array(base)
        self.encoding = {x: i for (i, x) in enumerate(self.classes)}
        self.ordered = ordered
        self.reverse = reverse

    def transform(self, y):
        """Transofrm an array like of data to encoded values"""
        for x in set(y):
            if x not in self.encoding:
                self.encoding[x] = x
        return np.array([self.encoding[x] for x in y])

    def inverse_transform(self, y):
        """Inverse transform, from encoded values to regular values"""
        return np.array([self.classes[x] for x in y])
        


def bin_classes(col):
    """Bin data if needed"""
    classes = sorted(set(col))
    val = 20.0
    if len(classes) < val:
        return classes
    split = int(len(classes) / val)
    return classes[0::split]


def tanimoto(list1, list2):
    """Calculate the tanimoto of two lists"""
    cross = np.dot(list1, list2)
    return cross / (np.dot(list1, list1) + np.dot(list2, list2) - cross)


def chunks(l, n):
    """Cut work into chunk size bit"""
    for i in range(0, len(l), n):
        yield l[i: i + n]

#To work on
def unit_read_convert(string, unit):
    """Read in a string containing starting unit and convert to preferred unit"""
    if any(x in unit for x in ['atm', 'Hg', 'torr', 'pa', 'bar']):
        print("Pressure conversion")
