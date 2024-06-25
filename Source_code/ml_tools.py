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
from itertools import combinations
from joblib import Parallel, delayed
from logging import info, warning
from operator import itemgetter
import pickle as pkl
import random
from time import time
import math
import multiprocessing
from functools import partial

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
    def __init__(self, data, job_name='Training', cross=None,
                 max_feats=None, scale=True, out_o=(), enc=None,
                 standards=None):
        """Initialize a Decision Tree instance"""
        # Initialize data and basic attributes
        self.data = data.reset_index(drop=True)
        self.job_name = job_name
        self.cross = cross
        self.enc = enc
        self.standards = None
        if len(standards) > 0:
            self.standards = self.data[self.data[cross].isin(standards)].index

        # Initialize class index
        feats = list(data.columns[4:])  # Consider using a named constant for 4
        self.class_idx = self.data[feats]

        # Determine max_feats
        self.max_feats = self.calculate_max_feats(feats, max_feats)

        # Initialize scales
        self.scales = self.calculate_scales(data, scale)
        self.weights = np.array(data['Target'].map(self.scales))
        self.true = np.array(data['Target'])

        # Initialize out_o directly
        self.out_o = tuple(unique_labels(self.data['Target']))
        if out_o is not None and tuple(out_o) != ():
            self.out_o = tuple(sorted(self.out_o, key=list(out_o).index))
        # Initialize other attributes
        self.rules = {}
        self.labels = {}
        self.score = None
        self.node_score = None
        self.tscore = 0
        values, counts = np.unique(self.true, return_counts=True)
        scales = np.array([self.scales[value] for value in values])
        result = scales * counts
        self.use_split = np.array([result, result])
        self.node_diff = 0
        self.check_df = pd.DataFrame(index=self.data.index)
        self.check_df.loc[:, 'Target'] = self.true
        self.check_df.loc[:, 'Path'] = [''] * self.check_df.shape[0]

    def calculate_max_feats(self, feats, max_feats):
        """Determine the value for max_feats."""
        sqrt_feats = int(len(feats) ** 0.5 + 0.5)
        if max_feats is None:
            max_feats = 0.01
        max_feats = int(max_feats * len(feats) + 0.5)
        log_feats = int(math.log2(len(feats) + 1) + 0.5)
        percentage_feats = int(0.1 * len(feats) + 0.5)
        return max(sqrt_feats, max_feats, log_feats, percentage_feats)


    def calculate_scales(self, data, scale):
        """Calculate scales based on data and scale flag."""
        if scale:
            targets_count = Counter(data['Target'])
            top = float(max(targets_count.values()))
            return {x: top / float(y) for x, y in targets_count.items()}
        else:
            return {x: 1 for x in set(data['Target'])}

    def fit(self, max_depth=5, metric='mcc'):
        """Fit the data, all data is used for fitting"""
        self.depth, start = 0, time()
        target_counts = Counter(self.data['Target'])
        total = sum(target_counts[x] * self.scales[x] for x in target_counts)
        self.score = self.calc_score(self.data['Predict'], self.data.index, metric, initial=True)
        stime = start
        info("Depth  Nodes    Score    DTime (s)  TTime (s)")
        info("=====  =====  =========  =========  =========")
        spc = '%-6s%6s%11.6f%11.1f%10.1f'

        while self.score != 1:
            dtime = time() - stime
            unique_paths_set = set(self.data['Path'])
            dnodes = sum(1 for x in unique_paths_set if len(x) == self.depth)
            log_msg = spc % (self.depth, dnodes, self.score, dtime, time() - start)
            
            # Write to log
            info(log_msg)
            
            stime = time()
            if self.depth == max_depth:
                break
            attempts = 1
            self.get_splits(metric)

            if self.depth == 0 and self.data['Path'].isin(unique_paths_set).all():
                attempts += 1
                while self.data['Path'].isin(unique_paths_set):
                    self.get_splits(metric)
                    if attempts == 30:
                        return

            self.depth += 1
            ends = set(self.data['Path'].apply(lambda x: x[-1]))

            if ends.issubset(['F', 'T']):
                break
        if self.standards is not None: 
            standards = self.data[self.data.index.isin(self.standards)]
            unique_targets = sorted(standards['Target'].unique())
            i = 0
            for target in unique_targets:
                filter_df = standards[standards['Target'] == target]
                unique_predict = sorted(filter_df['Predict'].unique())
                for predict in unique_predict:
                    idxs = ', '.join(map(str, list(filter_df[filter_df['Predict'] == predict].index)))
                    info(f'Target {target} and Predict {predict}: {idxs}')
            if i > len(unique_targets):
                warning('Not all standards correctly predicted')

    def calc_score(self, pred, true, metric, gini=None, initial=False, idxs=None):
        """Calculate a metric to use to fit the DecisionTree"""
        true = self.true[true]
        if initial:
            if metric == 'BalAcc':
                return sklm.balanced_accuracy_score(true, pred)
            if metric == 'Gini':
                return 1 - self.gini_index(true)
        true = pd.Series(true, index=idxs)
        node_1, node_2 = true[pred], true[~pred]
        node_1_label, node_2_label, mixed = self.set_labels(node_1, node_2)

        if metric == 'BalAcc':
            if node_1_label == node_2_label or 0 in [len(node_1), len(node_2)]:
                label = self.set_label(true)
                return self.node_score, (label, label), mixed
            pred = np.where(pred, node_1_label, node_2_label)
            full_pred = self.data['Predict']
            full_pred.loc[idxs] = pred

            score = sklm.balanced_accuracy_score(self.true, np.array(full_pred))
        elif metric == 'Gini':
            score = self.node_score + self.gini_impurity_gain(true, pred)
        return score, (node_1_label, node_2_label), mixed

    def get_splits(self, metric):
        """Split the decision tree for that level"""
        data = self.data  # Store a reference to self.data for faster access
        feats =  list(data.columns[4:])
        class_idx = self.class_idx  # Store a reference to self.class_idx
        labels = self.labels  # Store a reference to self.labels
        calc_score = self.calc_score
        update_tree_and_labels = self.update_tree_and_labels

        max_feats = self.max_feats
        data_len = len(self.data)

        for path in sorted(set(self.data['Path'])):
            if any(x in path for x in ['T', 'F']):
                continue
            self.tscore, self.node_diff = 0, 0
            cols = sorted(random.sample(feats, max_feats))
            other = data[data['Path'] != path]
            o_tar, o_pred = other['Target'], other['Predict']
            n_idxs = data[data['Path'] == path].index
            base_node = class_idx[class_idx.index.isin(n_idxs)]
            node = base_node.loc[:, cols]

            node_tar = data[data.index.isin(n_idxs)]['Target']

            # Check for conditions to terminate or skip this path
            if len(node) <= 0.01 * data_len or node_tar.nunique() == 1:
                self.handle_terminal_path(path, node)
                continue

            node_pred = data[data.index.isin(n_idxs)]['Predict']
            self.node_score = calc_score(np.array(node_pred), np.array(n_idxs), metric, initial=True, idxs=n_idxs)
            splits = self.calculate_splits(node, metric, o_tar, o_pred, node_tar)

            if len(splits) == 0:
                self.handle_terminal_path(path, node)
                continue

            stop = False
            best = splits.sort_values('score', ascending=False).head(1).reset_index().squeeze()

            classes = len(set([int(x) for x in best['classes'].split(',')]))

            if best['score'] < self.node_score or np.isnan(best['score']):
                stop = True
            if classes == 1:
                stop = True
            if metric == 'Gini':
                if best['score'] < 0.005:
                    stop = True
            elif metric == 'BalAcc':
                if best['score'] == 1:
                    stop = True
            if stop:
                terminal = path[::-1].title()[::-1]
                end = node.index
                self.data.loc[self.data.index.isin(end), 'Path'] = terminal
                labels[terminal] = self.labels[path]
                continue
            update_tree_and_labels(path, node, node_tar, best, metric)
        if metric == 'Gini':
            self.score = self.full_gini_score()
        else:
            self.score = self.calc_score(np.array(data['Predict']), data.index, metric, initial=True, idxs=data.index)

    def handle_terminal_path(self, path, node):
        """Handle a terminal path."""
        terminal = path[:-1] + path[-1].capitalize()
        end = node.index
    
        # Update labels
        self.labels[terminal] = self.labels[path]
    
        # Update the 'Path' column in the DataFrame
        self.data.loc[self.data.index.isin(end), 'Path'] = terminal

    def calculate_splits(self, node, metric, o_tar, o_pred, node_tar):
        """Calculate splits for a node based on the specified metric."""
        if metric == 'Gini':
            gini = self.gini_index(node_tar)
        else:
            gini = None

        splits = node.apply(self.test_split, metric=metric, node_tar=node_tar, gini=gini)
        
        # Convert the split results to a DataFrame
        splits = pd.DataFrame(splits.str.split(' ').tolist(), columns=['sign', 'threshold', 'score', 'classes', 'mixed'], index=splits.index)
        splits['score'] = pd.to_numeric(splits['score'], errors='coerce')
        splits = splits[~splits['score'].isna()]
        splits = splits[splits['score'] != 0]

        splits = self.filter_results(splits)

        return splits

    def test_split(self, col, metric, node_tar, gini):
        """Calculate metric based on data feature turning on or off"""
        mixed = 0
        if col.nunique() == 1:
            return f'== 0 {np.nan} 0'

        if col.nunique() <= 2:  # Handling categorical data
            df = self.handle_categorical(col, node_tar)
            if df.shape[0] == 0:
                return  f'> 0 {np.nan} {mixed}'
        else:  # Handling continuous data
            df = self.handle_continuous(col, node_tar)

        if df.shape[0] == 0:
            return  f'> 0 {np.nan} {mixed}'

        idxs = node_tar.index
        results = df.apply(lambda x: self.calc_score(x.values, true=node_tar.index, metric=metric, gini=gini, idxs=idxs)).T
        results = self.filter_results(results)
        result = self.get_best_score(results, node_tar, col)

        return result

    def handle_categorical(self, col, node_tar):
        condition = col > 0
        mask, node_diff = self.handle_condition(condition, node_tar)
        if node_diff > 0.95 * self.node_diff:
            return pd.DataFrame({0: mask})
        return pd.DataFrame()

    def handle_continuous(self, col, node_tar):
        # Handle continuous data
        unique_values = sorted(col.unique())
        numb_unique = len(unique_values)
        q75, q25 = np.percentile(col, [75, 25])
        iqr = q75 - q25
        n = len(col)
        bin_width = (2 * iqr) / (n ** (1/3))
        if bin_width == 0:
            return pd.DataFrame()
        min_col, max_col = min(col), max(col)
        try:
            num_bins = int((max_col - min_col) / bin_width)
            num_thresholds_to_test = min(numb_unique, 20)  # num_bins
        except OverflowError:
            num_thresholds_to_test = numb_unique
        thresholds = np.linspace(min_col, max_col, num_thresholds_to_test)
        results = []
        for threshold in thresholds[:-1]:
            condition = col > threshold
            mask, node_diff = self.handle_condition(condition, node_tar)
            if node_diff >= 0.95 * self.node_diff:
                results.append(pd.DataFrame({threshold: mask}))
        if len(results) > 0:
            return pd.concat(results, axis=1)
        return pd.DataFrame()

    def handle_condition(self, condition, node_tar):
        mask = np.zeros(len(node_tar), dtype=bool)
        mask[condition] = True
        node1, node2 = node_tar[mask], node_tar[~mask]
        node_diff = np.sum([self.find_diff(node) for node in [node1, node2] if len(node) != 0])
        return mask, node_diff

    def filter_results(self, results):
        try:
            lowest = min(results.iloc[:, -1])
        except ValueError:
            return results
        return results[results.iloc[:, -1] == lowest]

    def get_best_score(self, results, node_tar, col):
        score = float(results[0].max())
        if score > self.tscore:
            self.tscore = score
            best_row = results[results[0] == score].head(1)
            threshold = best_row.index[0]
            classes, mixed = list(best_row[1])[0], list(best_row[2])[0]
            send_classes = ','.join([str(x) for x in list(classes)])
            condition = col > threshold
            mask = np.zeros(len(node_tar), dtype=bool)
            mask[condition] = True
            preds = np.where(mask, True, False)
            node1, node2 = node_tar[preds], node_tar[~preds]
            node_diff = np.sum([self.find_diff(node) for node in [node1, node2] if len(node) != 0])
            self.node_diff = node_diff
            return f'> {threshold} {score} {send_classes} {mixed}'
        return f'> 0 {np.nan} 0,0 0'

    def find_diff(self, node):
        values, counts = np.unique(node, return_counts=True)
        scales = np.array([self.scales[value] for value in values])
        result = sorted(scales * counts)
        try:
            res = result[-1] - result[-2]
        except IndexError:
            res = result[-1]
        return res

    def set_labels(self, node1, node2):
        basic_search, mixed = True, 0
        if self.standards is not None and self.depth > 4:
            basic_search = False
            node1_stand = node1[node1.index.isin(self.standards)]
            node2_stand = node2[node2.index.isin(self.standards)]
            if node1_stand.empty and node2_stand.empty:
                basic_search = True

        if basic_search:
            node_1_label = self.set_label(node1)
            node_2_label = self.set_label(node2)
#            if node_1_label == node_2_label:
#                node_1_label, node_2_label = self.set_label(node1, node2=node2)

        else:
            if not node1_stand.empty and node1_stand.nunique() > 1:
                mixed += 1
            if not node2_stand.empty and node2_stand.nunique() > 1:
                mixed += 1
            node_1_label = self.set_label(node1_stand)
            node_2_label = self.set_label(node2_stand)
#            if node_1_label == node_2_label and node_1_label is not None:
#                node_1_label, node_2_label = self.set_label(node1_stand, node2=node2_stand)
            if node_1_label is None and node_2_label is not None:
                node_1_label = self.set_label(node1, ignore=[node_2_label])
                if node_1_label is None:
                    node = pd.concat([node1, node2])
                    node = node[~node.isin([node_2_label])].unique()
                    node_1_label = random.choice(node)
            if node_2_label is None and node_1_label is not None:
                node_2_label = self.set_label(node2, ignore=[node_1_label])
                if node_2_label is None:
                    node = pd.concat([node1, node2])
                    node = node[~node.isin([node_1_label])].unique()
                    node_2_label = random.choice(node)
        return node_1_label, node_2_label, mixed

    def set_label(self, node, node2=None, ignore=None):
        """Assign label based off scaled value and output order
        Label assignment favors weighted amount in group, and those
        being equal, the position in output order, with later output
        order being favored"""
        if ignore is not None:
            node = node[~node.isin(ignore)]
        if len(node) == 0:
            return None
        values, counts = np.unique(node, return_counts=True)
        if len(values) == 1 and node2 is None:
            return values[0]
        weighted_counts = counts * np.array([self.scales[value] for value in values])
        max_index = np.argmax(weighted_counts)
        max_label = values[max_index]

        if node2 is not None:
            second_max_label = values[np.argsort(weighted_counts)[-2]] if len(values) > 1 else None

            values2, counts2 = np.unique(node2, return_counts=True)
            weighted_counts2 = counts2 * np.array([self.scales[value] for value in values2])
            max_label2 = values2[np.argmax(weighted_counts2)]
            second_max_label2 = values2[np.argsort(weighted_counts2)[-2]] if len(values2) > 1 else None

            ratio1 = max(weighted_counts) / sum(weighted_counts)
            ratio2 = max(weighted_counts2) / sum(weighted_counts2)

            if ratio1 >= ratio2:
                return max_label, second_max_label2
            else:
                return second_max_label, max_label2

        return max_label

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

    def gini_impurity_gain(self, node_tar, pred):
        total = len(node_tar)
        gini_before_split = self.gini_index(node_tar)
        weighted_gini_after_split = 0
        node1 = node_tar[pred]
        node2 = node_tar[~pred]
        for node in [node1, node2]:
            weight = len(node) / total
            gini_after_split = self.gini_index(node)
            weighted_gini_after_split += weight * gini_after_split
        gini_gain = gini_before_split - weighted_gini_after_split
        return gini_gain

    def gini_index(self, data):
        if len(data) == 0:
            return 0
        labels, counts = np.unique(data, return_counts=True)
        counts = np.array([self.scales[x] * y for x, y in zip(labels, counts)])
        probabilities = counts / sum(counts)
        gini = 1.0 - np.sum(probabilities ** 2)
        return gini


    def update_tree_and_labels(self, path, node, node_tar, best, metric):
        descriptor, sign, threshold = best[['index', 'sign', 'threshold']]
        self.rules[path] = f'{descriptor} {sign} {threshold}'

        indices = node[descriptor] > float(threshold)
        true, false = [int(x) for x in best['classes'].split(',')]

        pred = pd.Series(np.where(indices, true, false), index=node.index)

        self.labels[path + 't'] = true
        self.labels[path + 'f'] = false
        
        # Modify the 'Path' column in a vectorized way
        self.data.loc[pred[pred == true].index, 'Path'] += 't'
        self.data.loc[pred[pred == false].index, 'Path'] += 'f'
    
        # Calculate 'Predict' using a lambda function
        old_pred = self.data.loc[self.data.index.isin(node.index)]['Predict']
        self.data['Predict'] = self.data['Path'].apply(lambda x: self.labels[x])
        true_series = self.data[self.data.index.isin(pred.index)]['Target']
        pred_series = self.data[self.data.index.isin(pred.index)]['Predict']
#        if self.depth > 3:
#            print(list(old_pred))
#            print(list(pred_series))
#            print(list(true_series))
        self.check_df.loc[list(self.data.index), f'From Update {self.depth}'] = np.array(self.data['Predict'])
        self.check_df.loc[list(self.data.index), f'From Node {self.depth + 1}'] = np.array(self.data['Predict'])
        self.check_df.loc[list(self.data.index), 'Path'] = self.data['Path']

    def full_gini_score(self):
        data = self.data[['Target', 'Path']]
        total = len(data)
        nodes = np.unique(data['Path'])
        nodes = [self.data[self.data['Path'] == x]['Target'] for x in nodes]
        gini_level = 0
        for node in nodes:
            weight = len(node) / total
            node_gini = self.gini_index(node)
            gini_level += weight * node_gini
        return 1 - gini_level

    def clean_up(self):
        for attr in ['data', 'class_idx', 'score', 'max_feats', 'depth',
                     'node_diff', 'use_split', 'tscore', 'node_score',
                     'scales', 'weights', 'true']:
            if hasattr(self, attr):
                delattr(self, attr)

    def predict(self, preds, i=None, results=False, depth=False):
        """Predict the outcome of decision tree"""
        start = time()
        rls = self.rules
        if not depth:
            depth = 999
        preds.reset_index(drop=True, inplace=True)
        preds['Path'] = ''
        preds['Predict'] = pd.Series([''] * len(preds), index=preds.index)
        for path in sorted(rls, key=len):
            if len(path) + 1 == depth:
                 break
            pred_path = preds['Path']
            feat, cond, val = rls[path].split()
            val = float(val)
            node = preds.loc[pred_path == path, feat]
            if cond == '<=':
                condition = node <= val
            else:
                condition = node > val
            preds.loc[node[condition].index, 'Path'] += 'r'
            preds.loc[node[~condition].index, 'Path'] += 'l'
        preds['Predict'] = preds['Path'].apply(lambda x: self.labels[x])
        try:
            return preds[[self.cross, 'Predict']]
        except KeyError:
            return preds['Predict']


class MyEncoder:
    """
    Class for encoding and decoding categorical data for machine learning models.
    This class allows sorting and reverse ordering of classes for proper encoding.
    """
    def __init__(self):
        """Initialize the encoder."""
        self.classes = np.array([])
        self.encoding = {}
        self.ordered = False
        self.reverse = False

    def fit(self, y: list, ordered: bool = False, reverse: bool = False) -> None:
        """
        Fit the encoder to the given data.

        Args:
            y (list): Data to fit the encoding to.
            ordered (bool): Keep the data in the same ordering as given.
            reverse (bool): Reverse the order of data.

        Returns:
            None
        """
        base = list(set(y))
        if ordered:
            base = sorted(base, key=y.index, reverse=reverse)
        self.classes = np.array(base)
        self.encoding = {x: i for (i, x) in enumerate(self.classes)}
        self.ordered = ordered
        self.reverse = reverse

    def transform(self, y: list) -> np.ndarray:
        """
        Transform an array-like of data to encoded values.

        Args: y (list): Data to be encoded.
        Returns: np.ndarray: Encoded values.
        """
        for x in set(y):
            if x not in self.encoding:
                self.encoding[x] = x
        return np.array([self.encoding[x] for x in y])

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """
        Inverse transform encoded values back to original values.

        Args: y (np.ndarray): Encoded values to be decoded.
        Returns: np.ndarray: Decoded original values.
        """
        return np.array([self.classes[x] for x in y])


def tanimoto(list1, list2):
    """Calculate the Tanimoto coefficient of two lists"""
    return np.dot(list1, list2) / (np.dot(list1, list1) + np.dot(list2, list2) - np.dot(list1, list2))

def chunks(lst, chunk_size):
    """Cut work into chunk size bits"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

#To work on
def unit_read_convert(string, unit):
    """Read in a string containing starting unit and convert to preferred unit"""
    if any(x in unit for x in ['atm', 'Hg', 'torr', 'pa', 'bar']):
        print("Pressure conversion")
