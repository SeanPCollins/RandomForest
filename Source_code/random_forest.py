#!/usr/bin/env python
"""
Title:        RandomForest
Author:       Sean Collins
Date:         2022 08 09 
Description:  Random Forest training code using homebrew decision tree code
Usage:        %./random_forest.py [options] [JOB_NAME]
"""

from itertools import combinations
from logging import info
import os
from os.path import exists
import pickle as pkl

import numpy as np
import pandas as pd
import pubchempy as pcp
from openbabel import pybel
import sklearn.metrics as sklm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels

from config import Options
from ml_tools import DecisionTree, MyEncoder

np.warnings.filterwarnings('ignore')

# ID values for system state
NOT_RUN = 0
RUNNING = 1
FINISHED = 2

__version_info__ = (0, 7, 0, 0)
__version__ = "%i.%i.%i.%i" % __version_info__


class RandomForest(object):
    """
    Single property to hold all information about the RandomForest training,
    process. It can pickle itself at any time, by calling dump_state()
    """
    def __init__(self, options):
        self.opt = options
        self.break_down = None
        self.skip = None
        self.enc = None
        self.out_o = None
        self.data = self.data_prep()
        self.state = {'init': NOT_RUN,
                      'data_break': NOT_RUN,
                      'trained': NOT_RUN}
        trees = ['Tree_{}'.format(x + 1) for x in range(self.opt.getint('trees'))]
        self.state.update({x: NOT_RUN for x in trees})
        self.tree_info = {x: {} for x in list(range(self.opt.getint('trees') + 1)) +  ['Train', 'Test']}
        self.state['init'] = FINISHED

    def dump_state(self):
        """Write the Forest.state file for the RF training"""
        info("Writing state file, {}.rfs".format(self.opt.get('job_name')))
        my_state = open("{}.rfs".format(self.opt.get('job_name')), 'wb')
        pkl.dump(self, my_state)
        my_state.close()

    def data_prep(self):
        """Load all nessecary data and prepare it for the decision tree"""
        cross = self.opt.get('cross_column')
        target = self.opt.get('target')
        data = pd.read_csv(self.opt.get('target_csv') + '.csv', index_col=False, encoding='latin1')
        if 'Binding' in target and 'CERAPP' in self.opt.get('target_csv'):
              counts = data[data['Binding Count'] > 3].index
        try:
            data = data[[cross, target]]
        except KeyError:
            data = pd.read_csv(self.opt.get('target_csv') + '.csv')
            data = data[[cross, target]]
        data.rename(columns={target: 'Target'}, inplace=True)
        for csv in self.opt.gettuple('descriptor_csvs'):
            try:
                new = pd.read_csv(csv + '.csv', index_col=False)
            except IOError:
                new = pd.read_pickle(csv + '.pkl')
            if new.index.name == cross:
                new = pd.read_csv(csv + '.csv', index_col=False)
            data = data.merge(new, on=cross, how='left')
        if 'Binding' in target and 'CERAPP' in self.opt.get('target_csv'):
            data = data[data.index.isin(counts)]
        data.drop_duplicates(subset=[cross], inplace=True)
        ignore = list(set(self.opt.gettuple('ignore_columns')).intersection(data.columns))
        data = data[[x for x in data.columns if x not in ignore + ['Unnamed: 0']]]
        data = data.dropna(subset=['Target']).reset_index(drop=True)
        features = data.columns[2:]
        data = data.dropna(subset=features, how='any').reset_index(drop=True)
        out_o = self.opt.gettuple('output_order')
        if out_o == ():
            out_o = sorted(set(data['Target'].astype('str')))
        endp = sorted(set(data['Target'].astype('str')))
        out_o = sorted(endp, key=out_o.index)
        enc = MyEncoder()
        enc.fit(out_o, ordered=True, reverse=True)
        info("================")
        info("Classes Encoded:")
        info("================")
        mx = len(max([str(x) for x in out_o], key=len))
        for enc_class in out_o:
            pads = "{:%s}: {}" % (mx + 1)
            info(pads.format(enc_class, enc.transform([enc_class])[0]))
        self.out_o = enc.transform(out_o)
        self.enc = enc
        data['Target'] = enc.transform(data['Target'].astype('str'))
        if not self.opt.getbool('pruned'):
            var = data[features].var()
            drops = var[var == 0].index
            data = data.drop(drops, axis=1)
            features = data.columns[2:]
            sim, have = pd.DataFrame(index=features, columns=features), {}
            for feat in features:
                if sum(data[feat]) not in have.keys():
                    have[sum(data[feat])] = [feat]
                else:
                    have[sum(data[feat])].append(feat)
            for val in have.values():
                for i, j in combinations(val, 2):
                    tan = sklm.jaccard_score(data[i], data[j], average='macro')
                    if tan == 1:
                        sim[i][j] = 1
                        sim[j][i] = 1
            sim.dropna(how='all', inplace=True)
            sim.dropna(axis=1, how='all', inplace=True)
            sim.fillna(value=0, inplace=True)
            while not sim.empty:
                max_sim = 0
                for val in sim.columns:
                    sims = sum(sim[val])
                    if sims > max_sim:
                        max_sim, drops = sims, val
                drop_col = list(sim.loc[sim[drops] == 1][drops].index)
                data.drop(labels=drop_col, axis=1, inplace=True)
                sim.drop(labels=drop_col + [drops], axis=0, inplace=True)
                sim.drop(labels=drop_col + [drops], axis=1, inplace=True)
        cols = data.columns.tolist()[2:]
        self.skip = 1 - 0.02 ** (1 / self.opt.getfloat('trees'))
        data['Predict'] = pd.Series(0 * len(data), index=data.index)
        data['Path'] = pd.Series([''] * len(data), index=data.index)
        data = data[[cross, 'Target', 'Predict', 'Path'] + cols]
        self.features = [x for x in features if x in data.columns]
        return data

    def fit(self):
        """Create the datasets for the RF training and test sets"""
        data, cross = self.data, self.opt.get('cross_column')
        trees = self.opt.getint('trees')
        ranges = range(1, trees + 1)
        if self.state['data_break'] == NOT_RUN:
            self.data_break = pd.DataFrame(np.nan, index=data.index,
                                           columns=self.tree_info.keys())
            self.data_break.drop(['Train', 'Test'], axis=1, inplace=True)
            self.res = data[[cross, 'Target']]
            self.res['Forest'] = 0
            info("Creating training and test sets")
            ratio = self.opt.getfloat('ratio')
            _xtr, _xte, ytr, yte = train_test_split([1] * len(data),
                                                    data['Target'],
                                                    test_size=ratio,
                                                    stratify=data['Target'])
            conds = [data.index.isin(ytr.index), data.index.isin(yte.index)]
            self.data_break[0] = np.select(conds, ['Tr', 'Te'])
            tree_data = data[data.index.isin(ytr.index)]
            for i in ranges:
                _xtr, _xte, ytr, yte = train_test_split([1] * len(tree_data),
                                                        tree_data['Target'],
                                                        test_size=ratio,
                                                        stratify=tree_data['Target'])
                conds = [data.index.isin(ytr.index), data.index.isin(yte.index)]
                self.data_break[i] = np.select(conds, ['Tr', 'Te'])
            self.data_break[cross] = self.data[cross]
            self.state['data_break'] = FINISHED
            self.dump_state()
        max_depth = self.opt.getint('max_depth')
        for i in ranges:
            if self.state['Tree_{}'.format(i)] == FINISHED:
                continue
            info("Training Tree {}...".format(i))
            tr_idx = self.data_break[self.data_break[i] == 'Tr'].index
            try:
                os.mkdir('Tree_{}'.format(i))
            except OSError:
                pass
            os.chdir('Tree_{}'.format(i))
            dt = DecisionTree(data[data.index.isin(tr_idx)], cross=cross,
                              job_name='Tree_{}'.format(i), max_feats=self.skip,
                              out_o=self.out_o, enc=self.enc)
            dt.fit(max_depth=max_depth, metric=self.opt.get('metric'))
            self.predict_all(dt)
            os.chdir('..')
            info("Trained Tree {}".format(i))
            dt.clean_up()
            self.tree_info[i]['DT'] = dt
            preds = dt.predict(data.copy()).rename(columns={'Predict': i})
            self.res = self.res.merge(preds, on=cross)
            self.state['Tree_{}'.format(i)] = FINISHED
            self.dump_state()
        classes = self.enc.inverse_transform(self.out_o)
        self.print_results('Tr', ranges, classes)
        self.print_results('Te', ranges, classes)
        self.res['Forest'] = forest_pred(self.res[ranges], 'majority')
        self.print_results('RF', ['Train', 'Test'], classes)
        self.state['trained'] = FINISHED
        exit()
        self.dump_state()

    def print_results(self, pref, ranges, classes):
        """Print the results from Random Forest training"""
        names = {'Tr': 'Training', 'Te': 'Testing', 'RF': 'Random Forest'}
        cross = self.opt.get('cross_column')
        out_o = list(self.out_o)
        if len(classes) == 2:
            info("-------------------------------------")
            info(names[pref] + ' Basic Data and Stats')
            info("-------------------------------------")
            info("Tree    TP     FP     TN     FN   Precis Recall")
            info("===== ====== ====== ====== ====== ====== ======")
        base = "%-7s%5d%7d%7d%7d%7.4f%7.4f"
        output_mat = {}
        for i in ranges:
            pred_id = i
            try:
                idxs = self.data_break[self.data_break[i] == pref].index
            except KeyError:
                pred_id = 'Forest'
                if i == 'Train':
                    idxs = self.data_break[self.data_break[0] == 'Tr'].index
                if i == 'Test':
                    idxs = self.data_break[self.data_break[0] == 'Te'].index
            ids = self.data[self.data.index.isin(idxs)][cross]
            reses = self.res[self.res[cross].isin(ids)]
            tar = reses['Target']
            pred = reses[pred_id]
            if len(classes) > 2:
                self.multi_metrics(tar, pred, i, pref)
            labels = sorted(unique_labels(tar, pred), key=out_o.index)
            con_mat = confusion_matrix(tar, pred, labels=labels)
            mat = np.array(con_mat.ravel())
            output_mat[i] = {}
            for x, y in zip(labels, con_mat):
                output_mat[i][x] = y
            if len(classes) == 2:
                self.metrics(tar, pred, i, pref)
                info(base % (i, mat[3], mat[1], mat[0], mat[2],
                             self.tree_info[i][pref + '_pre'],
                             self.tree_info[i][pref + '_rec']))
        info("-------------------------------------")
        info(names[pref] + ' Holistic Stats')
        info("-------------------------------------")
        if len(classes) == 2:
            info("Tree  Accura BalAcc   F1    F1.5   MCC   Jaccar")
        else:
            info("Tree  Accura BalAcc   APre    F1   MCC   Jaccar")
        info("===== ====== ====== ====== ====== ====== ======")
        base = "%-6s%5.4f%7.4f%7.4f%7.4f%7.4f%7.4f"
        for i in ranges:
            ti = self.tree_info[i]
            info(base % (i, ti[pref + '_acc'], ti[pref + '_bacc'],
                         ti[pref + '_f1'], ti[pref +'_f1.5'], ti[pref + '_mcc'],
                         ti[pref + '_jac']))
        if pred_id == 'Forest':
            for i in ranges:
                info("-------------------------------------")
                info("{} Forest Confusion Matrix".format(i))
                info("-------------------------------------")
                result = output_mat[i]
                groups = self.enc.inverse_transform(sorted(result.keys(),
                                                           key=out_o.index))
                names = tuple(['Actual'] + list(groups))
                lens = [len(str(x)) + 1 for x in names]
                lens[0] = max(lens)
                lens = np.clip(lens, 6, max(lens))
                dash = tuple('=' * (x - 1) for x in lens)
                spc = ''.join(['%-{}s'.format(x) for x in lens])
                info(spc % names)
                info(spc % dash)
                df = pd.DataFrame(0, index=classes, columns=classes)
                for x in out_o:
                    if x not in result.keys():
                        continue
                    label = self.enc.inverse_transform([x])
                    print_out = tuple(list(label) + list(result[x]))
                    info(spc % print_out)
                    for y, v in zip(classes, result[x]):
                        df[y][label] = v
                df.to_csv(i + '_Forest_cm.csv')

    def metrics(self, tar, pred, i, pref):
        """Calculate relevent metrics for output"""
        tar, pred = 1 - tar, 1 - pred #Done to get around reverse ordering
        ti = self.tree_info[i]
        stats = sklm.precision_recall_fscore_support(tar, pred)
        ti[pref + '_pre'] = stats[0][1]
        ti[pref + '_rec'] = stats[1][1]
        ti[pref + '_f1'] = stats[2][1]
        ti[pref + '_acc'] = sklm.accuracy_score(tar, pred)
        ti[pref + '_bacc'] = sklm.recall_score(tar, pred, average='macro')
        ti[pref + '_f1.5'] = sklm.fbeta_score(tar, pred, beta=1.5)
        ti[pref + '_mcc'] = sklm.matthews_corrcoef(tar, pred)
        ti[pref + '_jac'] = sklm.jaccard_score(tar, pred, average='macro')
        self.tree_info[i] = ti

    def multi_metrics(self, tar, pred, i, pref):
        """Calcuulate relevent metric for output for multiclassification
           problem"""
        ti = self.tree_info[i]
        stats = sklm.precision_recall_fscore_support(tar, pred)
        ti[pref + '_pre'] = np.mean(stats[0])
        ti[pref + '_rec'] = np.mean(stats[1]) #This is BalAcc
        ti[pref + '_f1'] = ti[pref + '_pre'] #Switch to pre for easy print
        ti[pref + '_acc'] = sklm.accuracy_score(tar, pred)
        ti[pref + '_bacc'] = sklm.recall_score(tar, pred, average='macro')
        ti[pref + '_f1.5'] = np.mean(stats[2]) #Switched to F1 for easy print
        ti[pref + '_mcc'] = sklm.matthews_corrcoef(tar, pred)
        ti[pref + '_jac'] = sklm.jaccard_score(tar, pred, average='macro')
        self.tree_info[i] = ti

    def predict_all(self, dt):
        """Predict all levels of a tree"""
        rls, preds, labels = dt.rules, self.data.copy(deep=True), dt.labels
        cross = self.opt.get('cross_column')
        preds['Path'] = pd.Series([''] * len(preds), index=preds.index)
        old_depth, results = 1, preds[[cross, 'Target']]
        for path in sorted(rls, key=len):
            depth = len(path) + 1
            if depth > old_depth:
                results[old_depth] = preds['Path']
            rule = rls[path]
            feat, cond, val = rule.split()
            node = preds[preds['Path'] == path]
            right = node[node[feat] >= float(val)].index
            left = node[node[feat] < float(val)].index
            if cond == '<':
                right, left = left, right
            preds.iloc[right, preds.columns.get_loc('Path')] += 'r'
            preds.iloc[left, preds.columns.get_loc('Path')] += 'l'
            old_depth = depth
        results[old_depth] = preds['Path']
        for col in results.columns[2:]:
            results[col] = results[col].apply(lambda x: labels[x])
        results['Forest'] = results[col]
        results.to_csv('All_Predictions.csv')
 

    def predict(self, job_name):
        """Given a csv of SMILES, predict their activity"""
        tp = pd.read_csv('/home/scollins/bin/tox_names.csv', index_col=0)
        tp = list(tp.columns)
        csv = self.opt.get('predict_csv')
        df = pd.read_csv(csv, index_col=False)
        df_col = list(df.columns)
        if exists('Features.csv'):
            base = pd.read_csv('Features.csv', index_col=0)
        else:
            base = df['SMILES'].apply(create_fp, tp=tp)
            base.to_csv('Features.csv')
        df = df.join(base)
        df['Path'] = ''
        df['Pred'] = 0
        predict_trees = self.opt.gettuple('predict_trees')
        if predict_trees == ():
            predict_trees = range(1, self.opt.getint('trees') + 1)
        else:
             predict_trees = [int(i) for i in predict_trees]
        depth = self.opt.getint('predict_depth')
        if depth == 0:
             depth = self.opt.getint('max_depth')
        for i in predict_trees:
            dt = self.tree_info[i]['DT']
            df['Pred'] += dt.predict(df, depth=depth)
        df.drop(['Predict', 'Path'], axis=1, inplace=True)
        df['Pred'] = df['Pred'] / float(len(predict_trees))
        df['Prob'] = 2 * abs(df['Pred'] - 0.5)
        df['Pred'] = df['Pred'].apply(lambda x: int(x + 0.5))
        df['Prob'] = df['Prob'].apply(lambda x: '%.3f' % x)
        df[df_col + ['Pred', 'Prob']].to_csv(csv.split('.csv')[0] + '_' +
                                             job_name +  '_Pred.csv')
        df.to_csv(csv.split('.csv')[0] + '_' + job_name +  '_Full_Pred.csv')


def create_fp(smi, tp):
    """Given a SMILES string, create a super fingerprint"""
    most = [1024, 56, 307, 166, 881, 729]
    mol_desc = ['atoms', #Number of atoms
                'HBA1', #Number of Hydrogen Bond Acceptors 1
                'HBA2', #Number of Hydrogen Bond Acceptors 2 (Different Method)
                'HBD', #Number of Hydgron Bond Donors
                'nF', #Number of Flourine Atoms
                'bonds', #Number of bonds
                'sbonds', #Number of single bonds
                'abonds', #Number of aromatic bonds
                'dbonds', #Number of double bonds
                'tbonds', #Number of triple bonds
                'logP', #Octanol/Water Partition Coefficient
                'MP', #Melting point
                'MR', #Molar refractivity
                'MW', #Molecular Weight
                'TPSA'] #Topological polar surface area
    fps = pd.Series(0, index=['fp2_F{}'.format(x) for x in range(most[0])] +
                             ['fp3_F{}'.format(x) for x in range(most[1])] +
                             ['fp4_F{}'.format(x) for x in range(most[2])] +
                             ['maccs_F{}'.format(x) for x in range(most[3])] +
                             ['PC_F{}'.format(x) for x in range(most[4])] +
                             tp + mol_desc).astype(float)
    mol = pybel.readstring('smi', smi)
    for key, val in mol.calcdesc().iteritems():
        if key not in mol_desc:
            continue
        fps[key] = val
    for i, fp in enumerate(['fp2', 'fp3', 'fp4', 'maccs']):
        mfp = mol.calcfp(fptype=fp)
        for x in mfp.bits:
            fps['{}_F{}'.format(fp, x)] = 1
    mol = pcp.get_compounds(smi, namespace='smiles', as_dataframe=True).squeeze()
    for i, bit in enumerate(mol['cactvs_fingerprint']):
        fps['PC_F{}'.format(i)] = bit
    return fps

def forest_pred(x, vote):
    """Calculate the forest result of a dataset"""
    if vote == 'majority':
        return x.mode(axis=1).min(axis=1)
    if vote == 'mean':
        return x.mean(axis=1).apply(lambda y: int(y + 0.5))


def main():
    """Main Execution of the code"""
    main_opts = Options(code='RF')
    info("Starting RandomForest {}".format(__version__))
    job_name= main_opts.get('job_name')
    rfs_name = '{}.rfs'.format(job_name)
    if exists(rfs_name):
        info("Found previous state: loading RandomForest...")
        load_rfs = open(rfs_name, 'rb')
        random_forest = pkl.load(load_rfs)
        random_forest.opt = main_opts
        load_rfs.close()
    else:
        info("Starting a new RandomForest training...")
        random_forest = RandomForest(main_opts)
        random_forest.dump_state()
    info('Succesfully Loaded Data')
    if not random_forest.state['trained'] == FINISHED:
        info("RandomForest not trained, training now...")
        random_forest.fit()
    if random_forest.opt.getbool('predict'):
        info("Looking to predict results...")
        random_forest.predict(job_name)


if __name__ in '__main__':
    main()
