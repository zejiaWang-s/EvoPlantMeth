from __future__ import division, print_function

from collections import OrderedDict
import numpy as np
import pandas as pd
import sklearn.metrics as skm
from scipy.stats import kendalltau
from six.moves import range

from .data import CPG_NAN, OUTPUT_SEP
from .utils import get_from_module

def cor(y, z):
    return np.corrcoef(y, z)[0, 1]

def kendall(y, z, nb_sample=100000):
    if len(y) > nb_sample:
        idx = np.random.choice(len(y), nb_sample, replace=False)
        y, z = y[idx], z[idx]
    return kendalltau(y, z)[0]

def mad(y, z):
    return np.mean(np.abs(y - z))

def mse(y, z):
    return np.mean((y - z)**2)

def rmse(y, z):
    return np.sqrt(mse(y, z))

def auc(y, z, round=True):
    if round: y = np.round(y)
    if len(y) == 0 or len(np.unique(y)) < 2: return np.nan
    return skm.roc_auc_score(y, z)

def acc(y, z, round=True):
    if round: y, z = np.round(y), np.round(z)
    return skm.accuracy_score(y, z)

def tpr(y, z, round=True):
    if round: y, z = np.round(y), np.round(z)
    return skm.recall_score(y, z)

def tnr(y, z, round=True):
    if round: y, z = np.round(y), np.round(z)
    c = skm.confusion_matrix(y, z)
    return c[0, 0] / c[0].sum() if c[0].sum() > 0 else np.nan

def mcc(y, z, round=True):
    if round: y, z = np.round(y), np.round(z)
    return skm.matthews_corrcoef(y, z)

def f1(y, z, round=True):
    if round: y, z = np.round(y), np.round(z)
    return skm.f1_score(y, z)

def cat_acc(y, z):
    return np.mean(y.argmax(axis=1) == z.argmax(axis=1))

CLA_METRICS = [auc, acc, tpr, tnr, f1, mcc]
REG_METRICS = [mse, mad, cor, rmse]
CAT_METRICS = [cat_acc]

def evaluate(y, z, mask=CPG_NAN, metrics=CLA_METRICS):
    z = z.ravel()
    if mask is not None:
        t = y != mask
        y, z = y[t], z[t]
    p = OrderedDict()
    for metric in metrics:
        p[metric.__name__] = metric(y, z) if len(y) else np.nan
    p['n'] = len(y)
    return p

def evaluate_regression(y, z, mask=CPG_NAN, metrics=REG_METRICS):
    z = z.ravel()
    if mask is not None:
        t = y != mask
        y, z = y[t], z[t]
    p = OrderedDict()
    for metric in metrics:
        p[metric.__name__] = metric(y, z) if len(y) else np.nan
    p['n'] = len(y)
    return p

def evaluate_cat(y, z, metrics=CAT_METRICS, binary_metrics=None):
    idx = y.sum(axis=1) > 0
    y, z = y[idx], z[idx]
    p = OrderedDict()
    for metric in metrics:
        p[metric.__name__] = metric(y, z)
    if binary_metrics:
        for i in range(y.shape[1]):
            for metric in binary_metrics:
                p['%s_%d' % (metric.__name__, i)] = metric(y[:, i], z[:, i])
    p['n'] = len(y)
    return p

def get_output_metrics(output_name):
    _output_name = output_name.split(OUTPUT_SEP)
    if _output_name[0] == 'cpg': return CLA_METRICS
    if _output_name[0] == 'bulk': return REG_METRICS + CLA_METRICS
    if _output_name[-1] in ['diff', 'mode', 'cat2_var']: return CLA_METRICS
    if _output_name[-1] == 'mean': return REG_METRICS + CLA_METRICS + [kendall]
    if _output_name[-1] == 'var': return REG_METRICS + [kendall]
    raise ValueError('Invalid output name "%s"!' % output_name)

def evaluate_outputs(outputs, preds):
    perf = []
    for output_name in outputs:
        _output_name = output_name.split(OUTPUT_SEP)
        if _output_name[-1] in ['cat_var']:
            tmp = evaluate_cat(outputs[output_name], preds[output_name], binary_metrics=[auc])
        else:
            metrics = get_output_metrics(output_name)
            tmp = evaluate(outputs[output_name], preds[output_name], metrics=metrics)
        tmp_df = pd.DataFrame({'output': output_name, 'metric': list(tmp.keys()), 'value': list(tmp.values())})
        perf.append(tmp_df)
    perf = pd.concat(perf)
    return perf[['metric', 'output', 'value']].sort_values(['metric', 'value'])

def evaluate_outputs_regression(outputs, preds):
    perf = []
    for output_name in outputs:
        tmp = evaluate_regression(outputs[output_name], preds[output_name], metrics=REG_METRICS + [kendall])
        tmp_df = pd.DataFrame({'output': output_name, 'metric': list(tmp.keys()), 'value': list(tmp.values())})
        perf.append(tmp_df)
    perf = pd.concat(perf)
    return perf[['metric', 'output', 'value']].sort_values(['metric', 'value'])

def is_binary_output(output_name):
    _output_name = output_name.split(OUTPUT_SEP)
    return _output_name[0] == 'cpg' or _output_name[-1] in ['diff', 'mode', 'cat2_var']

def evaluate_curve(outputs, preds, fun=skm.roc_curve, mask=CPG_NAN, nb_point=None):
    curves = []
    for output_name in outputs.keys():
        if not is_binary_output(output_name): continue
        output, pred = outputs[output_name].round().squeeze(), preds[output_name].squeeze()
        idx = output != mask
        output, pred = output[idx], pred[idx]
        x, y, thr = fun(output, pred)
        
        length = min(len(x), len(y), len(thr))
        idx_slice = np.linspace(0, length - 1, nb_point).astype(np.int32) if nb_point and length > nb_point else slice(0, length)
        x, y, thr = x[idx_slice], y[idx_slice], thr[idx_slice]
        
        curves.append(pd.DataFrame({'output': output_name, 'x': x, 'y': y, 'thr': thr}))
    return pd.concat(curves) if curves else None

def unstack_report(report):
    index = list(report.columns[~report.columns.isin(['metric', 'value'])])
    report = pd.pivot_table(report, index=index, columns='metric', values='value').reset_index()
    report.columns.name = None
    
    columns = list(report.columns)
    sorted_columns = []
    for fun in CAT_METRICS + CLA_METRICS + REG_METRICS:
        for column in columns:
            if column.startswith(fun.__name__): sorted_columns.append(column)
            
    sorted_columns = index + sorted_columns + [col for col in columns if col not in index + sorted_columns]
    report = report[sorted_columns]
    
    order = [('auc', False)] if 'auc' in report.columns else ([('mse', True)] if 'mse' in report.columns else ([('acc', False)] if 'acc' in report.columns else []))
    if order:
        report.sort_values([x[0] for x in order], ascending=[x[1] for x in order], inplace=True)
    return report

def get(name):
    return get_from_module(name, globals())