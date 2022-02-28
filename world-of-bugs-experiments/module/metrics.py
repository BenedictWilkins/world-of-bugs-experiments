#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 19-02-2022

   Metrics used to evaluate the trained models. Note that a positive label 1 indicates a bug in the observation. The score should reflect this, with higher scores indiciated "more anomalous".
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

from cProfile import label
import torch
import numpy as np
import sklearn
import sklearn.metrics
from sklearn.metrics._ranking import _binary_clf_curve

import matplotlib as mpl
import matplotlib.pyplot as plt
import wandb


__all__ = ("ROC", "PrecisionRecall", "AccuracyPrecisionRecallF1")


class LabelThresholdMetric:

    def __init__(self, label_thresholds=[0, 10, 50, 100, 200, 400]):
        assert len(label_thresholds) > 0
        self.label_thresholds = label_thresholds
        
    def __call__(self, logger, score, label, ax=None, **kwargs):
        score = score.cpu().numpy() if torch.is_tensor(score) else score
        label = label.cpu().numpy() if torch.is_tensor(label) else label
        assert score.shape == label.shape
        results = []
        labels = []
        for lt in self.label_thresholds:
            _label = (label > lt).astype(np.float32)
            labels.append(_label)
            results.append(self.forward(score, _label))
        # plot results
        kwargs['title'] = kwargs.get('title', self.name)
        xs, ys = [r['x'] for r in results], [r['y'] for r in results]
        if ax is None:
            _, ax = plt.subplots(1,1, figsize=(3,3))

        ax = self.plot(ax, xs, ys, self.label_thresholds, **kwargs)
        ax = self.plot_random_performance(ax, labels)
        #fig.canvas.draw() # ensure drawn...
        logger.log({f"{self.name}/{kwargs['title']}" : wandb.Image(ax)})
        return dict(zip(self.label_thresholds, results))

    def plot(self, ax, xs, ys, label_thresholds, alpha=0.8, title="",  x_lim=[0,1], y_lim=[0,1], x_label="",  y_label="",font={'fontsize':8}, cmap="viridis", legend=False):
        cmap = mpl.colormaps[cmap]
        ax.set_aspect('equal')
        ax.set_prop_cycle('color', cmap(1 - np.linspace(0,1,len(label_thresholds))))
        ax.set_title(title, fontdict=font)
        ax.set_xlabel(x_label), ax.set_ylabel(y_label)
        ax.set_xlim(x_lim), ax.set_ylim(y_lim)
        for x,y,lt in zip(xs, ys, label_thresholds):
            ax.plot(x, y, alpha=alpha, label=f"τ={lt}")
        if legend:
            ax.legend()
        return ax

    def plot_random_performance(self, fig, labels):
        raise NotImplementedError()

class ROC(LabelThresholdMetric):
    
    def forward(self, score, label):
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(label, score, drop_intermediate=False) 
        return dict(x=fpr, y=tpr, auc=sklearn.metrics.auc(fpr, tpr))

    def random_performance(self, label):
        return [[0,1],[0,1]]

    def plot_random_performance(self, ax, _):
        ax.plot(*self.random_performance(None), linestyle="--", color='black')
        return ax

    @property
    def name(self):
        return "Receiver Operate Characteristic Curve"
    
    @property
    def abreviation(self):
        return "(ROC)"

class PrecisionRecall(LabelThresholdMetric):

    def forward(self, score, label):
        p, r, thresholds = sklearn.metrics.precision_recall_curve(label, score) 
       
        return dict(y=p, x=r, auc=sklearn.metrics.auc(r, p))

    def random_performance(self, label):
        label_ratio =  label.sum() / len(label)
        return [[0, 1],[label_ratio, label_ratio]]

    def plot_random_performance(self, ax, labels):
        random_performance = list(sorted([self.random_performance(label) for label in labels], key=lambda x: x[1][0]))
        ax.plot(*random_performance[0], linestyle="--", color='black')
        ax.plot(*random_performance[-1], linestyle="--", color='black')
        return ax

    @property
    def name(self):
        return "Precision Recall Curve"
    
    @property
    def abreviation(self):
        return "(PR)"

class AccuracyPrecisionRecallF1(LabelThresholdMetric):

    def forward(self, score, label):
        # threshold the score

        fps, tps, thresholds = _binary_clf_curve(label, score)

        precision = tps / (tps + fps)
        precision[np.isnan(precision)] = 0
        recall = tps / tps[-1]
        f1 = (2 * precision * recall) / (precision + recall)
        accuracy = (tps + (fps[-1] - fps)) / (tps[-1] + fps[-1])

    def plot(self, ax, x, y):
        pass 


    @property
    def name(self):
        return "Accuracy Precision Recall F1"
    
    @property
    def abreviation(self):
        return "(APRF1)"
        