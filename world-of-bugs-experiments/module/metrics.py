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
        
    def __call__(self, logger, score, label, **kwargs):
        score = score.cpu().numpy() if torch.is_tensor(score) else score
        label = label.cpu().numpy() if torch.is_tensor(label) else label
        assert score.shape == label.shape
        results = []
        for lt in self.label_thresholds:
            _label = (label > lt).astype(np.float32)
            results.append(self.forward(score, _label))
        # plot results
        kwargs['title'] = kwargs.get('title', self.name)
        xs, ys = [r['x'] for r in results], [r['y'] for r in results]
        fig = self.plot(xs, ys, self.label_thresholds, **kwargs)
        fig.canvas.draw() # ensure drawn...
        logger.log({f"{self.name}/{kwargs['title']}" : wandb.Image(fig)})

    def plot(self, xs, ys, label_thresholds, alpha=0.8, title="", figsize=(3,3), x_lim=[0,1], y_lim=[0,1], x_label="",  y_label="",font={'fontsize':10}, cmap="viridis", legend=False):
        cmap = mpl.colormaps[cmap]
        fig, ax = plt.subplots(1,1, figsize=figsize)
        ax.set_aspect('equal')
        ax.set_prop_cycle('color', cmap(1 - np.linspace(0,1,len(label_thresholds))))
        ax.set_title(title, fontdict=font)
        ax.set_xlabel(x_label), ax.set_ylabel(y_label)
        ax.set_xlim(x_lim), ax.set_ylim(y_lim)
        for x,y,lt in zip(xs, ys, label_thresholds):
            ax.plot(x, y, alpha=alpha, label=f"τ={lt}")
        if legend:
            ax.legend()
        return fig

class ROC(LabelThresholdMetric):
    
    def forward(self, score, label):
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(label, score, drop_intermediate=False) 
        return dict(x=fpr, y=tpr, auc=sklearn.metrics.auc(fpr, tpr))

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

    def plot(self, x, y):
        pass 


    @property
    def name(self):
        return "Accuracy Precision Recall F1"
    
    @property
    def abreviation(self):
        return "(APRF1)"
        