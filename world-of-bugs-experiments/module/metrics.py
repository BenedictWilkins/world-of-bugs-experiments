#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 19-02-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import torch
import numpy as np
import sklearn

__all__ = ("ROC", "PrecisionRecall")

class LabelThresholdMetric:

    def __init__(self, label_thresholds=[0, 10, 50, 100, 200, 400]):
        assert len(label_thresholds) > 0
        self.label_thresholds = label_thresholds
        
    def __call__(self, score, label, **kwargs):
        score = score.cpu().numpy() if torch.is_tensor(score) else score
        label = label.cpu().numpy() if torch.is_tensor(label) else label
        assert score.shape == label.shape

        results = []
        for lt in self.label_thresholds:
            _label = (label > lt).astype(np.float32)
            result = self.forward(score, _label, **kwargs)
            results.append((lt, result))
        return results

class ROC(LabelThresholdMetric):
    
    def forward(self, score, label):
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(label, score) 
        return dict(fpr=fpr, tpr=tpr, auc=sklearn.metrics.auc(fpr, tpr))

class PrecisionRecall(LabelThresholdMetric):

    def forward(self, score, label):
        p, r, thresholds = sklearn.metrics.precision_recall_curve(label, score) 
        return dict(precision=p, recall=r, auc=sklearn.metrics.auc(r, p))