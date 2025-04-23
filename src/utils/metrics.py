import torch
import numpy as np
from sklearn import metrics
from easydict import EasyDict as edict
from ._utils import *

def get_metric(is_multiclass):
    if is_multiclass:
        metric = DefaultClassificationMetrics
    else:
        metric = MultiLabelClassificationMetrics
    return metric

def get_activation(is_multiclass):
    if is_multiclass:
        act = nn.Softmax(dim=1)
    else:
        act = nn.Sigmoid()
    return act

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

def mean_roc_auc(truths, predictions):
    """
    Calculating mean ROC-AUC:
        Assuming that the last dimension represent the classes
    """
    _truths = np.array(deepcopy(truths))
    _predictions = np.array(deepcopy(predictions))  
    n_classes = _predictions.shape[-1]
    avg_roc_auc = 0 
    for class_num in range(n_classes):
        auc = 0.5
        tar = (_truths[:,class_num] + _truths[:,class_num]**2 ) / 2
        if tar.sum() > 0:
            auc = metrics.roc_auc_score(tar, _predictions[:,class_num], 
                                        average='macro', 
                                        sample_weight=_truths[:, class_num] ** 2 + 1e-06, 
                                        multi_class = 'ovo')            
        avg_roc_auc += auc 
    return avg_roc_auc / n_classes

def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.
    Arguments
    ---------
    ranks : zerro-based ranks of positive images
    nres  : number of positive images
    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap

def compute_map(ranks, gnd, kappas=[]):
    """
    Computes the mAP for a given set of returned results.
         Usage:
           map = compute_map (ranks, gnd)
                 computes mean average precsion (map) only
           map, aps, pr, prs = compute_map (ranks, gnd, kappas)
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query
         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    map = 0.
    nq = len(gnd) # number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)

        # sorted positions of positive and junk images (0 based)
        pos  = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgndj)]

        k = 0;
        ij = 0;
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # compute precision @ k
        pos += 1 # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j]); 
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)

    return map, aps, pr, prs

class PCA():
    """
    Class to  compute and apply PCA.
    """
    def __init__(self, dim=256, whit=0.5):
        self.dim = dim
        self.whit = whit
        self.mean = None

    def train_pca(self, cov):
        """
        Takes a covariance matrix (np.ndarray) as input.
        """
        d, v = np.linalg.eigh(cov)
        eps = d.max() * 1e-5
        n_0 = (d < eps).sum()
        if n_0 > 0:
            d[d < eps] = eps

        # total energy
        totenergy = d.sum()

        # sort eigenvectors with eigenvalues order
        idx = np.argsort(d)[::-1][:self.dim]
        d = d[idx]
        v = v[:, idx]

        print("keeping %.2f %% of the energy" % (d.sum() / totenergy * 100.0))

        # for the whitening
        d = np.diag(1. / d**self.whit)

        # principal components
        self.dvt = np.dot(d, v.T)

    def apply(self, x):
        # input is from numpy
        if isinstance(x, np.ndarray):
            if self.mean is not None:
                x -= self.mean
            return np.dot(self.dvt, x.T).T

        # input is from torch and is on GPU
        if x.is_cuda:
            if self.mean is not None:
                x -= torch.cuda.FloatTensor(self.mean)
            return torch.mm(torch.cuda.FloatTensor(self.dvt), x.transpose(0, 1)).transpose(0, 1)

        # input if from torch, on CPU
        if self.mean is not None:
            x -= torch.FloatTensor(self.mean)
        return torch.mm(torch.FloatTensor(self.dvt), x.transpose(0, 1)).transpose(0, 1)
    
class DefaultClassificationMetrics:
    
    def __init__(self, n_classes, int_to_labels=None, act_threshold=0.5, mode=""):
        self.mode = mode
        self.prefix = ""
        if mode:
            self.prefix = mode + "_"
        self.n_classes = n_classes
        if int_to_labels is None:
            int_to_labels = {val:'class_'+str(val) for val in range(n_classes)}
        self.int_to_labels = int_to_labels
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.truths = []
        self.predictions = []
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.truths = []
        self.predictions = []        
    
    # add predictions to confusion matrix etc
    def add_preds(self, y_pred, y_true, using_knn=False):
        if not using_knn:
            y_pred = y_pred.max(dim = 1)[1].data
        y_true = y_true.flatten().detach().cpu().numpy()
        y_pred = y_pred.flatten().detach().cpu().numpy()
        self.truths += (y_true.tolist())
        self.predictions += (y_pred.tolist())
        np.add.at(self.confusion_matrix, (y_true, y_pred), 1)
    
    # Calculate and report metrics
    def get_value(self, use_dist=True):
        if use_dist:
            synchronize()
            truths = sum(dist_gather(self.truths), [])
            predictions = sum(dist_gather(self.predictions), [])
        else:
            truths = self.truths
            predictions = self.predictions     
        
        accuracy = metrics.accuracy_score(truths, predictions)
        precision = metrics.precision_score(truths, predictions, average='macro', zero_division=0)
        recall = metrics.recall_score(truths, predictions, average='macro', zero_division=0)
        f1 = metrics.f1_score(truths, predictions, average='macro', zero_division=0) 
        kappa = metrics.cohen_kappa_score(truths, predictions, 
                                          labels=list(range(self.n_classes)), weights='quadratic')
        
        # return metrics as dictionary
        return edict({self.prefix + "accuracy" : round(accuracy, 3),
                        self.prefix + "precision" : round(precision, 3),
                        self.prefix + "recall" : round(recall, 3),
                        self.prefix + "f1" : round(f1, 3),
                        self.prefix + "cohen_kappa" : round(kappa, 3)})
    
class MultiLabelClassificationMetrics:
    
    def __init__(self, n_classes, int_to_labels=None, act_threshold=0.5, mode=""):
        self.mode = mode
        self.prefix = ""
        if mode:
            self.prefix = mode + "_"       
        self.n_classes = n_classes
        if int_to_labels is None:
            int_to_labels = {val:'class_'+str(val) for val in range(n_classes)}
        self.labels = np.arange(n_classes)
        self.int_to_labels = int_to_labels
        self.truths = []
        self.predictions = []
        self.activation = nn.Sigmoid()
        self.act_threshold = act_threshold
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.truths = []
        self.predictions = []        
    
    # add predictions to confusion matrix etc
    def add_preds(self, y_pred, y_true, using_knn=False):       
        y_true = y_true.int().detach().cpu().numpy()
        y_pred = self.preds_from_logits(y_pred)
        self.truths += (y_true.tolist())
        self.predictions += (y_pred.tolist())
    
    # pass signal through activation and thresholding
    def preds_from_logits(self, preds):
        preds = self.activation(preds)
        return preds.detach().cpu().numpy()
    
    def threshold_preds(self, preds):
        preds = preds > self.act_threshold
        if isinstance(preds, torch.Tensor):
            return preds.int().detach().cpu().numpy()
        else:
            return preds * 1
    
    # Calculate and report metrics
    def get_value(self, use_dist=True):
        if use_dist:
            synchronize()
            truths = np.array(sum(dist_gather(self.truths), []))
            predictions = np.array(sum(dist_gather(self.predictions), []))
        else:
            truths = np.array(self.truths)
            predictions = np.array(self.predictions) 
            
        try:
            mAP = metrics.average_precision_score(truths, predictions, average='macro')
        except:
            mAP = 0.                    
        roc_auc = mean_roc_auc(truths, predictions)        
        
        predictions = self.threshold_preds(predictions)
        self.confusion_matrix = metrics.multilabel_confusion_matrix(truths, predictions)     
        
        accuracy = metrics.accuracy_score(truths, predictions)
        precision = metrics.precision_score(truths, predictions, average='macro', 
                                            labels=self.labels, zero_division=0)
        recall = metrics.recall_score(truths, predictions, average='macro', 
                                            labels=self.labels, zero_division=0)
        f1 = metrics.f1_score(truths, predictions, average='macro', 
                                            labels=self.labels, zero_division=0)
        
        # return metrics as dictionary
        return edict({self.prefix + "accuracy" : round(accuracy, 3),
                        self.prefix + "mAP" : round(mAP, 3),
                        self.prefix + "precision" : round(precision, 3),
                        self.prefix + "recall" : round(recall, 3),
                        self.prefix + "f1" : round(f1, 3),
                        self.prefix + "roc_auc" : round(roc_auc, 3)}
                    )