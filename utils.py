import scipy
from torchvision.utils import make_grid
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, auc, roc_curve, precision_recall_curve
import numpy as np
from sklearn.model_selection import KFold
from torchmetrics.classification import BinaryPrecisionRecallCurve
import time
import torch
def thr_to_f1(thr, Y_test, predictions):
   """Calculating the negative (binary) F1 score for a set of predictions and true values using a proposed threshold 
   to minimize using fmin"""
   return - f1_score(Y_test, np.array(predictions>=thr, dtype=np.int), average='binary')

def optimize_threshold(anomaly_scores, true_labels):
    """Optimizing the threshold used to classify a anomaly score as an anomaly. We use fmin for this with
    an initial guess of the mean + 1std as most samples usually are normal.

    Args:
        anomaly_scores (float:list): 
        true_labels (bool:list): 

    Returns:
        threshold (float): the found optimal threshold
    """
    # bprc = BinaryPrecisionRecallCurve()
    # p,r, thr = bprc(anomaly_scores, true_labels)
    # f1_scores_torch = 2*r*p/(r+p)
    anomaly_scores = [item for sublist in [tensor.cpu().numpy() for tensor in anomaly_scores] for item in sublist]
    true_labels = [item for sublist in [tensor.cpu().numpy() for tensor in true_labels] for item in sublist]

    # t1 = time.time()
    precision, recall, thresholds = precision_recall_curve(true_labels, anomaly_scores)
    f1_scores = 2*recall*precision/(recall+precision)
    # weights = confusion_matrix(true_labels, anomaly_scores).sum(axis=1)
    # weighted_f1_scores = np.average(f1_scores, weights=weights)
    # print('Best threshold: ', thresholds[np.argmax(f1_scores)])
    # print(f"Approach 1: {time.time() - t1}")
    # print(np.argmax(f1_scores), torch.argmax(f1_scores_torch))


    return thresholds[np.argmax(f1_scores)]
    t1 = time.time()
    print('Best F1-Score: ', np.max(f1_scores))
    print('VS:')
    print(scipy.optimize.fmin(thr_to_f1, args=(true_labels, anomaly_scores), x0=np.mean(anomaly_scores), disp=0))
    print(f"Approach 2: {time.time() - t1}")
    assert False

    return scipy.optimize.fmin(thr_to_f1, args=(true_labels, anomaly_scores), x0=np.mean(anomaly_scores)+np.std(anomaly_scores), disp=0)

#TODO: Do we want to keep this function?
def sample_images(model, device):
        rec_images = model(model.sample_images_normal.to(device)).squeeze(dim=1)
        grid1 = make_grid(model.sample_images_normal.to(device) + rec_images, nrow = 2)
        rec_images = model(model.sample_images_abnormal.to(device)).squeeze(dim=1)
        grid2 = make_grid(model.sample_images_abnormal.to(device) + rec_images, nrow = 2)

        return {"abnormal reconstruction images": grid1, "normal reconstruction images": grid2}

def calculate_metrics(true, anomaly_scores, threshold, _print=False):
    pred = np.array(anomaly_scores>=threshold, dtype=int)
    if _print: print(f"Number of predicted anomalies in the (test-)set: {np.sum(pred)}")
    
    tn, fp, fn, tp = confusion_matrix(true, pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(true, pred)

    results = {
        'AUC': auc(fpr, tpr),
        'ACC' : (tp + tn)/(tp+fp+fn+tn),
        'SEN' : tp / (tp+fn),
        'SPE' : tn / (tn + fp),
        'F1' : f1_score(true, pred, average='binary')
    }
    return results