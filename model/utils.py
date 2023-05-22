import scipy
from sklearn.metrics import f1_score
from torchvision.utils import make_grid
import numpy as np
  

def thr_to_f1(thr, Y_test, predictions):
   """Calculating the negative (binary) F1 score for a set of predictions and true values using a proposed threshold 
   to minimize using fmin"""
   return - f1_score(Y_test, np.array(predictions>thr, dtype=np.int), average='binary')

def optimize_threshold(anomaly_scores, true_labels):
    """Optimizing the threshold used to classify a anomaly score as an anomaly. We use fmin for this with
    an initial guess of the mean + 1std as most samples usually are normal.

    Args:
        anomaly_scores (float:list): 
        true_labels (bool:list): 

    Returns:
        threshold (float): the found optimal threshold
    """
    anomaly_scores = [item for sublist in [tensor.cpu().numpy() for tensor in anomaly_scores] for item in sublist]
    true_labels = [item for sublist in [tensor.cpu().numpy() for tensor in true_labels] for item in sublist]
    return scipy.optimize.fmin(thr_to_f1, args=(true_labels, anomaly_scores), x0=np.mean(anomaly_scores)+np.std(anomaly_scores), disp=0)

#TODO: Do we want to keep this function?
def sample_images(model, device):
        rec_images = model(model.sample_images_normal.to(device)).squeeze(dim=1)
        grid1 = make_grid(model.sample_images_normal.to(device) + rec_images, nrow = 2)
        rec_images = model(model.sample_images_abnormal.to(device)).squeeze(dim=1)
        grid2 = make_grid(model.sample_images_abnormal.to(device) + rec_images, nrow = 2)

        return {"abnormal reconstruction images": grid1, "normal reconstruction images": grid2}