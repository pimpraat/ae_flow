import scipy
from sklearn.metrics import f1_score
from torchvision.utils import make_grid
import torch
import numpy as np

  
# Given anomaly scores and the true labels (0/1) calculate using bisection to optimize for the highest F1 score:
def calculate_given_threshold(proposed_threshold, anomaly_scores, true_labels):
    preds = [score > proposed_threshold for score in anomaly_scores]
    score = -f1_score(y_true=true_labels[0].cpu(), y_pred=preds[0].cpu())
    print(f"Using threshold of {proposed_threshold} found F1 score of {score}, as preds are: {[preds]}, because: {anomaly_scores}")
    return score

def optimize_threshold(anomaly_scores, true_labels):

    sc = list(anomaly_scores)
    print(f"sc: {sc}")

    print(f"type of anomaly_scores: {type(sc)}") 
    # print(f"min: {torch.min(sc)}")
    print(f"min: {torch.min(torch.Tensor(sc))}")

    opt_threshold = scipy.optimize.bisect(f=calculate_given_threshold, a=np.min(sc[0].cpu()), b=np.max(sc[0].cpu()), 
                                          args=(anomaly_scores, true_labels)) #set tolerace to 0.1?

    return opt_threshold

def sample_images(model, device):
        rec_images = model(model.sample_images_normal.to(device)).squeeze(dim=1)
        grid1 = make_grid(model.sample_images_normal.to(device) + rec_images, nrow = 2)
        rec_images = model(model.sample_images_abnormal.to(device)).squeeze(dim=1)
        grid2 = make_grid(model.sample_images_abnormal.to(device) + rec_images, nrow = 2)

        return {"abnormal reconstruction images": grid1, "normal reconstruction images": grid2}
