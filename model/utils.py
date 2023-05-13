import scipy
from sklearn.metrics import f1_score, accuracy_score
from torchvision.utils import make_grid
import torch
import numpy as np

# Just a little timit nugget of code:
# python -mtimeit -s'l=[[1,2,3],[4,5,6], [7], [8,9]]*99' '[item for sublist in l for item in sublist]'
  
# Given anomaly scores and the true labels (0/1) calculate using bisection to optimize for the highest F1 score:
def calculate_given_threshold(proposed_threshold, anomaly_scores, true_labels):
    preds = [score > proposed_threshold for score in anomaly_scores]
    score = -f1_score(y_true=true_labels[0].cpu(), y_pred=preds[0].cpu())
    print(f"Using threshold of {proposed_threshold} found F1 score of {score}.")
    return score
        
def thr_to_accuracy(thr, Y_test, predictions):
   
   #TODO: to binary
   sc = f1_score(Y_test, np.array(predictions>thr, dtype=np.int), average='macro')
#    print(f"score: {sc}")
   return -sc

# Reason for fmin: https://www.southampton.ac.uk/~fangohr/training/python14/labs/lab_optimisation/index.html
def optimize_threshold(anomaly_scores, true_labels):

    anomaly_scores = [tensor.cpu().numpy() for tensor in anomaly_scores]
    anomaly_scores = [item for sublist in anomaly_scores for item in sublist]
    true_labels = [tensor.cpu().numpy() for tensor in true_labels]
    true_labels = [item for sublist in true_labels for item in sublist]

    # threshold = np.min(anomaly_scores)
    # best_threshold = 0.0
    # best_score = 0.0
    # while threshold < np.max(anomaly_scores):
    #     p = [score > threshold for score in anomaly_scores]
    #     score = f1_score(true_labels, p, average='macro')
    #     if score == 0: continue
    #     if score >= best_score: 
    #          best_threshold = threshold
    #          best_score = score
    #     threshold+=0.001
    # return best_threshold


    # return scipy.optimize.minimize_scalar(thr_to_accuracy, args=(true_labels, anomaly_scores))

    # print(f'anomaly scores: {anomaly_scores}')
    # assert(False)
    best_thr = scipy.optimize.fmin(thr_to_accuracy, args=(true_labels, anomaly_scores), x0=np.mean(anomaly_scores))
    return best_thr

    sc = [tensor.cpu().numpy() for tensor in anomaly_scores]
    sc = [item for sublist in sc for item in sublist]

    # print(f"sc: {sc}")

    # print(f"type of anomaly_scores: {type(sc)}") 
    # print(f"min: {torch.min(sc)}")
    # print(f"min: {np.min(sc)}")

    opt_threshold = scipy.optimize.bisect(f=calculate_given_threshold, a=np.partition(sc, 10)[10], b=np.partition(sc, -10)[-10], 
                                          args=(anomaly_scores, true_labels)) #set tolerace to 0.1?

    return opt_threshold

def sample_images(model, device):
        rec_images = model(model.sample_images_normal.to(device)).squeeze(dim=1)
        grid1 = make_grid(model.sample_images_normal.to(device) + rec_images, nrow = 2)
        rec_images = model(model.sample_images_abnormal.to(device)).squeeze(dim=1)
        grid2 = make_grid(model.sample_images_abnormal.to(device) + rec_images, nrow = 2)

        return {"abnormal reconstruction images": grid1, "normal reconstruction images": grid2}
