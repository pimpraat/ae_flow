import scipy
from sklearn.metrics import f1_score
from torchvision.utils import make_grid

  
# Given anomaly scores and the true labels (0/1) calculate using bisection to optimize for the highest F1 score:
def calculate_given_threshold(proposed_threshold, anomaly_scores, true_labels):
    preds = [score > proposed_threshold for score in anomaly_scores]
    score = f1_score(y_true=true_labels[0].cpu(), y_pred=preds[0].cpu())
    return score

def optimize_threshold(anomaly_scores, true_labels):
    opt_threshold = scipy.optimize.bisect(f=calculate_given_threshold, a=0.000, b=1.000, 
                                          args=(anomaly_scores, true_labels), xtol=0.01)
    return opt_threshold

def sample_images(model, device):
        rec_images = model(model.sample_images_normal.to(device)).squeeze(dim=1)
        grid1 = make_grid(model.sample_images_normal.to(device) + rec_images, nrow = 2)
        rec_images = model(model.sample_images_abnormal.to(device)).squeeze(dim=1)
        grid2 = make_grid(model.sample_images_abnormal.to(device) + rec_images, nrow = 2)

        return {"abnormal reconstruction images": grid1, "normal reconstruction images": grid2}
