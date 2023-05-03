import scipy
from sklearn.metrics import f1_score

  
# Given anomaly scores and the true labels (0/1) calculate using bisection to optimize for the highest F1 score:
def calculate_given_threshold(proposed_threshold, anomaly_scores, true_labels):
    # print(f"Anomaly scores: {anomaly_scores}")
    preds = [x >= proposed_threshold for x in anomaly_scores]
    # print(f"Calculate threshold using threshold-value of: {proposed_threshold}")
    # preds = [anomaly_scores >= proposed_threshold]
    # print(f"y_true={true_labels.tolist()}, y_pred={preds.tolist()}")
    # print(true_labels[0])
    # try:
    #     print(true_labels[0].tolist())
    # except:
    #     pass
    score = f1_score(y_true=true_labels[0].cpu(), y_pred=preds[0].cpu())
    return score

def optimize_threshold(anomaly_scores, true_labels):

    # print(f"Anomaly scores: {anomaly_scores}, True labels: {true_labels}")
    opt_threshold = scipy.optimize.bisect(f=calculate_given_threshold, a=0.000, b=1.000, 
                                          args=(anomaly_scores, true_labels), xtol=0.01)
    print(f"Found optimal threshold: {opt_threshold}")
    return opt_threshold
