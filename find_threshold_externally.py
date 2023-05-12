from model.utils import optimize_threshold
from train import calculate_metrics
import json
# First load the data

openfile = ""

with open('sample.json', 'r') as openfile: data_object = json.load(openfile)

n_epochs = 10

best_f1, best_f1_used_threshold = 0,0

#For every epoch: load the corresponding data, calculate the optimal threshold on training_complete data, 
# then calculate the f1score on the test_loader set, and keep track of both?
for epoch in range(n_epochs):
    epoch_object = data_object[epoch]

    optimal_threshold = optimize_threshold(epoch_object['anomaly_score_traincomplete'], epoch_object['true_label_traincomplete'])
    results = calculate_metrics(true=epoch_object['true_label_test'], anomaly_scores=epoch_object['anomaly_score_test'], threshold=optimal_threshold)

    if results['F1'] > best_f1:
        best_f1 = results['F1']
        best_f1_used_threshold = optimal_threshold
        print(f"Found an improvement at epoch {epoch}, using a threshold of {optimal_threshold}")


# Now save it to a file named to load automatically for evaluation again on Lisa



#Now using this found


# Finally print