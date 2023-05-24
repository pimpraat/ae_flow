from model.ae_flow_model import AE_Flow_Model
from dataloader import load
import torch
import numpy as np
import sklearn
from train import find_threshold, eval_model, calculate_metrics
import torch.utils.data as data


# model_names = ['1.pt', '59.pt', '85.pt', '91.pt', '68.pt']
model_names = ["85.pt", "59.pt"]
model_results = []

train_loader, train_abnormal, test_loader = load(data_dir='chest_xray',batch_size=64, num_workers=3, subset=False)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

assert not torch.equal(torch.load(f'models/model_seed/85.pt', map_location='cpu'), torch.load(f'models/model_seed/59.pt', map_location='cpu'))

for idx, model_path in enumerate(model_names):
    torch.cuda.empty_cache()
    # ress = {}
    model = AE_Flow_Model(subnet_architecture='resnet_like', n_flowblocks=8)
    model.load_state_dict(torch.load(f'models/model_seed/{model_path}', map_location='cpu'))
    model = model.to(device)
    model.eval()
    anomaly_scores, true_labels = [], []



    threshold_dataset = torch.utils.data.ConcatDataset([train_loader, train_abnormal])
    threshold_loader = data.DataLoader(threshold_dataset, num_workers = 3, batch_size=64)

    optimial_threshold = find_threshold(0, model, threshold_loader, running_ue_experiments=True)

    # thr_anomaly_scores, thr_true_labels = [], []
    # for batch_idx, data in enumerate(threshold_loader):
    #     torch.cuda.empty_cache()
    #     x = data[0]
    #     y = data[1]    
    #     original_x = x.to(device)
    #     reconstructed_x = model(original_x).squeeze(dim=1)


    #     anomaly_score = model.get_anomaly_score(_beta=0.9,original_x=original_x, reconstructed_x=reconstructed_x)
                                

    #     thr_anomaly_scores.append(anomaly_score)
    #     thr_true_labels.append(y)


    # optimal_threshold = optimize_threshold(anomaly_scores, true_labels)

    # for batch_idx, data in enumerate(test_loader):
    #     x = data[0]
    #     y = data[1]    
    #     original_x,y = x.to(device), y.to(device)
    #     reconstructed_x = model(original_x).squeeze(dim=1)
    #     recon_loss = model.get_reconstructionloss(original_x, reconstructed_x)
    #     flow_loss = model.get_flow_loss(bpd=True)
    #     loss = 0.5 * flow_loss + (1-0.5) * recon_loss
    
    #     anomaly_score = model.get_anomaly_score(_beta=0.9, 
    #                                             original_x=original_x, reconstructed_x=reconstructed_x)
    #     anomaly_scores.append(anomaly_score)

    #     true_labels.append(y)
    
    # true = [item for sublist in [tensor.cpu().numpy() for tensor in true_labels] for item in sublist]
    # anomaly_scores = [item for sublist in [tensor.cpu().numpy() for tensor in anomaly_scores] for item in sublist]

    true, anomaly_scores = eval_model(0, model, test_loader, threshold=optimial_threshold, return_only_anomaly_scores=True, running_ue_experiments=True)
    
    print(f"Just to check, running final inference using the test data: {calculate_metrics(true, anomaly_scores, optimial_threshold)} using model {model_path}")

    preds = np.array(anomaly_scores >= optimial_threshold, dtype=int)
    # ress['true'] = true
    # ress['preds'] = preds
    model_results.append([true, preds])
    print(f"Done with processing model {idx}")

n_models = [1, 2, 3, 4, 5]

# Accuracy
accuracies = []

print(model_results[0])

accuracies.append(sklearn.metrics.accuracy_score(y_true=model_results[0][0], y_pred=model_results[0][1]))
accuracies.append(sklearn.metrics.accuracy_score(y_true=np.mean([model_results[0][0], model_results[1][0]], axis=0, dtype=int), y_pred=np.mean([model_results[0][1], model_results[1][1]], axis=0, dtype=int)))
print(accuracies)

# Classification error
# print(sklearn.metrics.accuracy_score(y_true=model_results[0]['true'], y_pred=model_results[0]['preds'], normalize=False))

# Negative log-likelihood
print(sklearn.metrics.log_loss(y_true=model_results[0][0], y_pred=model_results[0][1]))
 
# Brier
# print(sklearn.metrics.brier_score_loss(y_true=model_results[0][0], y_pred=model_results[0][1]))





