from model.ae_flow_model import AE_Flow_Model
from dataloader import load
import torch
import numpy as np
import sklearn
from train import find_threshold, eval_model, calculate_metrics
import torch.utils.data as data
import os
import pickle
from experiment import Experiment
import wandb
def uncertainty_table(true, preds, stds, std_threshold=0.025, fname="ue.txt"):
    """
    
    """
    
    # 1. get array of high uncertainty predictions
    high_uncertainty = (stds > std_threshold).astype(int) # 1 if high uncertainty, otherwise 0
    # 2. get array of correct predictions
    correct = preds == true # True if correct, otherwise False
    incorrect= preds != true #True if incorrect, otherwise correct


    # 3. get array of uncertainty correct predictions
    correct_uncertain = high_uncertainty[correct]
    
    # uncertainty for incorrect
    incorrect_uncertain = high_uncertainty[incorrect]
    
    # 4. count
    # uncertain correct
    c_high = np.sum(correct_uncertain)
    
    # certain correct
    c_low = len(correct_uncertain) - c_high

    # uncertain incorrect
    ic_high = np.sum(incorrect_uncertain)


    # certain incorrect
    ic_low = len(incorrect_uncertain) - ic_high
    
    results = f"""
            LOW    HIGH
  CORRECT    {c_low}    {c_high}
INCORRECT    {ic_low}  {ic_high}
    """

    if not os.path.exists("results/ue"):
        os.mkdir("results/ue")
    
    with open("results/ue/ue.txt", "w") as f:
        f.write(results)

    print(results)

    return c_low, c_high, ic_low, ic_high

def main(args):
    # model_names = ['1.pt', '59.pt', '85.pt', '91.pt', '68.pt']
    model_names = ["1.pt", "59.pt", "85.pt"]
    model_results = []

    train_loader, train_abnormal, test_loader = load(data_dir='chest_xray',batch_size=8, num_workers=3, subset=False)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # assert not torch.equal(torch.load(f'models/model_seed/85.pt', map_location='cpu'), torch.load(f'models/model_seed/59.pt', map_location='cpu'))

    optimal_threshold = 0
    

    for idx, model_path in enumerate(model_names):
        if os.path.exists(f"models/model_seed/results{idx}.pkl"):
            model_results = pickle.load(open(f"models/model_seed/results{idx}.pkl","rb"))
            model_threshold = pickle.load(open(f"models/model_seed/thresh{idx}.pkl","rb"))
            # save anomaly scores
            optimal_threshold += model_threshold
            print(f"Done with processing model {idx}")
            continue
        torch.cuda.empty_cache()
        # ress = {}
        
        model = AE_Flow_Model(subnet_architecture='conv_like', n_flowblocks=8)
        state_dict = torch.load(f'models/model_seed/{model_path}', map_location='cpu')
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        anomaly_scores, true_labels = [], []



        threshold_dataset = torch.utils.data.ConcatDataset([train_loader, train_abnormal])
        threshold_loader = data.DataLoader(threshold_dataset, num_workers = 3, batch_size=64)

        wandb.init()
        model_threshold = find_threshold(0, model, threshold_loader)

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

        true, anomaly_scores = eval_model(0, model, test_loader, threshold=model_threshold, return_only_anomaly_scores=True, running_ue_experiments=True)
        
        print(f"Just to check, running final inference using the test data: {calculate_metrics(true, anomaly_scores, model_threshold)} using model {model_path}")


        # save anomaly scores
        scores = np.array(anomaly_scores, dtype=float)
        model_results.append([true, scores])
        print(f"Done with processing model {idx}")

        # i'm not taking any chances here
        pickle.dump(model_threshold, open(f"models/model_seed/thresh{idx}.pkl","wb"))
        pickle.dump(model_results, open(f"models/model_seed/results{idx}.pkl","wb"))

        optimal_threshold += model_threshold

    # calculate optimal threshold
    optimal_threshold /= len(model_names)

    # get means and stds
    #softmax_res = [torch.nn.functional.sigmoid(torch.tensor([model_results[i][1]]).unsqueeze(-1)).squeeze() for i in range(len(model_names))]
    #softmax_res = [np.array(s) for s in softmax_res]
    res = [model_results[i][1] for i in range(len(model_names))]

    # calculate mean and standard deviation
    true_labels = np.array(model_results[0][0], dtype=int)
    means = np.mean(res, axis=0, dtype=float)
    stds = np.std(res, axis=0, dtype=float)

    # prediction based on means
    preds = np.array(means > optimal_threshold, dtype=int)

    print("MEAN DIST:", np.mean(means), np.std(means))
    print("STD:", np.mean(stds), np.std(stds))
    results = uncertainty_table(true_labels, preds, stds, std_threshold=0.0785)

    n_models = [1, 2, 3, 4, 5]

    # Accuracy
    accuracies = []

    accuracies.append(sklearn.metrics.accuracy_score(y_true=true_labels, y_pred=model_results[0][1]))
    

    # Classification error
    # print(sklearn.metrics.accuracy_score(y_true=model_results[0]['true'], y_pred=model_results[0]['preds'], normalize=False))

    # Negative log-likelihood
    print(sklearn.metrics.log_loss(y_true=model_results[0][0], y_pred=model_results[0][1]))
    
    # Brier
    # print(sklearn.metrics.brier_score_loss(y_true=model_results[0][0], y_pred=model_results[0][1]))





if __name__ == '__main__':
    
    """# Training settings
    parser = argparse.ArgumentParser(description='AE Normalized Flow')

    # Optimizer hyper-parameters 
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size to use for training')
    parser.add_argument('--loss_alpha', type=float, default=0.5,
                        help='')
    parser.add_argument('--loss_beta', type=float, default=0.9,
                        help='')
    parser.add_argument('--optim_lr', type=float, default=2e-3,
                        help='')
    parser.add_argument('--optim_momentum', type=float, default=0.9, 
                        help='')
    parser.add_argument('--optim_weight_decay', type=float, default=10e-5,
                        help='')
    parser.add_argument('--dataset',default='chest_xray', type=str, help='Which dataset to run. Choose from: [OCT2017, chest_xray, ISIC, BRATS, MIIC]')
    parser.add_argument('--model',default='ae_flow', type=str, help='Which dataset to run. Choose from: [autoencoder, fastflow, ae_flow]')

    parser.add_argument('--subnet_architecture', default='conv_like', type=str,
                        help='Which subflow architecture to use when using the ae_flow model: conv_like or resnet_like')

    # Other hyper-parameters
    parser.add_argument('--data_dir', default='../data/', type=str,
                        help='Directory where to look for the data. For jobs on Lisa, this should be $TMPDIR.')

    parser.add_argument('--final_experiments', default=False, type=bool, help='Whether to save results as for final experiments')
    
    parser.add_argument('--ue_model', default=False, type=bool, help='Whether to just train a model with a specific seed and save it.')

    parser.add_argument('--n_validation_folds', default=5, type=int, help='')
    parser.add_argument('--n_flowblocks', default=8, type=int, help='')
    parser.add_argument('--fully_deterministic', default=True, type=bool, help='Whether to run with torch.backends.cudnn.deterministic')
    parser.add_argument('--torch_benchmark', default=False, type=bool, help='Whether to run with torch.backends.cudnn.benchmark')

    parser.add_argument('--epochs', default=15, type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to use in the data loaders.' +
                             'To have a truly deterministic run, this has to be 0.')"""

    #args = parser.parse_args()
    main("FUCK YOU")
