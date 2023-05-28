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
import argparse

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

    if args.score_threshold > 0:
        optimal_threshold = args.score_threshold

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
    results = uncertainty_table(true_labels, preds, stds, std_threshold=args.std_threshold)



if __name__ == '__main__':
    
    # Training settings
    parser = argparse.ArgumentParser(description='Deep ensemble for AE Normalized Flow')

    # Optimizer hyper-parameters 
    parser.add_argument('--score_threshold', default=0., type=float,
                        help='Anomaly score threshold')
    parser.add_argument('--std_threshold', type=float, default=0.0785,
                        help='Standard deviation threshold')

    args = parser.parse_args()

    main(args)
