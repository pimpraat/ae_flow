import torch
import torchmetrics
import argparse
import copy
import os
torch.manual_seed(42) # Setting the seed
import torch.utils.data as data

from model.ae_flow_model import AE_Flow_Model
# from baselines.ganomaly import GanomalyModel
from dataloader import load
from model.flow import FlowModule
from model.encoder import Encoder
from model.utils import optimize_threshold, sample_images
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, auc, roc_curve
import wandb
import torchvision
import numpy as np
import sklearn
import time
import json
from sklearn.model_selection import KFold

from tqdm import tqdm

# Make sure the following reads to a file with your own W&B API/Server key
WANDBKEY = open("wandbkey.txt", "r").read()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

def train_step(epoch, model, train_loader,
                  optimizer):

    model.train()
    train_loss_epoch = 0

    for batch_idx, (x, _) in enumerate(train_loader):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        original_x = x.to(device)

        optimizer.zero_grad(set_to_none=True)
        reconstructed_x = model(original_x).squeeze(dim=1)  

        recon_loss = model.get_reconstructionloss(original_x, reconstructed_x)
        flow_loss = model.get_flow_loss(bpd=True)
        wandb.log({'recon_loss':recon_loss, 'flow loss':flow_loss})
        # print(f"recon_loss:{recon_loss}, 'flow loss':{flow_loss}")

        loss = args.loss_alpha * flow_loss + (1-args.loss_alpha) * recon_loss

        loss.backward()
        optimizer.step()
        train_loss_epoch += loss.item()
    wandb.log({'Train loss per epoch:': train_loss_epoch / len(train_loader)})
    print('====> Epoch {} : Average loss: {:.4f}'.format(epoch, train_loss_epoch / len(train_loader)))

@torch.no_grad()
def find_threshold(epoch, model, train_loader, _print=False):
    anomaly_scores, true_labels = [], []

    for batch_idx, (x, y) in enumerate(train_loader):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        original_x = x.to(device)

        reconstructed_x = model(original_x).squeeze(dim=1)
        anomaly_score = model.get_anomaly_score(_beta=args.loss_beta, 
                                                    original_x=original_x, reconstructed_x=reconstructed_x)
        

        anomaly_scores.append(anomaly_score)
        true_labels.append(y)

    wandb.log({'mean anomaly_score':torch.mean(anomaly_score)})

    print(f"Now moving onto finding the appropriate threshold (based on training data):")
    optimal_threshold = optimize_threshold(anomaly_scores, true_labels)
    wandb.log({'optimal threshold': optimal_threshold})
    print(f"Optimal threshold: {optimal_threshold}")
    return optimal_threshold

def calculate_metrics(true, anomaly_scores, threshold):
    results = {}
    pred = [x >= threshold for x in anomaly_scores]
    print(f"Number of predicted anomalies in the test-set: {np.sum(pred)}")
    
    tn, fp, fn, tp = confusion_matrix(true, pred, labels=[0, 1]).ravel()
    fpr, tpr, thresholds = roc_curve(true, pred)

    results['AUC'] = auc(fpr, tpr)
    results['ACC'] = (tp + tn)/(tp+fp+fn+tn)
    results['SEN'] = tp / (tp+fn)
    results['SPE'] = tn / (tn + fp)
    results['F1'] = f1_score(true, pred)

    return results


# maybe at epoch as an optional argument? 
def eval_model(epoch, model, data_loader, threshold=None, _print=False, return_only_anomaly_scores=False, track_results=True, test_eval=False):

    with torch.no_grad(): # Deactivate gradients for the following code
        anomaly_scores, true_labels = [], []

        for batch_idx, (x, y) in enumerate(data_loader):
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

            x,y = x.to(device), y.to(device)

            original_x = x
            reconstructed_x = model(original_x).squeeze(dim=1)
          
            anomaly_score = model.get_anomaly_score(_beta=args.loss_beta, 
                                                     original_x=original_x, reconstructed_x=reconstructed_x)

            anomaly_scores.append(anomaly_score)
            true_labels.append(y)

    # true = true_labels[0].cpu()
    true = [tensor.cpu().numpy() for tensor in true_labels]
    true = [item for sublist in true for item in sublist]


    anomaly_scores = [tensor.cpu().numpy() for tensor in anomaly_scores]
    anomaly_scores = [item for sublist in anomaly_scores for item in sublist]
    if return_only_anomaly_scores: return true, anomaly_scores

    results = calculate_metrics(true, anomaly_scores, threshold)
    
    if test_eval and track_results:
        test_results = {}
        for metric in results:
            test_results[metric+'-test'] = results[metric]
        wandb.log(test_results) 

    # the validation set contains very few samples
    # only log the results on the test set (per 10 epochs)
        # this does not influence the best model selection
    elif track_results == True:
        wandb.log(results)

    if _print: print(f"Epoch {epoch}: {results}")   
    return results

def main(args):
    """
    """

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    if args.dataset == "chest_xray": args.optim_weight_decay, args.optim_lr = 0.0, 1e-3


    #TODO: Make private!
    wandb.login(key=WANDBKEY)

    wandb.init(
    # set the wandb project where this run will be logged
    project="ae_flow",
    
    # track hyperparameters and run metadata
    #TODO: Update these to be in line with the arguments
    config={
    "model": args.model,
    "subnet_arc": args.subnet_architecture,
    "custom_computation_graph": args.custom_computation_graph,
    "n_flowblocks": args.n_flowblocks,
    "dataset": args.dataset,
    "epochs": args.epochs,
    'loss_alpha': args.loss_alpha,
    'loss_beta': args.loss_beta,
    'optim_lr': args.optim_lr,
    'optim_momentum': args.optim_momentum,
    'optim_weight_decay': args.optim_weight_decay
    }
)

    #TODO: Make args!
    k_folds = 5


    #TODO: @Andre: train_complete should be train_abnormal
    train_loader, train_abnormal, validate_loader, test_loader = load(data_dir=args.dataset,batch_size=args.batch_size, num_workers=args.num_workers, return_dataloaders=False)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(f"Length of the train loader: {len(train_loader)} given a batch size of {args.batch_size}")
    
    # Create model and push tvco the device
    if args.model == 'ae_flow': model = AE_Flow_Model(subnet_architecture=args.subnet_architecture, custom_comptutation_graph=args.custom_computation_graph, n_flowblocks=args.n_flowblocks)
    # if args.model == 'ganomaly': model = GanomalyModel(input_size=(256,256), latent_vec_size=100, num_input_channels=3, n_features=None)

    model = model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.optim_lr, weight_decay=args.optim_weight_decay, betas=(args.optim_momentum, 0.999))
    current_best_score, used_thr = 0.0, 0.0
    best_model = None
    
    # Training loop
    for epoch in range(args.epochs):
        fold_metrics = []

        kfold_normal = KFold(n_splits=k_folds, shuffle=True)
        kfold_abormal = KFold(n_splits=k_folds, shuffle=True)

        #train_split_normal, test_split_normal
        normal_split = list(kfold_normal.split(train_loader))
        train_split_normal = [kfold[0] for kfold in normal_split]
        test_split_normal = [kfold[1] for kfold in normal_split]


        abnormal_split = list(kfold_abormal.split(train_abnormal))
        train_split_abnormal = [kfold[0] for kfold in abnormal_split]
        test_split_abnormal = [kfold[1] for kfold in abnormal_split]

        for fold in tqdm(range(k_folds)):
            train_ids_normal = train_split_normal[fold]
            train_ids_abnormal = train_split_abnormal[fold]
            test_ids_normal = test_split_normal[fold]
            test_ids_abnormal = test_split_abnormal[fold]

            train_normal_dataset = torch.utils.data.dataset.Subset(train_loader,train_ids_normal)
            train_normal_loader = data.DataLoader(train_normal_dataset, num_workers = args.num_workers, batch_size=args.batch_size)
            train_step(epoch, model, train_normal_loader, optimizer)

            train_abnormal_dataset =  torch.utils.data.dataset.Subset(train_abnormal,train_ids_abnormal)
            threshold_dataset = torch.utils.data.ConcatDataset([train_abnormal_dataset, train_normal_dataset])
            threshold_loader = data.DataLoader(threshold_dataset, num_workers = args.num_workers, batch_size=args.batch_size)
            threshold = find_threshold(epoch, model, threshold_loader, _print=False)

            validate_loader_normal = torch.utils.data.dataset.Subset(train_loader,test_ids_normal)
            validate_loader_abnormal = torch.utils.data.dataset.Subset(train_abnormal,test_ids_abnormal)
            
            validate_loader_combined = torch.utils.data.ConcatDataset([validate_loader_normal, validate_loader_abnormal])

            validate_loader_combined = data.DataLoader(validate_loader_combined, num_workers = args.num_workers, batch_size=args.batch_size)
            
            if fold % 5 == 0:
                printeval=True
            else:
                printeval=False
            results = eval_model(epoch, model, validate_loader_combined, threshold, _print=printeval)
            fold_metrics.append(results['F1'])


        start = time.time()

        #train_step(epoch, model, train_loader,
        #          optimizer)

        # If we calculate the threshold externally (removed from Lisa), 
        # we need to save at every epoch the anomaly scores for both train_complete and test_loader
        # if args.find_threshold_externally:
        #     true_label_traincomplete, anomaly_score_traincomplete = eval_model(epoch, model, train_complete, threshold=used_thr, return_only_anomaly_scores=True)
        #     true_label_test, anomaly_score_test = eval_model(epoch, model, test_loader, threshold=used_thr, return_only_anomaly_scores=True)

        #     data = {'0': {
        #                 'true_label_traincomplete': true_label_traincomplete,
        #                 'anomaly_score_traincomplete': anomaly_score_traincomplete,
        #                 'true_label_test': true_label_test,
        #                 'anomaly_score_test': anomaly_score_test}}
            
        #     with open(f"{str({wandb.config})}.json", "a") as outfile: json.dump(data, outfile)
        #     if epoch % 10 == 0: torch.save(model.state_dict(), str(f"models/{wandb.config}_at_epoch_{epoch}.pt"))
        #     continue


        # threshold = find_threshold(epoch, model, train_complete, _print=True)
        # results = eval_model(epoch, model, validate_loader, threshold, _print=True)

        # Todo: fix again that images as being pushed to w&b
        # if args.model == 'ae_flow': wandb.log(sample_images(model, device))


        print(f"Duration for epoch {epoch}: {time.time() - start}")
        wandb.log({'time per epoch': time.time() - start})
    
        # Save if best eval:
        if np.mean(results['F1']) >= current_best_score:
            current_best_score = np.mean(results['F1'])
            torch.save(model.state_dict(), str(f"models/{wandb.config}.pt"))
            best_model = copy.deepcopy(model)
            used_thr = threshold

        if epoch % 10 == 0:
            print(f'Results on test set after {epoch} epochs')
            eval_model(epoch, best_model, test_loader, used_thr, _print=True, track_results=True, test_eval=True)
            # save model every 10 epoch
            if not args.custom_computation_graph:
                torch.save(model.state_dict(), str(f'models/per_epoch/{wandb.config}_epoch_{epoch}.pt'))

    results = eval_model(epoch, best_model, test_loader, threshold=used_thr, _print=True, track_results=True, test_eval=True)
    
    print(f"Final, best results on test dataset: {results}")
    
    wandb.finish()


if __name__ == '__main__':
    
    # Training settings
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
                        help='Which subflow architecture to use when using the ae_flow model: subnet or resnet_like')

    # Other hyper-parameters
    parser.add_argument('--data_dir', default='../data/', type=str,
                        help='Directory where to look for the data. For jobs on Lisa, this should be $TMPDIR.')
    parser.add_argument('--find_threshold_externally', default=False, type=bool,
                        help='')
    parser.add_argument('--externally_found_threshold', default=-1.0, type=float,
                        help='')
    parser.add_argument('--externally_found_threshold_epoch', default=-1.0, type=float,
                        help='')
    
    # new custom computation graph
    parser.add_argument('--custom_computation_graph', default=False, type=bool,
                        help='')
    parser.add_argument('--n_flowblocks', default=8, type=int, help='')


    parser.add_argument('--epochs', default=15, type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to use in the data loaders.' +
                             'To have a truly deterministic run, this has to be 0.')

    args = parser.parse_args()
    main(args)
