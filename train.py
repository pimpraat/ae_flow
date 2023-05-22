import torch
import torchmetrics
import argparse
import copy
import os
import torch.utils.data as data
import torch.nn as nn
from nflows.distributions import normal

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
from anomalib.models.fastflow.torch_model import FastflowModel
from anomalib.models.ganomaly.torch_model import GanomalyModel
import matplotlib.pyplot as plt 

# Make sure the following reads to a file with your own W&B API/Server key
WANDBKEY = open("wandbkey.txt", "r").read()

def train_fastflow_step(model, dataloader, optimizer, device, anomalib_dataset=False):
    model.train()
    train_loss_epoch = 0

    for batch_idx, data in enumerate(dataloader):
    #for batch_idx, (x, y) in enumerate(dataloader):
        if anomalib_dataset:
            x = data['image']
            y = data['label']
        else:
            x = data[0]
            y = data[1]    

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        original_x = x.to(device)#data['image'].to(device)

        optimizer.zero_grad(set_to_none=True)
        z_prime, log_jac_det = model(original_x)

        # take [-1] as the fastflow model returns the z_prime of all layers, we only want to look at the last
        log_z = normal.StandardNormal(shape=z_prime[-1].shape[1:]).log_prob(z_prime[-1])
        
        # again, only want the log_jac_det corresponding to the last layer
        log_p = log_z + log_jac_det[-1]

        flow_loss = -log_p.mean()

        #flow_loss = model.get_flow_loss(bpd=True)
        wandb.log({'flow loss (train)':flow_loss})
        flow_loss.backward()
        optimizer.step()
        train_loss_epoch += flow_loss.item()
    wandb.log({'Train loss per epoch:': train_loss_epoch / len(dataloader)})

def train_ganomaly_step(model, dataloader, optimizer, device, anomalib_dataset=False):
    model.train()
    train_loss_epoch = 0

    for batch_idx, data in enumerate(dataloader):
    #for batch_idx, (x, y) in enumerate(dataloader):
        if anomalib_dataset:
            x = data['image']
            y = data['label']
        else:
            x = data[0]
            y = data[1]    

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        original_x = x.to(device)#data['image'].to(device)

        optimizer.zero_grad(set_to_none=True)
        
        # fake is the reconstruction
        padded_batch, fake, latent_i, latent_o = model(original_x)
        recon_loss = nn.functional.mse_loss(padded_batch, fake)

        wandb.log({'recon loss (train)':recon_loss})
        recon_loss.backward()
        optimizer.step()
        train_loss_epoch += recon_loss.item()
    wandb.log({'Train loss per epoch:': train_loss_epoch / len(dataloader)})

def train_step(epoch, model, train_loader,
                  optimizer, anomalib_dataset=False, _print=False):

    model.train()
    train_loss_epoch = 0

    for batch_idx, data in enumerate(train_loader):
        if anomalib_dataset:
            x = data['image']
            y = data['label']
        else:
            x = data[0]
            y = data[1]   
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        original_x = x.to(device)

        optimizer.zero_grad(set_to_none=True)
        reconstructed_x = model(original_x).squeeze(dim=1)  

        recon_loss = model.get_reconstructionloss(original_x, reconstructed_x)
        flow_loss = model.get_flow_loss(bpd=True)
        wandb.log({'recon_loss (train)':recon_loss, 'flow loss (train)':flow_loss})

        loss = args.loss_alpha * flow_loss + (1-args.loss_alpha) * recon_loss

        loss.backward()
        optimizer.step()
        train_loss_epoch += loss.item()
    wandb.log({'Train loss per epoch:': train_loss_epoch / len(train_loader)})
    if _print: print('====> Epoch {} : Average loss: {:.4f}'.format(epoch, train_loss_epoch / len(train_loader)))

@torch.no_grad()
def find_threshold(epoch, model, train_loader, _print=False, baseline=False, anomalib_dataset=False):
    anomaly_scores, true_labels = [], []
    model.training = False
    # e.g. fastflow
    if baseline:
        for batch_idx, data in enumerate(train_loader):
        #for batch_idx, (x, y) in enumerate(train_loader):
            if anomalib_dataset:
                x = data['image']
                y = data['label']
            else:
                x = data[0]
                y = data[1]    

            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            x, y = x.to(device), y.to(device)

            if type(model) == FastflowModel:
                anomaly_mapping = model(x).squeeze(dim=1)

                # mean over all dimensions, but the batch dimension
                # this gives an average anomaly score for each image
                anomaly_score = torch.mean(anomaly_mapping, axis=(1, 2))
            
            # TODO: is this the appropriate way to retreive the anomaly score? 
            # https://github.com/openvinotoolkit/anomalib/blob/main/src/anomalib/models/ganomaly/torch_model.py
            elif type(model) == GanomalyModel:
                anomaly_score = model(x)

            anomaly_scores.append(anomaly_score)
            true_labels.append(y)
    else:
        #for batch_idx, (x, y) in enumerate(train_loader):
        for batch_idx, data in enumerate(train_loader):
            if anomalib_dataset:
                x = data['image']
                y = data['label']
            else:
                x = data[0]
                y = data[1]    
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            original_x = x.to(device)
            reconstructed_x = model(original_x).squeeze(dim=1)
            anomaly_score = model.get_anomaly_score(_beta=args.loss_beta, 
                                                        original_x=original_x, reconstructed_x=reconstructed_x)
            

            anomaly_scores.append(anomaly_score)
            true_labels.append(y)

    wandb.log({'std anomaly_score of all (training) samples':torch.std(anomaly_score)})
    if _print: print(f"Now moving onto finding the appropriate threshold (based on training data including abnormal samples):")
    optimal_threshold = optimize_threshold(anomaly_scores, true_labels)
    wandb.log({'optimal (selection) threshold': optimal_threshold})
    if _print: print(f"Optimal threshold: {optimal_threshold}")
    return optimal_threshold

def calculate_metrics(true, anomaly_scores, threshold, _print=False):
    pred = np.array(anomaly_scores>threshold, dtype=int)
    if _print: print(f"Number of predicted anomalies in the (test-)set: {np.sum(pred)}")
    
    tn, fp, fn, tp = confusion_matrix(true, pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(true, pred)

    results = {
        'AUC': auc(fpr, tpr),
        'ACC' : (tp + tn)/(tp+fp+fn+tn),
        'SEN' : tp / (tp+fn),
        'SPE' : tn / (tn + fp),
        'F1' : f1_score(true, pred, average='binary')
    }
    return results

@torch.no_grad()
def eval_model(epoch, model, data_loader, threshold=None, _print=False, return_only_anomaly_scores=False, track_results=True, test_eval=False, baseline=False, anomalib_dataset=False):
    
    anomaly_scores, true_labels = [], []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if baseline:
        model.training = False

        for batch_idx, data in enumerate(data_loader):
            if anomalib_dataset:
                x = data['image']
                y = data['label']
            else:
                x = data[0]
                y = data[1]    
            x, y = x.to(device), y.to(device)

            if type(model) == FastflowModel:
                anomaly_mapping = model(x).squeeze(dim=1)

                # try sum
                anomaly_score = torch.mean(anomaly_mapping, axis=(1, 2))
            # TODO: is this the appropriate way to retreive the anomaly score? 
            # https://github.com/openvinotoolkit/anomalib/blob/main/src/anomalib/models/ganomaly/torch_model.py
            elif type(model) == GanomalyModel:
                anomaly_score = model(x)

            anomaly_scores.append(anomaly_score)
            true_labels.append(y)
    else:
        for batch_idx, data in enumerate(data_loader):
            if anomalib_dataset:
                x = data['image']
                y = data['label']
            else:
                x = data[0]
                y = data[1]    
            original_x,y = x.to(device), y.to(device)
            reconstructed_x = model(original_x).squeeze(dim=1)
            
            anomaly_score = model.get_anomaly_score(_beta=args.loss_beta, 
                                                        original_x=original_x, reconstructed_x=reconstructed_x)
            anomaly_scores.append(anomaly_score)

            true_labels.append(y)

    true = [item for sublist in [tensor.cpu().numpy() for tensor in true_labels] for item in sublist]
    anomaly_scores = [item for sublist in [tensor.cpu().numpy() for tensor in anomaly_scores] for item in sublist]

    if return_only_anomaly_scores: return true, anomaly_scores


    results = calculate_metrics(true, anomaly_scores, threshold, _print=_print)
    if test_eval and track_results:
        test_results = {}
        for metric in results:
            test_results[metric+'-test'] = results[metric]
        wandb.log(test_results) 

    # the validation set contains very few samples
    # only log the results on the test set (per 10 epochs)
        # this does not influence the best model selection
    elif track_results: wandb.log(results)
    if _print: print(f"Epoch {epoch}: {results}")
    return results

def main(args):
    """
    """

    # For our final experiments we want both of these to be set to True for full reproducibility
    torch.backends.cudnn.deterministic = args.fully_deterministic
    torch.backends.cudnn.benchmark = args.torch_benchmark

    # In the paper (Section 3.2) the authors mention other hyperparameters for the chest-xray set, so we enforce it:
    if args.dataset == "chest_xray": args.optim_weight_decay, args.optim_lr = 0.0, 1e-3


    wandb.login(key=WANDBKEY)
    wandb.init(
    project=str(f"ae_flow{'_final_experiments' if args.final_experiments else ''}"),
    config={
    "model": args.model,
    "subnet_arc": args.subnet_architecture,
    "n_flowblocks": args.n_flowblocks,
    "dataset": args.dataset,
    "epochs": args.epochs,
    'loss_alpha': args.loss_alpha,
    'loss_beta': args.loss_beta,
    'optim_lr': args.optim_lr,
    'optim_momentum': args.optim_momentum,
    'optim_weight_decay': args.optim_weight_decay,
    'n_validation_folds': args.n_validation_folds
    }
    )
    torch.manual_seed(args.seed) # Setting the seed

    # rather than training the entire model on all of the different subsets, we train and evaluate the model seperately on each subset
    if args.dataset == 'btech':
        subsets = ['01', '02', '03']
        anomalib_dataset = True
    elif args.dataset == 'mvtec':
        subsets = ['pill', 'toothbrush', 'wood', 'grid', 'capsule', 'transistor', 'screw', 'carpet', 'cable', 'bottle', 'tile', 'metal_nut', 'hazelnut', 'leather', 'zipper']
        anomalib_dataset = True
    else:
        subsets = [None]
        anomalib_dataset = False

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    baseline = False
    # this outer loop is relevant for btad and mvtec, where we train seperate models for each class within the dataset, and average after
    for subset in subsets:
        if subset != None:
            print(f'Running on subset: {subset}')
        subset_results = []

        # Selecting the correct model with it's model settings specified in the experiments:
        if args.model == 'ae_flow': model = AE_Flow_Model(subnet_architecture=args.subnet_architecture, n_flowblocks=args.n_flowblocks)
        elif args.model == 'fastflow':
            model = FastflowModel(input_size=(256, 256), backbone="wide_resnet50_2", flow_steps=8, pre_trained=False)
            model.training = True
            baseline = True

        # TODO: are these the correct parameters?
        elif args.model == 'ganomaly':
            model = GanomalyModel(input_size=(256, 256), num_input_channels=3, n_features=1, latent_vec_size=1024)
            model.training = True
            baseline = True

        # Loading the data in a splitted way for later use, see the blogpost, discarding the validation set due to it's limited size
        # NOTE: for MVTEC or BTECH the train_abnormal loader will be a validation loader
        train_loader, train_abnormal, test_loader = load(data_dir=args.dataset,batch_size=args.batch_size, num_workers=args.num_workers, subset=subset, anomalib_dataset=anomalib_dataset)
   

        model = model.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.optim_lr, weight_decay=args.optim_weight_decay, betas=(args.optim_momentum, 0.999))
        current_best_score, used_thr = 0.0, 0.0
        best_model = None
        
        # Training loop
        for epoch in range(args.epochs):
            fold_metrics = []

            # Splitting all folds:
            kfold_normal, kfold_abormal = KFold(n_splits=args.n_validation_folds, shuffle=True), KFold(n_splits=args.n_validation_folds, shuffle=True)
            normal_split, abnormal_split = list(kfold_normal.split(train_loader)), list(kfold_abormal.split(train_abnormal))
            train_split_normal = [kfold[0] for kfold in normal_split]
            test_split_normal = [kfold[1] for kfold in normal_split]
            train_split_abnormal = [kfold[0] for kfold in abnormal_split]
            test_split_abnormal = [kfold[1] for kfold in abnormal_split]
            start = time.time()
            for fold in tqdm(range(args.n_validation_folds)):
                if args.model in ['fastflow', 'ganomaly']:
                    model.training = True

                train_ids_normal = train_split_normal[fold]
                train_ids_abnormal = train_split_abnormal[fold]
                test_ids_normal = test_split_normal[fold]
                test_ids_abnormal = test_split_abnormal[fold]

                train_normal_dataset = torch.utils.data.dataset.Subset(train_loader,train_ids_normal)
                train_normal_loader = data.DataLoader(train_normal_dataset, num_workers = args.num_workers, shuffle=True, batch_size=args.batch_size)
                train_abnormal_dataset =  torch.utils.data.dataset.Subset(train_abnormal,train_ids_abnormal)
                threshold_dataset = torch.utils.data.ConcatDataset([train_abnormal_dataset, train_normal_dataset])
                threshold_loader = data.DataLoader(threshold_dataset, num_workers = args.num_workers, batch_size=args.batch_size)
                
                # Performing the training step on just the normal samples:
                if args.model == 'fastflow':
                    train_fastflow_step(model, train_normal_loader, optimizer, device, anomalib_dataset=anomalib_dataset)
                elif args.model == 'ganomaly':
                    train_ganomaly_step(model, train_normal_loader, optimizer, device, anomalib_dataset=anomalib_dataset)
                else:
                    train_step(epoch, model, train_normal_loader, optimizer, anomalib_dataset=anomalib_dataset)

                threshold = find_threshold(epoch, model, threshold_loader, _print=False, baseline=baseline, anomalib_dataset=anomalib_dataset)

                validate_loader_normal = torch.utils.data.dataset.Subset(train_loader,test_ids_normal)
                validate_loader_abnormal = torch.utils.data.dataset.Subset(train_abnormal,test_ids_abnormal)
                validate_loader_combined = torch.utils.data.ConcatDataset([validate_loader_normal, validate_loader_abnormal])
                validate_loader_combined = data.DataLoader(validate_loader_combined, num_workers = args.num_workers, batch_size=args.batch_size)
                
                results = eval_model(epoch, model, validate_loader_combined, threshold, _print=False, baseline=baseline, anomalib_dataset=anomalib_dataset)
                fold_metrics.append(results['F1'])

            # Save reconstruction resuls every epoch for later analysis:
            # if args.model == 'ae_flow': wandb.log(sample_images(model, device))
            print(f"Duration for epoch {epoch}: {time.time() - start}")
            wandb.log({'time per epoch': time.time() - start})
        
            # Save if best evaluation according to the cross validation:
            # dont save if the model is subset specific
            if (np.mean(fold_metrics) >= current_best_score):
                current_best_score = np.mean(results['F1'])
                
                best_model = copy.deepcopy(model)
                used_thr = threshold

                if subset == None:
                    torch.save(model.state_dict(), str(f"models/{wandb.config}.pt"))
                else:
                    torch.save(model.state_dict(), str(f"models/{wandb.config}_{subset}.pt"))


            track_test_performance = True
            if (epoch % 10 == 0) and (epoch != 0) and (track_test_performance == True):
                print(f'Results on test set after {epoch} epochs')
                eval_model(epoch, best_model, test_loader, used_thr, _print=True, track_results=True, test_eval=True, baseline=baseline, anomalib_dataset=anomalib_dataset)
        results = eval_model(epoch, best_model, test_loader, threshold=used_thr, _print=True, track_results=True, test_eval=True, baseline=baseline, anomalib_dataset=anomalib_dataset)
        if subset != None:
            results = {key+f'-{subset}': results[key] for key in results}
        wandb.log(results)
        subset_results.append(results)
    
    # average the subset results for the btad and mvtec datasets
    if subsets != [None]:
        result_metrics = results.keys()
        final_results = {key+'-subsets-avg': np.mean([d[key] for d in subset_results]) for key in result_metrics}
    # otherwise, the last results represent our final results on the test set
    else:
        final_results = results

    wandb.log(final_results) 

    print(f"Final, best results on test dataset: {final_results}")
    
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
                        help='Which subflow architecture to use when using the ae_flow model: conv_like or resnet_like')

    # Other hyper-parameters
    parser.add_argument('--data_dir', default='../data/', type=str,
                        help='Directory where to look for the data. For jobs on Lisa, this should be $TMPDIR.')

    parser.add_argument('--final_experiments', default=False, type=bool, help='Whether to save results as for final experiments')

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
                             'To have a truly deterministic run, this has to be 0.')

    args = parser.parse_args()
    main(args)
