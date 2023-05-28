import torch
import argparse
import copy
from nflows.distributions import normal

from model.ae_flow_model import AE_Flow_Model
from model.Auto_encoder_seperate import AE_Model

# from dataloader import load, split_data, fold_to_loaders
from utils import optimize_threshold, calculate_metrics
from experiment import Experiment
import wandb
import numpy as np


from tqdm import tqdm
from anomalib.models.fastflow.torch_model import FastflowModel

# Make sure the following reads to a file with your own W&B API/Server key
WANDBKEY = open("wandbkey.txt", "r").read()

def train_step(epoch, model, train_loader,optimizer, anomalib_dataset=False, _print=False):
    model.train()
    if type(model) == FastflowModel : model.training = True 

    train_loss_epoch = 0
    for batch_idx, data in enumerate(train_loader):
        if anomalib_dataset:
            x = data['image']
        else:
            x = data[0]
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        original_x = x.to(device)

        optimizer.zero_grad(set_to_none=True)

        if type(model) == AE_Flow_Model:
            reconstructed_x = model(original_x).squeeze(dim=1)  

            recon_loss = model.get_reconstructionloss(original_x, reconstructed_x)
            flow_loss = model.get_flow_loss(bpd=True)
            wandb.log({'recon_loss (train)':recon_loss, 'flow loss (train)':flow_loss})
            loss = args.loss_alpha * flow_loss + (1-args.loss_alpha) * recon_loss

        if type(model) == AE_Model:
            reconstructed_x = model(original_x).squeeze(dim=1)  
            loss = model.get_reconstructionloss(original_x, reconstructed_x)
            wandb.log({'recon_loss (train)':loss})
        
        if type(model) == FastflowModel:
            out = model(original_x)

            # take [-1] as the fastflow model returns the z_prime of all layers, we only want to look at the last
            z_prime, log_jac_det = out[0][-1], out[1][-1]
            # delete the full model output to save memory
            del out
            log_z = normal.StandardNormal(shape=z_prime.shape[1:]).log_prob(z_prime)

            # again, only want the log_jac_det corresponding to the last layer
            log_p = log_z + log_jac_det[-1]
            loss = -log_p.mean()
            wandb.log({'flow loss (train)':loss})
        loss.backward()
        optimizer.step()
        train_loss_epoch += loss.item()
    wandb.log({'Train loss per epoch:': train_loss_epoch / len(train_loader)})
    if _print: print('====> Epoch {} : Average loss: {:.4f}'.format(epoch, train_loss_epoch / len(train_loader)))
    return model


@torch.no_grad()
def get_anomaly_scores(model, dataloader, anomalib_dataset):
    if type(model) == FastflowModel: model.training = False
    anomaly_scores, true_labels = [], []

    for batch_idx, data in enumerate(dataloader):
        if anomalib_dataset:
            x, y = data['image'], data['label']
        else:
            x,y = data[0], data[1] 
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        original_x, y = x.to(device), y.to(device)
        reconstructed_x = model(original_x).squeeze(dim=1)

        try:
            beta = args.loss_beta
        except NameError:
            beta = 0.9 # 

        if type(model) == FastflowModel: anomaly_score = torch.mean(reconstructed_x, axis=(1, 2))
        if type(model) == AE_Flow_Model: anomaly_score = model.get_anomaly_score(_beta=beta, original_x=original_x, reconstructed_x=reconstructed_x)
        if type(model) == AE_Model: anomaly_score = model.get_anomaly_score(original_x=original_x, reconstructed_x=reconstructed_x)

        anomaly_scores.append(anomaly_score)
        true_labels.append(y)
    true_labels = [item for sublist in [tensor.cpu().numpy() for tensor in true_labels] for item in sublist]
    anomaly_scores = [item for sublist in [tensor.cpu().numpy() for tensor in anomaly_scores] for item in sublist]
    return anomaly_scores, true_labels

@torch.no_grad()
def find_threshold(epoch, model, train_loader, verbose=False, anomalib_dataset=False):
    anomaly_scores, true_labels = get_anomaly_scores(model, train_loader, anomalib_dataset)
    if verbose: print(f"Now moving onto finding the appropriate threshold (based on training data including abnormal samples):")
    optimal_threshold = optimize_threshold(anomaly_scores, true_labels)
    wandb.log({'optimal (selection) threshold': optimal_threshold})
    if verbose: print(f"Optimal threshold: {optimal_threshold}")
    return optimal_threshold

@torch.no_grad()
def eval_model(epoch, model, data_loader, threshold=None, _print=False, return_only_anomaly_scores=False, track_results=True, test_eval=False, anomalib_dataset=False, running_ue_experiments=False):
    
    anomaly_scores, true = get_anomaly_scores(model, data_loader, anomalib_dataset)
    if return_only_anomaly_scores: return true, anomaly_scores

    results = calculate_metrics(true, anomaly_scores, threshold, _print=_print)
    if test_eval and track_results:
        test_results = {}
        for metric in results: test_results[metric+'-test'] = results[metric]
        wandb.log(test_results) 

    # the validation set contains very few samples, only log the results on the test set (per 10 epochs), this does not influence the best model selection
    elif track_results: wandb.log(results)
    if _print: print(f"Epoch {epoch}: {results}")
    return results

def model_checkpoint(epoch, model, threshold_loader_all, checkpoint_loader, current_best_score, used_thr, best_model, verbose=False, anomalib_dataset=False):
    if (epoch % 5 == 0) and (epoch != 0):
        threshold = find_threshold(epoch, model, threshold_loader_all, verbose=False, anomalib_dataset=anomalib_dataset)
        results = eval_model(epoch, model, checkpoint_loader, threshold, _print=True, anomalib_dataset=anomalib_dataset)
        if verbose: print(f"Running model checkpoint using threshold_loader_all and checkpoint_loader, F1 score now is: {results['F1']}")
        if results['F1'] > current_best_score:
            current_best_score = results['F1']
            if verbose: print(f"Found new best: {current_best_score}")
            used_thr = threshold
            best_model = copy.deepcopy(model)
    return used_thr, best_model, current_best_score

def main(args):
    """
    """

    # For our final experiments we want both of these to be set to True for full reproducibility
    torch.backends.cudnn.deterministic = args.fully_deterministic
    torch.backends.cudnn.benchmark = args.torch_benchmark

    # In the paper (Section 3.2) the authors mention other hyperparameters for the chest-xray set, so we enforce it:
    if args.dataset == "chest_xray": args.optim_weight_decay, args.optim_lr = 0.0, 1e-3
    if args.dataset == "chest_xray": args.subnet_arc = 'convnet_like'
    if args.model == 'autoencoder': args.loss_alpha, args.loss_beta = 0,0

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
    'n_validation_folds': args.n_validation_folds}
    )
    torch.manual_seed(args.seed) # Setting the seed
    print('running with seed: ', args.seed)

    experiment = Experiment(args)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # this outer loop is relevant for btad and mvtec, where we train seperate models for each class within the dataset, and average after
    for subset in experiment.subsets:
        experiment.initialize_model(subset)
   
        # Loading the data in a splitted way for later use, see the blogpost, discarding the validation set due to it's limited size
        # NOTE: for MVTEC or BTECH the train_abnormal loader will be a validation loader
        model = experiment.model.to(device)
        #for _, param in model.named_parameters(): param.requires_grad=True
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.optim_lr, weight_decay=args.optim_weight_decay, betas=(args.optim_momentum, 0.999))
        current_best_score, used_thr, best_model = 0.0, 0.0, None

        experiment.load_data()

        # Training loop
        metrics_per_fold = []
        for fold in tqdm(range(args.n_validation_folds)):
            train_normal_loader, threshold_loader, validate_loader_combined = experiment.load_fold_data(fold)
            for epoch in range(1, args.epochs+1):
                train_step(epoch, model, train_normal_loader,optimizer, experiment.anomalib_dataset)
                
                used_thr, best_model, current_best_score = model_checkpoint(epoch, model, experiment.threshold_loader_all, experiment.checkpoint_loader, current_best_score, used_thr, best_model, verbose=True, anomalib_dataset=experiment.anomalib_dataset)

            ## After every fold we use the validate-loader 
            threshold = find_threshold(epoch, model, threshold_loader, anomalib_dataset=experiment.anomalib_dataset)
            results = eval_model(epoch, model, validate_loader_combined, threshold, anomalib_dataset=experiment.anomalib_dataset)
            print(f"After fold {fold} results on validate_loader_combined: {results}")
            metrics_per_fold.append(results['F1'])

            ## After every fold also run quickly test analysis:
            threshold = find_threshold(epoch, best_model, experiment.threshold_loader_all, verbose=False, anomalib_dataset=experiment.anomalib_dataset)
            final_results = eval_model(epoch, best_model, experiment.test_loader, threshold, _print=True, anomalib_dataset=experiment.anomalib_dataset)
            print(f"After fold {fold}, performance on test set is the following: {final_results}")

        print(f"F1 scores per fold: {metrics_per_fold}, mean={np.mean(metrics_per_fold)}")
        ## Only after all training we are interested in thresholding! Only part of inference not training
        threshold = find_threshold(epoch, best_model, experiment.threshold_loader_all, verbose=False, anomalib_dataset=experiment.anomalib_dataset)
        final_results = eval_model(epoch, best_model, experiment.test_loader, threshold, _print=True, anomalib_dataset=experiment.anomalib_dataset)
        experiment.subset_results.append(final_results)
    
    torch.save(model.state_dict(), f'models/seed/{args.seed}.pth')

    # for datasets with multiple classes
    result_metrics = final_results.keys()
    final_results = {key: np.mean([d[key] for d in experiment.subset_results]) for key in result_metrics}
    print(final_results)

    wandb.log(final_results) 

    print(f"Final, best results on test dataset: {final_results}")
    print(f'Final results using last model: {results}')
    
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
                             'To have a truly deterministic run, this has to be 0.')

    args = parser.parse_args()
    main(args)
