import torch
import torchmetrics
import argparse
import copy
torch.manual_seed(42) # Setting the seed
import copy

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



# Make sure the following reads to a file with your own W&B API/Server key
WANDBKEY = open("wandbkey.txt", "r").read()


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


def eval_model(epoch, model, data_loader, threshold=None, _print=False, return_only_anomaly_scores=False, validation=True):

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
    
    # the validation set contains very few samples
    # only log the results on the test set (per 10 epochs)
        # this does not influence the best model selection
    if validation == False:
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
        "dataset": args.dataset,
        "epochs": args.epochs,
        'loss_alpha': args.loss_alpha,
        'loss_beta': args.loss_beta,
        'optim_lr': args.optim_lr,
        'optim_momentum': args.optim_momentum,
        'optim_weight_decay': args.optim_weight_decay
        }
    )

    train_loader, train_complete, validate_loader, test_loader = load(data_dir=args.dataset,batch_size=args.batch_size, num_workers=3)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(f"Length of the train loader: {len(train_loader)} given a batch size of {args.batch_size}")
    
    # Create model and push tvco the device
    if args.model == 'ae_flow': model = AE_Flow_Model(args.subnet_architecture)
    # if args.model == 'ganomaly': model = GanomalyModel(input_size=(256,256), latent_vec_size=100, num_input_channels=3, n_features=None)
        
        

    model = model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.optim_lr, weight_decay=args.optim_weight_decay, betas=(args.optim_momentum, 0.999))
    current_best_score, used_thr = 0.0, 0.0
    best_model = None
    
    # Training loop
    for epoch in range(args.epochs):
        start = time.time()

        train_step(epoch, model, train_loader,
                  optimizer)

        # If we calculate the threshold externally (removed from Lisa), 
        # we need to save at every epoch the anomaly scores for both train_complete and test_loader
        if args.find_threshold_externally:
            true_label_traincomplete, anomaly_score_traincomplete = eval_model(epoch, model, train_complete, threshold=used_thr, return_only_anomaly_scores=True)
            true_label_test, anomaly_score_test = eval_model(epoch, model, test_loader, threshold=used_thr, return_only_anomaly_scores=True)

            data = {'0': {
                        'true_label_traincomplete': true_label_traincomplete,
                        'anomaly_score_traincomplete': anomaly_score_traincomplete,
                        'true_label_test': true_label_test,
                        'anomaly_score_test': anomaly_score_test}}
            
            with open(f"{str({wandb.config})}.json", "a") as outfile: json.dump(data, outfile)
            if epoch % 10 == 0: torch.save(model.state_dict(), str(f"models/{wandb.config}_at_epoch_{epoch}.pt"))
            continue


        threshold = find_threshold(epoch, model, train_complete, _print=True)
        results = eval_model(epoch, model, validate_loader, threshold, _print=True)

        # Todo: fix again that images as being pushed to w&b
        # if args.model == 'ae_flow': wandb.log(sample_images(model, device))


        print(f"Duration for epoch {epoch}: {time.time() - start}")
        wandb.log({'time per epoch': time.time() - start})
        
        title = f"{args.model}_{args.subnet_arc}_{args.dataset}_a{args.loss_alpha}_b{args.loss_beta}_lr{args.optim_lr}_m{args.optim_momentum}_wd{args.optim_weight_decay}"

        # Save if best eval:
        if results['F1'] >= current_best_score:
            current_best_score = results['F1']
            torch.save(model.state_dict(), str(f"models/{title}.pt"))
            best_model = copy.deepcopy(model)
            used_thr = threshold

        if epoch % 10 == 0:
            print(f'Results on test set after {epoch} epochs')
            eval_model(epoch, best_model, test_loader, used_thr, _print=True, validation=False)
            # save model every 10 epoch
            
            torch.save(model.state_dict(), str(f'models/per_epoch/{title}.pt'))

    results = eval_model(epoch, best_model, test_loader, threshold=used_thr, _print=True, validation=False)
    
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


    parser.add_argument('--epochs', default=15, type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to use in the data loaders.' +
                             'To have a truly deterministic run, this has to be 0.')

    args = parser.parse_args()
    main(args)
