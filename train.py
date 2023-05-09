import torch
import torchmetrics
import argparse
torch.manual_seed(42) # Setting the seed

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


def eval_model(epoch, model, test_loader, threshold, _print=False):
    results = {}

    with torch.no_grad(): # Deactivate gradients for the following code
        anomaly_scores, true_labels = [], []

        for batch_idx, (x, y) in enumerate(test_loader):
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
    pred = [x >= threshold for x in anomaly_scores]
    print(f"Number of predicted anomalies in the test-set: {np.sum(pred)}")

    tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
    fpr, tpr, thresholds = roc_curve(true, pred)

    results['AUC'] = auc(fpr, tpr)
    results['ACC'] = (tp + tn)/(tp+fp+fn+tn)
    results['SEN'] = tp / (tp+fn)
    results['SPE'] = tn / (tn + fp)
    results['F1'] = f1_score(true, pred)

    wandb.log(results)

    if _print: print(f"Epoch {epoch}: {results}")   
    return results

def main(args):
    """
    """

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    #TODO: Make private!
    wandb.login(key='10f35d76229e73f4650338d78de2b411d51fa3ae')

    wandb.init(
    # set the wandb project where this run will be logged
    project="ae_flow",
    
    # track hyperparameters and run metadata
    #TODO: Update these to be in line with the arguments
    config={
    "model": args.model,
    "subnet_arc": args.subnet_architecture,
    "dataset": args.dataset,
    "epochs": args.epochs
    }
)

    train_loader, train_complete, validate_loader, test_loader = load(data_dir="chest_xray",batch_size=64, num_workers=3)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(f"Length of the train loader: {len(train_loader)} given a batch size of {args.batch_size}")
    
    # Create model and push to the device
    if args.model == 'ae_flow': model = AE_Flow_Model()
    # if args.model == 'ganomaly': model = GanomalyModel(input_size=(256,256), latent_vec_size=100, num_input_channels=3, n_features=None)

    model = model.to(device)

    # Save validation images to the model for later on:
    # im_normal, _ = next(iter(validate_loader[0]))
    # im_abnormal, _ = next(iter(validate_loader[1]))
    # model.sample_images_normal = im_normal[:3]
    # model.sample_images_abnormal = im_abnormal[:3]

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.optim_lr, weight_decay=args.optim_weight_decay, )
    
    # Training loop
    for epoch in range(args.epochs):
        start = time.time()

        train_step(epoch, model, train_loader,
                  optimizer)
        
        # TRAIN_COMPLETE

        threshold = find_threshold(epoch, model, train_complete, _print=True)
        eval_model(epoch, model, test_loader, threshold, _print=True)

            # Todo: fix again that images as being pushed to w&b
           # if args.model == 'ae_flow': wandb.log(sample_images(model, device))


        print(f"Duration for epoch {epoch}: {time.time() - start}")
        wandb.log({'time per epoch': time.time() - start})
    
    # Save the current model (only after training is fully done right now):
    # TODO: Only save best model according to validation loss loop
    wandb.save(model.state_dict())

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
    parser.add_argument('--optim_lr', type=float, default=2e-4,
                        help='')
    parser.add_argument('--optim_momentum', type=float, default=0.9, 
                        help='')
    parser.add_argument('--optim_weight_decay', type=float, default=1e-5,
                        help='')
    parser.add_argument('--dataset',default='chest_xray', type=str, help='Which dataset to run. Choose from: [OCT2017, chest_xray, ISIC, BRATS, MIIC]')
    parser.add_argument('--model',default='ae_flow', type=str, help='Which dataset to run. Choose from: [autoencoder, fastflow, ae_flow]')

    parser.add_argument('--subnet_architecture', default='subnet', type=str,
                        help='Which subflow architecture to use when using the ae_flow model: subnet or resnet_like')

    # Other hyper-parameters
    parser.add_argument('--data_dir', default='../data/', type=str,
                        help='Directory where to look for the data. For jobs on Lisa, this should be $TMPDIR.')
    parser.add_argument('--epochs', default=40, type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to use in the data loaders.' +
                             'To have a truly deterministic run, this has to be 0.')

    args = parser.parse_args()
    main(args)
