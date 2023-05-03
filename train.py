import torch
import torchmetrics
import argparse
torch.manual_seed(42) # Setting the seed

from model.ae_flow_model import AE_Flow_Model
from dataloader import load
from model.flow import FlowModule
from model.encoder import Encoder
from model.utils import optimize_threshold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, auc, roc_curve

import time

def train_step(epoch, model, train_loader,
                  optimizer):

    model.train()
    train_loss_epoch = 0

    for batch_idx, (x, _) in enumerate(train_loader):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        original_x = x.to(device)

        optimizer.zero_grad()
        # print(f"Shape of input-batch in train function: {original_x.shape}")
        reconstructed_x = model(original_x).squeeze(dim=1)

        # print(f"x shape in loss: {original_x.shape}")
        # print(f"recon_x shape in loss: {reconstructed_x.shape}")
        recon_loss = model.get_reconstructionloss(original_x, reconstructed_x)
        flow_loss = model.get_flow_loss()
        print(f"Recon loss: {recon_loss}, Flow loss: {flow_loss}")

        loss = args.loss_alpha * flow_loss + (1-args.loss_alpha) * recon_loss

        loss.backward()
        optimizer.step()
        train_loss_epoch += loss.item()

    print('====> Epoch {} : Average loss: {:.4f}'.format(epoch, train_loss_epoch / len(train_loader)))

def find_threshold(epoch, model, train_loader, _print=False):
    with torch.no_grad(): # Deactivate gradients for the following code
        anomaly_scores, true_labels = [], []

        for batch_idx, (x, y) in enumerate(train_loader):
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            original_x = x.to(device)
            y = y.to(device)

            reconstructed_x = model(original_x).squeeze(dim=1)
            anomaly_score = model.get_anomaly_score(_beta=args.loss_beta, 
                                                     original_x=original_x, reconstructed_x=reconstructed_x)
            anomaly_scores.append(anomaly_score)
            true_labels.append(y)

    print(f"Now moving onto finding the appropriate threshold (based on training data):")
    optimal_threshold = optimize_threshold(anomaly_scores, true_labels)
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
            # print(f"Shape of orignal/reconstructed: {original_x.shape, reconstructed_x.shape}")
            anomaly_score = model.get_anomaly_score(_beta=args.loss_beta, 
                                                     original_x=original_x, reconstructed_x=reconstructed_x)
            # print(f"Shape of anomaly scores: {anomaly_score}")
            anomaly_scores.append(anomaly_score)
            true_labels.append(y)

    true = true_labels[0].cpu()
    pred = [x >= threshold for x in anomaly_scores][0].cpu()

    tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
    fpr, tpr, thresholds = roc_curve(true, pred)

    results['AUC'] = auc(fpr, tpr)
    results['ACC'] = (tp + tn)/(tp+fp+fn+tn)
    results['SEN'] = tp / (tp+fn)
    results['SPE'] = tn / (tn + fp)
    results['F1'] = f1_score(true, pred)

    if _print: print(f"Epoch {epoch}: {results}")   
    return results

def main(args):
    """
    """

    train_loader, test_loader, _ = load(data_dir='data/chest_xray/',batch_size=64, num_workers=3)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(len(train_loader))
    # assert(False)
    # load_data(dataset=args.dataset, root=args.data_dir,
                        #  batch_size=args.batch_size,
                        #  num_workers=args.num_workers)

    # args.cuda = not args.no_cuda and torch.cuda.is_available()
    # device = torch.device("cuda:0" if args.cuda else "cpu")

    # Create model and push to the device
    model = AE_Flow_Model()
    model = model.to(device)

    # train_loader, test_loader = train_loader.to(device), test_loader.to(device)
# 
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.optim_lr, weight_decay=args.optim_weight_decay, )
    
    # Training loop
    # print(f"Using device {device}")
    for epoch in range(args.epochs):
        start = time.time()
        # Training epoch
        train_step(epoch, model, train_loader,
                  optimizer)

        if epoch % 5 == 0:
            threshold = find_threshold(epoch, model, train_loader, _print=True)

            eval_model(epoch, model, test_loader, threshold, _print=True)

        print(f"Duration for epoch {epoch}: {time.time() - start}")
    


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='AE Normalized Flow')

    # Optimizer hyper-parameters 
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size to use for training') ##TODO: Paper uses 128
    parser.add_argument('--loss_alpha', type=float, default=0.5,
                        help='')
    parser.add_argument('--loss_beta', type=float, default=0.9,
                        help='')
    parser.add_argument('--optim_lr', type=float, default=2e-4,
                        help='')
    parser.add_argument('--optim_momentum', type=float, default=0.9, ## still needs to be set in the Optimizer!
                        help='')
    parser.add_argument('--optim_weight_decay', type=float, default=1e-5,
                        help='')
    
    parser.add_argument('--dataset',default='OCT', type=str, help='Which dataset to run. Choose from: [OCT, XRAY, ISIC, BRATS, MIIC]')

    # Other hyper-parameters
    parser.add_argument('--data_dir', default='../data/', type=str,
                        help='Directory where to look for the data. For jobs on Lisa, this should be $TMPDIR.')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to use in the data loaders.' +
                             'To have a truly deterministic run, this has to be 0.')

    args = parser.parse_args()
    main(args)
