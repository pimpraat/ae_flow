import torch
import torchmetrics
import argparse
torch.manual_seed(42) # Setting the seed

from model.ae_flow_model import AE_Flow_Model
from dataloader import load


def train_step(epoch, model, train_loader,
                  optimizer):
    """
    Function for training an model on a dataset for a single epoch.
    Inputs:
        epoch - Current epoch
        model -  model to train
        train_loader - Data Loader for the dataset you want to train on
        optimizere - The optimizer used to update the parameters
    """
    model.train()
    train_loss = 0
    for batch_idx, (x, _) in enumerate(train_loader):
        original_x = x # .to(model.device)

        optimizer.zero_grad()
        # print(f"Shape of input-batch in train function: {original_x.shape}")
        reconstructed_x = model(original_x).squeeze(dim=1)


        # print(f"x shape in loss: {original_x.shape}")
        # print(f"recon_x shape in loss: {reconstructed_x.shape}")
        loss = model.get_reconstructionloss(original_x, reconstructed_x)
        loss.backward()
        optimizer.step()

        # print(f"Loss at epoch{epoch}: {loss}")

        train_loss += loss

    print('====> Epoch {} : Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader)))

#TODO: Finish implementation
# See: https://pytorch.org/torcheval/stable/metric_example.html
def eval_model(epoch, model, test_loader, _print=False):
    results = {}

    with torch.no_grad(): # Deactivate gradients for the following code
        pred, true = [], []
        for data_inputs, data_labels in test_loader:

            # data_inputs = data_inputs.to(args.device)
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)
            preds = torch.sigmoid(preds) # Sigmoid to map predictions between 0 and 1
            #TODO: Check this with the paper!
            pred_labels = (preds >= 0.5).long() # Binarize predictions to 0 and 1 (normal and anomaly)
            
            pred.append(pred_labels)
            true.append(data_labels)

    results['AUC'] = None
    results['F1'] = None
    results['ACC'] = torchmetrics.Accuracy(task='binary', average='none')(pred, true)
    results['SEN'] = None
    results['SPE'] = None

    if _print: print(f"Epoch {epoch}: {results}")
    return results

def main(args):
    """
    """

    train_loader, test_loader, _ = load(data_dir='data/chest_xray/',batch_size=64, num_workers=4)
    

    print(len(train_loader))
    # assert(False)
    # load_data(dataset=args.dataset, root=args.data_dir,
                        #  batch_size=args.batch_size,
                        #  num_workers=args.num_workers)

    # args.cuda = not args.no_cuda and torch.cuda.is_available()
    # device = torch.device("cuda:0" if args.cuda else "cpu")

    # Create model and push to the device
    model = AE_Flow_Model()
    # model = model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.optim_lr, weight_decay=args.optim_weight_decay, )
    
    # Training loop
    # print(f"Using device {device}")
    for epoch in range(args.epochs):
        # Training epoch
        train_step(epoch, model, train_loader,
                  optimizer)

        eval_model(epoch, model, test_loader, _print=True)


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
