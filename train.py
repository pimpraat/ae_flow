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
import wandb
from torchvision.utils import make_grid, save_image
import torchvision

import time

def train_step(epoch, model, train_loader,
                  optimizer):

    model.train()
    train_loss_epoch = 0

    for batch_idx, (x, _) in enumerate(train_loader):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        original_x = x.to(device)
        # original_x = x.to(device).to(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)
        # optimizer.zero_grad()
        # print(f"Shape of input-batch in train function: {original_x.shape}")
        reconstructed_x = model(original_x).squeeze(dim=1)

        # print(f"x shape in loss: {original_x.shape}")
        # print(f"recon_x shape in loss: {reconstructed_x.shape}")
        recon_loss = model.get_reconstructionloss(original_x, reconstructed_x)
        flow_loss = model.get_flow_loss()
        wandb.log({'recon_loss':recon_loss, 'flow loss':flow_loss})
        # print(f"Recon loss: {recon_loss}, Flow loss: {flow_loss}")

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
        # original_x = x.to(device).to(memory_format=torch.channels_last)

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
            # x,y = x.to(device).to(memory_format=torch.channels_last), y.to(device)
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
    # wandb.name = 'testing'
    # (mode="disabled")

    wandb.init(
    # set the wandb project where this run will be logged
    project="ae_flow",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CHEST_XRAY",
    "epochs": 10,
    }
)


    train_loader, test_loader, validate_loader = load(data_dir='data/chest_xray/',batch_size=64, num_workers=3)
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

    # Save validation images to the model for later on:
    im_normal, _ = next(iter(validate_loader[0]))
    im_abnormal, _ = next(iter(validate_loader[1]))
    # (img, label) = next(iter(validate_loader))
    # i, (data, target) = enumerate(validate_loader, 0)
    # print(target)

    model.sample_images_normal = im_normal[:3]
    model.sample_images_abnormal = im_abnormal[:3]
    # model.sample_images_normal = [data for i, (data, target) in enumerate(validate_loader, 0) if target==0]
    # model.sample_images_abnormal = [torch.Tensor(item[0]) for item in list(enumerate(validate_loader)) if item[1]==1]
    # print(model.sample_images_normal)
    # assert(False)

    ## Using Channels last memory (https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)
    # Does not seem to work
    # model = model.to(memory_format=torch.channels_last)

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

            #TODO: Extend to include 3 normal and 3 abnormal images
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            
            rec_images = model(model.sample_images_normal.to(device)).squeeze(dim=1)
            grid = make_grid(model.sample_images_normal.to(device) + rec_images, nrow = 2)
            images = wandb.Image(grid, caption="Top: Input, Middle: Reconstructed")
            wandb.log({"normal reconstruction images": images})

            rec_images = model(model.sample_images_abnormal.to(device)).squeeze(dim=1)
            grid = make_grid(model.sample_images_abnormal.to(device) + rec_images, nrow = 2)
            images = wandb.Image(grid, caption="Top: Input, Middle: Reconstructed")
            wandb.log({"abnormal reconstruction images": images})


        print(f"Duration for epoch {epoch}: {time.time() - start}")
        wandb.log({'time per epoch': time.time() - start})
    
    wandb.finish()


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
