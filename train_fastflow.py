"""Implementation of FastFlow in PyTorch. Code sourced from the following repository:
https://github.com/gathierry/FastFlow
"""

import argparse
import os

import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, auc, roc_curve

import baselines.fastflow.constants as const
import baselines.fastflow.torch_model as fastflow
import baselines.fastflow.utils as utils

import wandb

from dataloader import load

# Make sure the following reads to a file with your own W&B API/Server key
WANDBKEY = open("wandbkey.txt", "r").read()

def build_model():
    # init default FastFlow model
    model = fastflow.FastFlow(
        input_size = 256,
        backbone_name = const.BACKBONE_RESNET18,
        flow_steps = 8,
        hidden_ratio = 1.0,
        conv3x3_only = False
    )
    print(
        "Model A.D. Param#: {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    return model


def build_optimizer(model):
    return torch.optim.Adam(
        model.parameters(), lr=const.LR, weight_decay=const.WEIGHT_DECAY
    )


def train_one_epoch(dataloader, model, optimizer, epoch):
    model.train()
    loss_meter = utils.AverageMeter()
    for step, data in enumerate(dataloader):
        data = torch.Tensor(data)
        # forward
        data = data.cuda()
        ret = model(data)
        loss = ret["loss"]
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # log
        loss_meter.update(loss.item())
        if (step + 1) % const.LOG_INTERVAL == 0 or (step + 1) == len(dataloader):
            print(
                "Epoch {} - Step {}: loss = {:.3f}({:.3f})".format(
                    epoch + 1, step + 1, loss_meter.val, loss_meter.avg
                )
            )


def eval_model(epoch, model, data_loader, threshold=None, _print=False, return_only_anomaly_scores=False, validation=True):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    with torch.no_grad(): # Deactivate gradients for the following code
        anomaly_scores, true_labels = [], []

        for batch_idx, (x, y) in enumerate(data_loader):
    
            data = torch.Tensor(data)
            data, targets = data.cuda(), targets.cuda()
            
            ret = model(data)
            outputs = ret["anomaly_map"].cpu().detach()
            outputs = outputs.flatten()
            targets = targets.flatten()

            true_labels.append(y)

    # true = true_labels[0].cpu()
    true = [tensor.cpu().numpy() for tensor in true_labels]
    true = [item for sublist in true for item in sublist]


    anomaly_scores = [tensor.cpu().numpy() for tensor in anomaly_scores]
    anomaly_scores = [item for sublist in anomaly_scores for item in sublist]
    if return_only_anomaly_scores: return true, anomaly_scores

def eval_once(dataloader, model):
    model.eval()

    for data, targets in dataloader:
        data = torch.Tensor(data)
        data, targets = data.cuda(), targets.cuda()
        with torch.no_grad():
            ret = model(data)
        outputs = ret["anomaly_map"].cpu().detach()
        outputs = outputs.flatten()
        targets = targets.flatten()
        
    print("AUROC: {}".format(auroc))


def train(args):

    #TODO: Make private!
    wandb.login(key=WANDBKEY)

    """wandb.init(
        # set the wandb project where this run will be logged
        project="ae_flow",
        
        # track hyperparameters and run metadata
        #TODO: Update these to be in line with the arguments
        config={
        "model": "Fastflow",
        "dataset": args.dataset,
        "epochs": args.epochs,
        "optim_lr": args.optim_lr,
        "optim_momentum": args.optim_momentum,
        "optim_weight_decay": args.optim_weight_decay
        }
    )"""

    train_loader, train_complete, validate_loader, test_loader = load(data_dir=args.dataset,batch_size=args.batch_size, num_workers=args.num_workers)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    os.makedirs(const.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(
        const.CHECKPOINT_DIR, "exp%d" % len(os.listdir(const.CHECKPOINT_DIR))
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    #config = yaml.safe_load(open(args.config, "r"))
    model = build_model()
    optimizer = build_optimizer(model)

    model.cuda()

    for epoch in range(args.epochs):
        train_one_epoch(train_loader, model, optimizer, epoch)
        if (epoch + 1) % const.EVAL_INTERVAL == 0:
            eval_once(test_loader, model)
        if (epoch + 1) % const.CHECKPOINT_INTERVAL == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(checkpoint_dir, "%d.pt" % epoch),
            )
    
    eval_once(args, model)
    wandb.finish()



def evaluate(args, test_dataloader):
    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.cuda()
    eval_once(test_dataloader, model)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="run eval only")
    parser.add_argument(
        "-ckpt", "--checkpoint", type=str, help="path to load checkpoint",
        default="./baselines/fastflow/checkpoints"
    )
    
    # additional arguments
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size to use for training')
    parser.add_argument('--dataset',default='chest_xray', type=str, help='Which dataset to run. Choose from: [OCT2017, chest_xray, ISIC, BRATS, MIIC]')
    parser.add_argument('--optim_lr', type=float, default=2e-3,
                        help='')
    parser.add_argument('--optim_momentum', type=float, default=0.9, 
                        help='')
    parser.add_argument('--optim_weight_decay', type=float, default=10e-5,
                        help='')
    parser.add_argument('--epochs', default=15, type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to use in the data loaders.' +
                             'To have a truly deterministic run, this has to be 0.')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)