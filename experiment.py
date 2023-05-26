from model.ae_flow_model import AE_Flow_Model
from model.Auto_encoder_seperate import AE_Model
from anomalib.models.fastflow.torch_model import FastflowModel
from dataloader import load, split_data, fold_to_loaders
import torch.utils.data as data
import torch

# We use this object to keep track of experiment/run relevant information, 
# especially to deal with 
class Experiment():
   
    def __init__(self, args, verbose=True):

        super().__init__()
        self.dataset = args.dataset
        self.args = args

        if self.dataset == 'btech':
            self.subsets = ['01', '02', '03']
            self.anomalib_dataset = True
        elif self.dataset == 'mvtec':
            self.subsets = ['pill', 'toothbrush', 'wood', 'grid', 'capsule', 'transistor', 'screw', 'carpet', 'cable', 'bottle', 'tile', 'metal_nut', 'hazelnut', 'leather', 'zipper']
            self.anomalib_dataset = True
        else:
            self.subsets = [None]
            self.anomalib_dataset = False
        
        self.verbose= verbose
        self.model = None

    def initialize_model(self, current_subset):
        self.current_subset = current_subset
        self.subset_results = []
        if self.args.model == 'ae_flow': self.model = AE_Flow_Model(subnet_architecture=self.args.subnet_architecture, n_flowblocks=self.args.n_flowblocks)
        elif self.args.model == 'fastflow':
            self.model = FastflowModel(input_size=(256, 256), backbone="wide_resnet50_2", flow_steps=8, pre_trained=False)
            self.model.training = True

        elif self.args.model == 'autoencoder':
            self.model = AE_Model()
        else:
            raise NotImplementedError
        
    def load_data(self):

        if self.subsets != None: print(f'Running on subset: {self.current_subset}')
        self.train_loader, self.train_abnormal, self.test_loader = load(data_dir=self.args.dataset,batch_size=self.args.batch_size, num_workers=self.args.num_workers, subset=self.current_subset, anomalib_dataset=self.anomalib_dataset)
        self.train_split_normal, self.test_split_normal, self.train_split_abnormal, self.test_split_abnormal = split_data(n_splits=self.args.n_validation_folds, normal_data=self.train_loader, 
                                                                                                      abnormal_data=self.train_abnormal)

        test_split_normal_all = [item for sublist in self.test_split_normal for item in sublist]
        test_split_abnormal_all = [item for sublist in self.test_split_abnormal for item in sublist]

        test_split_normall = torch.utils.data.dataset.Subset(self.train_loader,test_split_normal_all)
        test_split_abnormall = torch.utils.data.dataset.Subset(self.train_abnormal,test_split_abnormal_all)
        self.checkpoint_loader = data.DataLoader(torch.utils.data.ConcatDataset([test_split_normall, test_split_abnormall]), num_workers = 3, batch_size=64)
        
        self.threshold_loader_all = data.DataLoader(torch.utils.data.ConcatDataset([self.train_loader, self.train_abnormal]), num_workers = 3, batch_size=64)
    
    def load_fold_data(self, fold):
        return fold_to_loaders(fold, self.train_split_normal, self.test_split_normal,self.train_split_abnormal, self.test_split_abnormal, self.args.num_workers, self.train_loader, self.train_abnormal)
   