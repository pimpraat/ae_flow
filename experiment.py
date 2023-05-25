from model.ae_flow_model import AE_Flow_Model
from model.Auto_encoder_seperate import AE_Model
from anomalib.models.fastflow.torch_model import FastflowModel

# We use this object to keep track of experiment/run relevant information, 
# especially to deal with 
class Experiment():
   
    def __init__(self, args):

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
        
        self.baseline=False

        self.model = None

    def initialize_model(self):
        if self.args.model == 'ae_flow': self.model = AE_Flow_Model(subnet_architecture=self.args.subnet_architecture, n_flowblocks=self.args.n_flowblocks)
        elif self.args.model == 'fastflow':
            self.model = FastflowModel(input_size=(256, 256), backbone="wide_resnet50_2", flow_steps=8, pre_trained=False)
            self.model.training = True
            self.baseline = True

        elif self.args.model == 'autoencoder':
            self.model = AE_Model()
        else:
            raise NotImplementedError
