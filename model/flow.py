import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import FrEIA.framework as Ff
import FrEIA.modules as Fm

class FlowModule(nn.Module):

    def __init__(self, subnet_architecture='conv_like', custom_computation_graph=False):
        super(FlowModule, self).__init__()
        
        
        # Most direct computation for this part can be found here:
        # https://vislearn.github.io/FrEIA/_build/html/tutorial/graph_inns.html
        if custom_computation_graph:
            in = Ff.InputNode(1024, 16, 16)
            
            
#             self.inn = Ff.GraphINN([in1, ])
            
      
            
        
        self.inn = Ff.SequenceINN(1024, 16, 16)
        for k in range(8):
            if subnet_architecture == 'conv_like':
                self.inn.append(Fm.AllInOneBlock, subnet_constructor=FlowModule.subnet_conv_3x3_1x1, permute_soft=False)
            if subnet_architecture == 'resnet_like':
                #self.inn.append(Fm.AllInOneBlock, subnet_constructor=FlowModule.resnet_type_network, permute_soft=False)
                #self.inn.append(Fm.AllInOneBlock, subnet_constructor=FlowModule.shortcut_connection, permute_soft=False)
                self.inn.append((Fm.AllInOneBlock, subnet_constructor=FlowModule.resnet, permute_soft=False))
                # Here just concatenat?
    
    # from Pim: let's try to see if this works to have a proper shortcut conncection
    def resnet(c_in, c_out):
        return FlowModule.subnet_conv_3x3_1x1(c_in, c_out) + FlowModule.shortcut_connection(c_in, c_out)

    def subnet_conv_3x3_1x1(c_in, c_out):
        return nn.Sequential(nn.Conv2d(c_in, 256,   3, padding=1), nn.ReLU(),
                            nn.Conv2d(256,  c_out, 1))
    
    def resnet_type_network(c_in, c_out):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding='same'), nn.ReLU(), 
            nn.BatchNorm2d(c_out)
            )
    
    def shortcut_connection(c_in, c_out):
        return nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=1))
        

    def forward(self, x):
        z, log_jac_det = self.inn(x)
        return z, log_jac_det
    

    def reverse(self,z):
        x_rev, log_jac_det_rev = self.inn(z, rev=True)
        return x_rev, log_jac_det_rev
    
