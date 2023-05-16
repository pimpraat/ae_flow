import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import FrEIA

from FrEIA.modules import InvertibleModule
from typing import Sequence, Union
from copy import deepcopy

# FIXME: this should be put somewhere else
class SumFMmodule(InvertibleModule):
    """Invertible merge operation.

    Concatenates a list of incoming tensors along a given dimension and passes
    on the result. Inverse is the corresponding split operation.
    """

    def __init__(self,
                 dims_in: Sequence[Sequence[int]],
                 dim: int = 0,
     ):
        """Inits the Concat module with the attributes described above and
        checks that all dimensions are compatible.

        Args:
          dims_in:
            A list of tuples containing the non-batch dimensionality of all
            incoming tensors. Handled automatically during compute graph setup.
            Dimensionality of incoming tensors must be identical, except in the
            merge dimension ``dim``. Concat only makes sense with multiple input
            tensors.
          dim:
            Index of the dimension along which to concatenate, not counting the
            batch dimension. Defaults to 0, i.e. the channel dimension in structured
            data.
        """
        super().__init__(dims_in)
        assert len(dims_in) > 1, ("Concatenation only makes sense for "
                                  "multiple inputs")
        assert len(dims_in[0]) >= dim, "Merge dimension index out of range"
        assert all(len(dims_in[i]) == len(dims_in[0])
                   for i in range(len(dims_in))), (
                           "All input tensors must have same number of "
                           "dimensions"
                   )
        assert all(dims_in[i][j] == dims_in[0][j] for i in range(len(dims_in))
                   for j in range(len(dims_in[i])) if j != dim), (
                           "All input tensor dimensions except merge "
                           "dimension must be identical"
                   )
        self.dim = dim
        self.split_size_or_sections = [dims_in[i][dim]
                                       for i in range(len(dims_in))]

    def forward(self, x, rev=False, jac=True):
        """See super class InvertibleModule.
        Jacobian log-det of concatenation is always zero."""
        if rev:
            return x[0] - x[1], 0
            #return torch.split(x[0], self.split_size_or_sections,
            #                   dim=self.dim+1), 0
        else:
            summed = x[0] + x[1]
            return [summed], 0

    def output_dims(self, input_dims):
        """See super class InvertibleModule."""
        assert len(input_dims) > 1, ("Concatenation only makes sense for "
                                     "multiple inputs")
        output_dims = deepcopy(list(input_dims[0]))
        output_dims[self.dim] = sum(input_dim[self.dim]
                                    for input_dim in input_dims)
        return [tuple(output_dims)]
    

class FlowModule(nn.Module):

    def __init__(self, subnet_architecture='conv_like', custom_computation_graph=False, n_flowblocks=8):
        super(FlowModule, self).__init__()
        
        # Most direct computation for this part can be found here:
        # https://vislearn.github.io/FrEIA/_build/html/tutorial/graph_inns.html
        if custom_computation_graph:
            print('using custom computation graph, now with custom subnet constructor')

            # Try this based on Cyril's answer:
            self.inn = Ff.SequenceINN(1024, 16, 16)

            local_constr = FlowModule.Conv3x3_res_1x1()

            for k in range(8):
                self.inn.append(Fm.AllInOneBlock, subnet_constructor=local_constr, permute_soft=False)


            # outputs = [Ff.InputNode(1024, 16, 16, name="Input at the beginning")]
            # final_nodes = [outputs[0]]
            
            # for k in range(n_flowblocks):
            #     net = Ff.Node(outputs[-1], Fm.AllInOneBlock, {'subnet_constructor': FlowModule.resnet, 'permute_soft': False})
            #     shortcut = Ff.Node(outputs[-1], Fm.AllInOneBlock, {'subnet_constructor': FlowModule.shortcut_connection, 'permute_soft':False})
            #     concat = Ff.Node([net.out0, shortcut.out0], SumFMmodule, {'dim':1}, name=str(f'Concat with shortcut connection at block {k}'))
            #     final_nodes.extend([net, shortcut, concat])
            #     outputs.extend([concat])
                
            # final_nodes.append(Ff.OutputNode(outputs[-1], name="Final output"))
            # self.inn = Ff.GraphINN(final_nodes)
        if not custom_computation_graph:
            self.inn = Ff.SequenceINN(1024, 16, 16)
            for k in range(8):
                if subnet_architecture == 'conv_like':
                    self.inn.append(Fm.AllInOneBlock, subnet_constructor=FlowModule.subnet_conv_3x3_1x1, permute_soft=False)
                if subnet_architecture == 'resnet_like':
                    self.inn.append(Fm.AllInOneBlock, subnet_constructor=FlowModule.resnet, permute_soft=False)
                    # Here just concatenat?
                if subnet_architecture == 'resnet_like_old':
                    self.inn.append(Fm.AllInOneBlock, subnet_constructor=FlowModule.resnet_type_network, permute_soft=False)
                    self.inn.append(Fm.AllInOneBlock, subnet_constructor=FlowModule.shortcut_connection, permute_soft=False)


    # Try the approach of a custom subnet_constructor
    class Conv3x3_res_1x1(nn.Module):
        def __init__(self, size_in, size_out):
            super().__init__()
            self.conv = nn.Conv2d(size_in, size_out, kernel_size=3, stride=1, bias=1)
            self.bn = nn.BatchNorm2d(size_out)
            self.relu = nn.ReLU(inplace=True)
            self.res = nn.Conv2d(size_in, size_out, kernel_size=1, stride=1, bias=0)
        def forward(self, x):
            output = self.conv(x)
            output = self.bn(output)
            output = self.relu(output)
            return output + self.res(x)


    # from Pim: let's try to see if this works to have a proper shortcut conncection
    def resnet(c_in, c_out):

        # what about this? 256 or something else?
        return nn.Sequential(FlowModule.subnet_conv_3x3_1x1(c_in, 256), nn.ReLU(), FlowModule.shortcut_connection(256, c_out))

        # this doesn't work, as input and output are mismatched between the two
        #return FlowModule.subnet_conv_3x3_1x1(c_in, c_out) + FlowModule.shortcut_connection(c_in, c_out)

    def subnet_conv_3x3_1x1(c_in, c_out):
        return nn.Sequential(nn.Conv2d(c_in, 256,   3, padding=1), nn.ReLU(),
                            nn.Conv2d(256,  c_out, 1))

    # old resnet type
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
    
