
import torch
from torch import nn
from copy import deepcopy
import torch.nn.functional as F

from layers import BBBLinear

class ELBO(nn.Module):
    def __init__(self, kl_weight=1.0):
        """
        Initialize the ELBO loss class.

        Args:
            kl_weight (float): Weight for the KL divergence term.
        """
        super(ELBO, self).__init__()
        self.kl_weight = kl_weight

    def forward(self, logits, aux_targets, kl_divergence):
        """
        Compute the ELBO loss.

        Args:
            logits (torch.Tensor): Logits output from the model.
            aux_targets (torch.Tensor): Auxiliary targets for the NLL loss.
            kl_divergence (torch.Tensor): KL divergence term from the model.

        Returns:
            torch.Tensor: The computed ELBO loss.
        """
        # Compute the negative log-likelihood loss
        nll_loss = F.nll_loss(F.log_softmax(logits, dim=1), aux_targets)

        # Combine NLL loss and KL divergence
        elbo_loss = nll_loss + self.kl_weight * kl_divergence

        return elbo_loss
    
        return ce + self.beta * kl
    


class LLL_Net(nn.Module):
    """Basic class for implementing networks"""

    def __init__(self, model, remove_existing_head=False):
        head_var = model.head_var
        assert type(head_var) == str
        assert not remove_existing_head or hasattr(model, head_var), \
            "Given model does not have a variable called {}".format(head_var)
        assert not remove_existing_head or type(getattr(model, head_var)) in [nn.Sequential, nn.Linear, BBBLinear], \
            "Given model's head {} does is not an instance of nn.Sequential or nn.Linear".format(head_var)
        super(LLL_Net, self).__init__()

        self.model = model
        last_layer = getattr(self.model, head_var)

        if remove_existing_head:
            if isinstance(last_layer, nn.Sequential):
                self.out_size = last_layer[-1].in_features
                del last_layer[-1]
            elif isinstance(last_layer, nn.Linear):
                self.out_size = last_layer.in_features
                setattr(self.model, head_var, nn.Sequential())
            elif isinstance(last_layer, BBBLinear):  # <-- ADD THIS
                self.out_size = last_layer.in_features
                setattr(self.model, head_var, nn.Sequential())
            else:
                raise TypeError(
                    f"Given model's head {head_var} is not an instance of "
                    f"nn.Sequential, nn.Linear, or BBBLinear"
                )

        self.heads = nn.ModuleList()
        self.task_cls = []
        self.task_offset = []
        self._initialize_weights()

    def add_head(self, num_outputs):
        """Add a new head with the corresponding number of outputs. Also update the number of classes per task and the
        corresponding offsets
        """
        self.heads.append(BBBLinear(self.out_size, num_outputs))
        self.task_cls = torch.tensor([head.out_features for head in self.heads])
        self.task_offset = torch.cat([torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]])

    def forward(self, x, return_features=False):
        """Applies the forward pass

        Simplification to work on multi-head only -- returns all head outputs in a list
        Args:
            x (tensor): input images
            return_features (bool): return the representations before the heads
        """
        x = self.model(x)
        assert (len(self.heads) > 0), "Cannot access any head"
        y = []
        for head in self.heads:
            y.append(head(x))
        if return_features:
            return y, x
        else:
            return y

    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))
        return

    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False

    def freeze_backbone(self):
        """Freeze all parameters from the main model, but not the heads"""
        for param in self.model.parameters():
            param.requires_grad = False

    def freeze_bn(self):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _initialize_weights(self):
        """Initialize weights using different strategies"""
        # TODO: add different initialization strategies
        pass

    def KL(self):
        """Compute the total KL divergence from all Bayesian layers."""
        kl = 0
        for module in self.model.modules():
            if hasattr(module, 'kl_loss'):
                kl += module.kl_loss()
        for head in self.heads:
            if hasattr(head, 'kl_loss'):
                kl += head.kl_loss()
        return kl