import math

import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import Module

class CosineLinear(Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def normalized_weight_matrix(self):
        """Return the row-normalized classifier matrix with shape [C, D]."""
        return F.normalize(self.weight, p=2, dim=1)

    def angular_forward_normalized(self, normalized_input):
        """Classify already-normalized features [B, D] without sigma scaling."""
        return F.linear(normalized_input, self.normalized_weight_matrix())

    def angular_forward(self, input):
        """Return scale-free cosine scores [B, C] for raw features [B, D]."""
        normalized_input = F.normalize(input, p=2, dim=1)
        return self.angular_forward_normalized(normalized_input)

    def _scale_angles(self, angles):
        """Apply the optional learned scalar to angular scores [B, C]."""
        if self.sigma is not None:
            return self.sigma * angles
        return angles

    def forward_with_angles(self, input):
        """Return ``(scaled_logits, angular_scores)`` from one head evaluation."""
        angles = self.angular_forward(input)
        return self._scale_angles(angles), angles

    def forward(self, input):
        """Return the original sigma-scaled cosine logits [B, C]."""
        logits, _ = self.forward_with_angles(input)
        return logits

class SplitCosineLinear(Module):
    def __init__(self, in_features, out_features1, out_features2, sigma=True):
        super(SplitCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features1 + out_features2
        self.fc1 = CosineLinear(in_features, out_features1, False)
        self.fc2 = CosineLinear(in_features, out_features2, False)
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter('sigma', None)

    def normalized_weight_matrix(self):
        """Return normalized old/new weights in model class order, shape [C, D]."""
        return torch.cat((self.fc1.normalized_weight_matrix(),
                          self.fc2.normalized_weight_matrix()), dim=0)

    def angular_forward_normalized(self, normalized_input):
        """Classify normalized features [B, D] without shared sigma scaling."""
        old_angles = self.fc1.angular_forward_normalized(normalized_input)
        new_angles = self.fc2.angular_forward_normalized(normalized_input)
        return torch.cat((old_angles, new_angles), dim=1)

    def angular_forward(self, x):
        """Return scale-free old/new cosine scores [B, C] for raw features."""
        normalized_input = F.normalize(x, p=2, dim=1)
        return self.angular_forward_normalized(normalized_input)

    def _scale_angles(self, angles):
        """Apply the optional shared scalar to angular scores [B, C]."""
        if self.sigma is not None:
            return self.sigma * angles
        return angles

    def forward_with_angles(self, x):
        """Return scaled logits and angles while evaluating each sub-head once."""
        old_angles = self.fc1(x)
        new_angles = self.fc2(x)
        angles = torch.cat((old_angles, new_angles), dim=1)
        return self._scale_angles(angles), angles

    def forward(self, x):
        """Return the original shared-sigma cosine logits [B, C]."""
        logits, _ = self.forward_with_angles(x)
        return logits
