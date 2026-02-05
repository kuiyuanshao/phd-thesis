import torch.nn as nn
import copy
import torch
import math

class SWAG(nn.Module):
    def __init__(self, base_model, max_num_models=20, var_clamp=1e-30):
        super(SWAG, self).__init__()
        self.base = copy.deepcopy(base_model)
        self.base.train()
        self.max_num_models = max_num_models
        self.var_clamp = var_clamp
        self.n_models = torch.zeros([1], dtype=torch.long)
        self.params = list()

        for name, param in self.base.named_parameters():
            safe_name = name.replace(".", "_")
            self.register_buffer(f"{safe_name}_mean", torch.zeros_like(param.data))
            self.register_buffer(f"{safe_name}_sq_mean", torch.zeros_like(param.data))
            self.register_buffer(f"{safe_name}_cov_mat_sqrt", torch.empty(0, param.numel()))
            self.params.append((name, safe_name, param))

    def collect_model(self, base_model):
        curr_params = dict(base_model.named_parameters())
        n = self.n_models.item()
        for name, safe_name, _ in self.params:
            if name not in curr_params: continue
            param = curr_params[name]
            mean = getattr(self, f"{safe_name}_mean")
            sq_mean = getattr(self, f"{safe_name}_sq_mean")
            cov_mat_sqrt = getattr(self, f"{safe_name}_cov_mat_sqrt")

            mean = mean * n / (n + 1.0) + param.data.to(mean.device) / (n + 1.0)
            sq_mean = sq_mean * n / (n + 1.0) + (param.data.to(sq_mean.device) ** 2) / (n + 1.0)
            dev = (param.data.to(mean.device) - mean).view(-1, 1)
            cov_mat_sqrt = torch.cat((cov_mat_sqrt, dev.t()), dim=0)

            if (cov_mat_sqrt.size(0)) > self.max_num_models:
                cov_mat_sqrt = cov_mat_sqrt[1:, :]

            setattr(self, f"{safe_name}_mean", mean)
            setattr(self, f"{safe_name}_sq_mean", sq_mean)
            setattr(self, f"{safe_name}_cov_mat_sqrt", cov_mat_sqrt)
        self.n_models.add_(1)

    def sample(self, scale=1, cov=True):
        scale_sqrt = scale ** 0.5
        for name, safe_name, base_param in self.params:
            mean = getattr(self, f"{safe_name}_mean")
            sq_mean = getattr(self, f"{safe_name}_sq_mean")
            cov_mat_sqrt = getattr(self, f"{safe_name}_cov_mat_sqrt")

            var = torch.clamp(sq_mean - mean ** 2, min=self.var_clamp)
            var_sample = var.sqrt() * torch.randn_like(var)

            if cov and cov_mat_sqrt.size(0) > 0:
                K = cov_mat_sqrt.size(0)
                z2 = torch.randn(K, 1, device=mean.device)
                cov_sample = cov_mat_sqrt.t().matmul(z2).view_as(mean)
                cov_sample /= (self.max_num_models - 1) ** 0.5
                rand_sample = var_sample + cov_sample
            else:
                rand_sample = var_sample

            sample = mean + scale_sqrt * rand_sample
            base_param.data.copy_(sample)
        return self.base