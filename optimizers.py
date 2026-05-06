import torch
from torch.optim import Optimizer

class AdaSmoothDelta(Optimizer):
    def __init__(self, params, lr=1.0, rho=0.9, smooth=0.1, eps=1e-6):
        defaults = dict(lr=lr, rho=rho, smooth=smooth, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            rho = group['rho']
            smooth = group['smooth']
            eps = group['eps']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['grad_avg'] = torch.zeros_like(p)
                    state['delta_avg'] = torch.zeros_like(p)
                    state['acc_grad'] = torch.zeros_like(p)
                    state['acc_delta'] = torch.zeros_like(p)

                grad_avg = state['grad_avg']
                delta_avg = state['delta_avg']
                acc_grad = state['acc_grad']
                acc_delta = state['acc_delta']

                grad_avg.mul_(1 - smooth).add_(grad * smooth)
                acc_grad.mul_(rho).addcmul_(grad_avg, grad_avg, value=1 - rho)

                update = (acc_delta + eps).sqrt() / (acc_grad + eps).sqrt() * grad_avg

                delta_avg.mul_(1 - smooth).add_(update * smooth)
                p.add_(delta_avg, alpha=-lr)

                acc_delta.mul_(rho).addcmul_(delta_avg, delta_avg, value=1 - rho)

        return loss