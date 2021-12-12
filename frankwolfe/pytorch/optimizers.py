# ===========================================================================
# Project:      StochasticFrankWolfe 2020 / IOL Lab @ ZIB
# File:         pytorch/optimizers.py
# Description:  Pytorch implementation of Stochastic Frank Wolfe, AdaGradSFW and SGD with projection
# ===========================================================================
import torch


class SFW(torch.optim.Optimizer):
    """Stochastic Frank Wolfe Algorithm
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        learning_rate (float): learning rate between 0.0 and 1.0
        rescale (string or None): Type of learning_rate rescaling. Must be 'diameter', 'gradient' or None
        momentum (float): momentum factor, 0 for no momentum
    """

    def __init__(self, params, learning_rate=0.1, rescale='diameter', momentum=0.9):
        # if not (0.0 <= learning_rate <= 1.0):
        #    raise ValueError("Invalid learning rate: {}".format(learning_rate))
        if not (0.0 <= momentum <= 1.0):
            raise ValueError("Momentum must be between [0, 1].")
        if not (rescale in ['diameter', 'gradient', None]):
            raise ValueError("Rescale type must be either 'diameter', 'gradient' or None.")

        # Parameters
        self.rescale = rescale

        defaults = dict(lr=learning_rate, momentum=momentum)
        super(SFW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, constraints, closure=None):
        """Performs a single optimization step.
        Args:
            constraints (iterable): list of constraints, where each is an initialization of Constraint subclasses
            parameter groups
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad

                # Add momentum
                momentum = group['momentum']
                if momentum > 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = d_p.detach().clone()
                    else:
                        param_state['momentum_buffer'].mul_(momentum).add_(d_p, alpha=1 - momentum)
                        d_p = param_state['momentum_buffer']

                v = constraints[idx].lmo(d_p)  # LMO optimal solution

                if self.rescale == 'diameter':
                    # Rescale lr by diameter
                    factor = 1. / constraints[idx].get_diameter()
                elif self.rescale == 'gradient':
                    # Rescale lr by gradient
                    factor = torch.norm(d_p, p=2) / torch.norm(p - v, p=2)
                else:
                    # No rescaling
                    factor = 1

                lr = max(0.0, min(factor * group['lr'], 1.0))  # Clamp between [0, 1]
                #print('factor:', factor, 'effect_lr:', lr)

                p.mul_(1 - lr)
                p.add_(v, alpha=lr)
                idx += 1
        return loss


class AdaGradSFW(torch.optim.Optimizer):
    """AdaGrad Stochastic Frank-Wolfe algorithm.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        inner_steps (integer, optional): number of inner iterations (default: 2)
        learning_rate (float, optional): learning rate (default: 1e-2)
        delta (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)
        momentum (float, optional): momentum factor
    """

    def __init__(self, params, inner_steps=2, learning_rate=1e-2, delta=1e-8, momentum=0.9):
        if not 0.0 <= learning_rate:
            raise ValueError("Invalid learning rate: {}".format(learning_rate))
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("Momentum must be between [0, 1].")
        if not 0.0 <= delta:
            raise ValueError("Invalid delta value: {}".format(delta))
        if not int(inner_steps) == inner_steps and not 0.0 <= inner_steps:
            raise ValueError("Number of inner iterations needs to be a positive integer: {}".format(inner_steps))

        self.K = inner_steps

        defaults = dict(lr=learning_rate, delta=delta, momentum=momentum)
        super(AdaGradSFW, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['sum'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    @torch.no_grad()
    def step(self, constraints, closure=None):
        """Performs a single optimization step.
        Args:
            constraints (iterable): list of constraints, where each is an initialization of Constraint subclasses
            parameter groups
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad
                param_state = self.state[p]
                momentum = group['momentum']

                if momentum > 0:
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = d_p.detach().clone()
                    else:
                        param_state['momentum_buffer'].mul_(momentum).add_(d_p, alpha=1 - momentum)
                        d_p = param_state['momentum_buffer']

                param_state['sum'].addcmul_(d_p, d_p, value=1)  # Holds the cumulative sum
                H = torch.sqrt(param_state['sum']).add(group['delta'])

                y = p.detach().clone()
                for _ in range(self.K):
                    d_Q = d_p.addcmul(H, y - p, value=1. / group['lr'])
                    y_v_diff = y - constraints[idx].lmo(d_Q)
                    gamma = group['lr'] * torch.div(torch.sum(torch.mul(d_Q, y_v_diff)),
                                                    torch.sum(H.mul(torch.mul(y_v_diff, y_v_diff))))
                    gamma = max(0.0, min(gamma, 1.0))  # Clamp between [0, 1]

                    y.add_(y_v_diff, alpha=-gamma)  # -gamma needed as we want to add v-y, not y-v
                p.copy_(y)
                idx += 1
        return loss


class SGD(torch.optim.Optimizer):
    """Modified SGD which allows projection via Constraint class"""

    def __init__(self, params, lr=0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if momentum is None:
            momentum = 0
        if weight_decay is None:
            weight_decay = 0
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not (0.0 <= momentum <= 1.0):
            raise ValueError("Momentum must be between [0, 1].")
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, constraints, closure=None):
        """Performs a single optimization step.
        Args:
            constraints (iterable): list of constraints, where each is an initialization of Constraint subclasses
            parameter groups
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        idx = 0
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])

                # Project if necessary
                if not constraints[idx].is_unconstrained():
                    p.copy_(constraints[idx].euclidean_project(p))
                idx += 1

        return loss
