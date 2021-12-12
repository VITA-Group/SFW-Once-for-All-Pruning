import torch
from torch import nn
from utils.gradinit_optimizers import RescaleAdam
from neural_network.modules import Scale, Bias
import numpy as np
import os


def get_ordered_params(net):
    param_list = []
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
            param_list.append(m.weight)
            if m.bias is not None:
                param_list.append(m.bias)
        elif isinstance(m, Scale):
            param_list.append(m.weight)
        elif isinstance(m, Bias):
            param_list.append(m.bias)

    return param_list


def set_param(module, name, alg, eta, grad):
    weight = getattr(module, name)
    # remove this parameter from parameter list
    del module._parameters[name]

    # compute the update steps according to the optimizers
    if alg.lower() == 'sgd':
        gstep = eta * grad
    elif alg.lower() == 'adam':
        gstep = eta * grad.sign()
    else:
        raise RuntimeError("Optimization algorithm {} not defined!".format(alg))

    # add the updated parameter as the new parameter
    module.register_parameter(name + '_prev', weight)

    # recompute weight before every forward()
    updated_weight = weight - gstep.data
    setattr(module, name, updated_weight)


def set_param_sfw(module, name, grad, v, eta, rescale = 'gradient'):
    weight = getattr(module, name)
    # remove this parameter from parameter list
    del module._parameters[name]

    if rescale == 'gradient':
        # Rescale lr by gradient
        factor = torch.norm(grad, p=2) / torch.norm(weight - v, p=2)
    else:
        # No rescaling
        factor = 1
    lr = max(0.0, min(factor * eta, 1.0))  # Clamp between [0, 1]

    # add the updated parameter as the new parameter
    module.register_parameter(name + '_prev', weight)

    # recompute weight before every forward()
    updated_weight = weight * (1-lr) + v * lr
    setattr(module, name, updated_weight)


def take_opt_step(net, grad_list, alg='sgd', eta=0.1):
    """Take the initial step of the chosen optimizer.
    """
    assert alg.lower() in ['adam', 'sgd']

    idx = 0
    for n, m in net.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
            grad = grad_list[idx]
            set_param(m, 'weight', alg, eta, grad)
            idx += 1

            if m.bias is not None:
                grad = grad_list[idx]
                set_param(m, 'bias', alg, eta, grad)
                idx += 1
        elif isinstance(m, Scale):
            grad = grad_list[idx]
            set_param(m, 'weight', alg, eta, grad)
            idx += 1
        elif isinstance(m, Bias):
            grad = grad_list[idx]
            set_param(m, 'bias', alg, eta, grad)
            idx += 1

        
def take_opt_step_sfw(net, grad_list, constraints, config):
    idx = 0
    for n, m in net.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
            grad = grad_list[idx]
            v = constraints[idx].lmo(grad)
            set_param_sfw(m, 'weight', grad, v, config['lr'])
            idx += 1
            if m.bias is not None:
                grad = grad_list[idx]
                v = constraints[idx].lmo(grad)
                set_param_sfw(m, 'bias', grad, v, config['lr'])
                idx += 1
        elif isinstance(m, Scale):
            grad = grad_list[idx]
            v = constraints[idx].lmo(grad)
            set_param_sfw(m, 'weight', grad, v, config['lr'])
            idx += 1
        elif isinstance(m, Bias):
            grad = grad_list[idx]
            v = constraints[idx].lmo(grad)
            set_param_sfw(m, 'weight', grad, v, config['lr'])
            idx += 1


def recover_params(net):
    """Reset the weights to the original values without the gradient step
    """

    def recover_param_(module, name):
        delattr(module, name)
        setattr(module, name, getattr(module, name + '_prev'))
        del module._parameters[name + '_prev']

    for n, m in net.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
            recover_param_(m, 'weight')
            if m.bias is not None:
                recover_param_(m, 'bias')
        elif isinstance(m, Scale):
            recover_param_(m, 'weight')
        elif isinstance(m, Bias):
            recover_param_(m, 'bias')


def set_bn_modes(net):
    """Switch the BN layers into training mode, but does not track running stats.
    """
    for n, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            m.training = True
            m.track_running_stats = False


def recover_bn_modes(net):
    for n, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = True


def get_scale_stats(model, optimizer):
    stat_dict = {}
    # all_s_list = [p.norm().item() for n, p in model.named_parameters() if 'bias' not in n]
    all_s_list = []
    for param_group in optimizer.param_groups:
        for p in param_group['params']:
            all_s_list.append(optimizer.state[p]['alpha'])
    stat_dict['s_max'] = max(all_s_list)
    stat_dict['s_min'] = min(all_s_list)
    stat_dict['s_mean'] = np.mean(all_s_list)
    all_s_list = []
    for n, p in model.named_parameters():
        if 'bias' not in n:
            all_s_list.append(optimizer.state[p]['alpha'])
    stat_dict['s_weight_max'] = max(all_s_list)
    stat_dict['s_weight_min'] = min(all_s_list)
    stat_dict['s_weight_mean'] = np.mean(all_s_list)

    return stat_dict


def get_batch(data_iter, data_loader):
    try:
        inputs, targets = next(data_iter)
    except:
        data_iter = iter(data_loader)
        inputs, targets = next(data_iter)
    inputs, targets = inputs.cuda(), targets.cuda()
    return data_iter, inputs, targets


def gradinit(net, dataloader, args):
    if args.gradinit_resume:
        print("Resuming GradInit model from {}".format(args.gradinit_resume))
        sdict = torch.load(args.gradinit_resume)
        net.load_state_dict(sdict)
        return

    # if isinstance(net, torch.nn.DataParallel):
    #     net_top = net.module
    # else:
    #     net_top = net

    bias_params = [p for n, p in net.named_parameters() if 'bias' in n]
    weight_params = [p for n, p in net.named_parameters() if 'weight' in n]

    optimizer = RescaleAdam([{'params': weight_params, 'min_scale': args.gradinit_min_scale, 'lr': args.gradinit_lr},
                                      {'params': bias_params, 'min_scale': 0, 'lr': args.gradinit_lr}],
                                     grad_clip=args.gradinit_grad_clip)

    criterion = nn.CrossEntropyLoss()

    net.eval() # This further shuts down dropout, if any.

    set_bn_modes(net) # Should be called after net.eval()


    total_loss, total_l0, total_l1, total_residual, total_gnorm = 0, 0, 0, 0, 0
    total_sums, total_sums_gnorm = 0, 0
    cs_count = 0
    total_iters = 0
    obj_loss, updated_loss, residual = -1, -1, -1
    data_iter = iter(dataloader)
    # get all the parameters by order
    params_list = get_ordered_params(net)
    while True:
        eta = args.gradinit_eta

        # continue
        # get the first half of the minibatch
        data_iter, init_inputs_0, init_targets_0 = get_batch(data_iter, dataloader)

        # Get the second half of the data.
        data_iter, init_inputs_1, init_targets_1 = get_batch(data_iter, dataloader)

        init_inputs = torch.cat([init_inputs_0, init_inputs_1])
        init_targets = torch.cat([init_targets_0, init_targets_1])
        # compute the gradient and take one step
        outputs = net(init_inputs)
        init_loss = criterion(outputs, init_targets)

        all_grads = torch.autograd.grad(init_loss, params_list, create_graph=True)

        # Compute the loss w.r.t. the optimizer
        if args.gradinit_alg.lower() == 'adam':
            # grad-update inner product
            gnorm = sum([g.abs().sum() for g in all_grads])
            loss_grads = all_grads
        else:
            gnorm_sq = sum([g.square().sum() for g in all_grads])
            gnorm = gnorm_sq.sqrt()
            if args.gradinit_normalize_grad:
                loss_grads = [g / gnorm for g in all_grads]
            else:
                loss_grads = all_grads

        total_gnorm += gnorm.item()
        total_sums_gnorm += 1
        if gnorm.item() > args.gradinit_gamma:
            # project back into the gradient norm constraint
            optimizer.zero_grad()
            gnorm.backward()
            optimizer.step(is_constraint=True)

            cs_count += 1
        else:
            # take one optimization step
            take_opt_step(net, loss_grads, alg=args.gradinit_alg, eta=eta)

            total_l0 += init_loss.item()

            data_iter, inputs_2, targets_2 = get_batch(data_iter, dataloader)
            if args.batch_no_overlap:
                # sample a new batch for the half
                data_iter, init_inputs_0, init_targets_0 = get_batch(data_iter, dataloader)
            updated_inputs = torch.cat([init_inputs_0, inputs_2])
            updated_targets = torch.cat([init_targets_0, targets_2])

            # compute loss using the updated network
            # net_top.opt_mode(True)
            updated_outputs = net(updated_inputs)
            # net_top.opt_mode(False)
            updated_loss = criterion(updated_outputs, updated_targets)

            # If eta is larger, we should expect obj_loss to be even smaller.
            obj_loss = updated_loss 

            recover_params(net)
            optimizer.zero_grad()
            obj_loss.backward()
            optimizer.step(is_constraint=False)
            total_l1 += updated_loss.item()

            total_loss += obj_loss.item()
            total_sums += 1

        total_iters += 1
        if (total_sums_gnorm > 0 and total_sums_gnorm % 10 == 0) or total_iters == args.gradinit_iters or total_iters == args.gradinit_iters:
            stat_dict = get_scale_stats(net, optimizer)
            print_str = "Iter {}, obj iters {}, eta {:.3e}, constraint count {} loss: {:.3e} ({:.3e}), init loss: {:.3e} ({:.3e}), update loss {:.3e} ({:.3e}), " \
                        "total gnorm: {:.3e} ({:.3e})\t".format(
                total_sums_gnorm, total_sums, eta, cs_count,
                float(obj_loss), total_loss / total_sums if total_sums > 0 else -1,
                float(init_loss), total_l0 / total_sums if total_sums > 0 else -1,
                float(updated_loss), total_l1 / total_sums if total_sums > 0 else -1,
                float(gnorm), total_gnorm / total_sums_gnorm)

            for key, val in stat_dict.items():
                print_str += "{}: {:.2e}\t".format(key, val)
            print(print_str)

        if total_iters == args.gradinit_iters:
            break

    recover_bn_modes(net)
    if not os.path.exists('chks'):
        os.makedirs('chks')
    torch.save(net.state_dict(), 'chks/{}_init.pth'.format(args.expname))


def gradinit_sfw(model, dataloader, constraints, config):

    gradinit_min_scale = 0.01
    gradinit_lr = 0.001
    gradinit_grad_clip= 1
    batch_no_overlap = False
    gradinit_iters = 390

    bias_params = [p for n, p in model.named_parameters() if 'bias' in n]
    weight_params = [p for n, p in model.named_parameters() if 'weight' in n]

    optimizer = RescaleAdam([{'params': weight_params, 'min_scale': gradinit_min_scale, 'lr': gradinit_lr},
                                      {'params': bias_params, 'min_scale': 0, 'lr': gradinit_lr}],
                                     grad_clip=gradinit_grad_clip)
    criterion = nn.CrossEntropyLoss()
    model.eval() # This further shuts down dropout, if any.
    set_bn_modes(model) # Should be called after net.eval()

    total_loss, total_l0, total_l1, total_residual, total_gnorm = 0, 0, 0, 0, 0
    total_sums = 0
    total_iters = 0
    obj_loss, updated_loss, residual = -1, -1, -1
    data_iter = iter(dataloader)
    # get all the parameters by order
    params_list = get_ordered_params(model)
    while True:
        # continue
        # get the first half of the minibatch
        data_iter, init_inputs_0, init_targets_0 = get_batch(data_iter, dataloader)
        # Get the second half of the data.
        data_iter, init_inputs_1, init_targets_1 = get_batch(data_iter, dataloader)

        init_inputs = torch.cat([init_inputs_0, init_inputs_1])
        init_targets = torch.cat([init_targets_0, init_targets_1])
        # compute the gradient and take one step
        outputs = model(init_inputs)
        init_loss = criterion(outputs, init_targets)
        all_grads = torch.autograd.grad(init_loss, params_list, create_graph=True)

        take_opt_step_sfw(model, all_grads, constraints, config)
        total_l0 += init_loss.item()
        data_iter, inputs_2, targets_2 = get_batch(data_iter, dataloader)
        if batch_no_overlap:
            # sample a new batch for the half
            data_iter, init_inputs_0, init_targets_0 = get_batch(data_iter, dataloader)
        updated_inputs = torch.cat([init_inputs_0, inputs_2])
        updated_targets = torch.cat([init_targets_0, targets_2])

        # compute loss using the updated network
        updated_outputs = model(updated_inputs)
        # net_top.opt_mode(False)
        updated_loss = criterion(updated_outputs, updated_targets)

        # If eta is larger, we should expect obj_loss to be even smaller.
        obj_loss = updated_loss / config['lr']
        recover_params(model)
        optimizer.zero_grad()
        obj_loss.backward()
        optimizer.step(is_constraint=False)
        total_l1 += updated_loss.item()
        total_loss += obj_loss.item()
        total_sums += 1
        total_iters += 1
        if (total_sums > 0 and total_sums % 10 == 0) or total_iters == gradinit_iters:
            stat_dict = get_scale_stats(model, optimizer)
            print_str = "Iter {}, obj iters {}, eta {:.3e}, loss: {:.3e} ({:.3e}), init loss: {:.3e} ({:.3e}), update loss {:.3e} ({:.3e})".format(
                total_sums, total_sums, config['lr'], 
                float(obj_loss), total_loss / total_sums if total_sums > 0 else -1,
                float(init_loss), total_l0 / total_sums if total_sums > 0 else -1,
                float(updated_loss), total_l1 / total_sums if total_sums > 0 else -1)

            for key, val in stat_dict.items():
                print_str += "{}: {:.2e}\t".format(key, val)
            print(print_str)

        if total_iters == gradinit_iters:
            break
    recover_bn_modes(model)


def gradinit_sfw_after_mask(model, dataloader, constraints, config):
    gradinit_min_scale = 0.01
    gradinit_lr = 0.001
    gradinit_grad_clip= 1
    batch_no_overlap = False
    gradinit_iters = 390

    bias_params = [p for n, p in model.named_parameters() if 'bias_orig' in n]
    weight_params = [p for n, p in model.named_parameters() if 'weight_orig' in n]

    optimizer = RescaleAdam([{'params': weight_params, 'min_scale': gradinit_min_scale, 'lr': gradinit_lr},
                                      {'params': bias_params, 'min_scale': 0, 'lr': gradinit_lr}],
                                     grad_clip=gradinit_grad_clip)
    criterion = nn.CrossEntropyLoss()
    model.eval() # This further shuts down dropout, if any.
    set_bn_modes(model) # Should be called after net.eval()

    total_loss, total_l0, total_l1, total_residual, total_gnorm = 0, 0, 0, 0, 0
    total_sums = 0
    total_iters = 0
    obj_loss, updated_loss, residual = -1, -1, -1
    data_iter = iter(dataloader)
    # get all the parameters by order
    params_list = get_ordered_params(model)
    while True:
        # continue
        # get the first half of the minibatch
        data_iter, init_inputs_0, init_targets_0 = get_batch(data_iter, dataloader)
        # Get the second half of the data.
        data_iter, init_inputs_1, init_targets_1 = get_batch(data_iter, dataloader)

        init_inputs = torch.cat([init_inputs_0, init_inputs_1])
        init_targets = torch.cat([init_targets_0, init_targets_1])
        # compute the gradient and take one step
        outputs = model(init_inputs)
        init_loss = criterion(outputs, init_targets)
        all_grads = torch.autograd.grad(init_loss, params_list, create_graph=True)

        take_opt_step_sfw(model, all_grads, constraints, config)
        total_l0 += init_loss.item()
        data_iter, inputs_2, targets_2 = get_batch(data_iter, dataloader)
        if batch_no_overlap:
            # sample a new batch for the half
            data_iter, init_inputs_0, init_targets_0 = get_batch(data_iter, dataloader)
        updated_inputs = torch.cat([init_inputs_0, inputs_2])
        updated_targets = torch.cat([init_targets_0, targets_2])

        # compute loss using the updated network
        updated_outputs = model(updated_inputs)
        # net_top.opt_mode(False)
        updated_loss = criterion(updated_outputs, updated_targets)

        # If eta is larger, we should expect obj_loss to be even smaller.
        obj_loss = updated_loss / config['lr']
        recover_params(model)
        optimizer.zero_grad()
        obj_loss.backward()
        optimizer.step(is_constraint=False)
        total_l1 += updated_loss.item()
        total_loss += obj_loss.item()
        total_sums += 1
        total_iters += 1
        if (total_sums > 0 and total_sums % 10 == 0) or total_iters == gradinit_iters:
            stat_dict = get_scale_stats(model, optimizer)
            print_str = "Iter {}, obj iters {}, eta {:.3e}, loss: {:.3e} ({:.3e}), init loss: {:.3e} ({:.3e}), update loss {:.3e} ({:.3e})".format(
                total_sums, total_sums, config['lr'], 
                float(obj_loss), total_loss / total_sums if total_sums > 0 else -1,
                float(init_loss), total_l0 / total_sums if total_sums > 0 else -1,
                float(updated_loss), total_l1 / total_sums if total_sums > 0 else -1)

            for key, val in stat_dict.items():
                print_str += "{}: {:.2e}\t".format(key, val)
            print(print_str)

        if total_iters == gradinit_iters:
            break
    recover_bn_modes(model)