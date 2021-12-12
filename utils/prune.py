import torch
import copy 
import torch.nn as nn
from torch.nn.modules.conv import Conv2d
import torch.nn.utils.prune as prune
import numpy as np


def layer_prune(config, model):
    with torch.no_grad():
        for params in model.parameters():
            n = params.numel()
            temp = params.reshape([1, n])
            temp = torch.abs(temp)
            temp = temp.sort()[0]
            prune_threshold = temp[0][int((1-config['prune_frac'])*(n-1))]
            params[torch.abs(params)<prune_threshold] = 0


def global_prune(config, model):
    global_params = torch.zeros([1,1])
    sum_params = 0
    with torch.no_grad():
        for params in model.parameters():
            n = params.numel()
            sum_params += n
            temp = params.reshape([1, n])
            global_params = torch.cat([global_params, temp], axis=1)
        global_params = torch.abs(global_params)
        global_params = global_params.sort()[0]
        global_prune_threshold = global_params[0][int((1-config['prune_frac'])*(sum_params))]
        for params in model.parameters():
            params[torch.abs(params)<global_prune_threshold] = 0
        

def global_prune_distribution(model, prune_frac_distribution):
    global_params = torch.zeros([1,1])
    sum_params = 0
    with torch.no_grad():
        for params in model.parameters():
            n = params.numel()
            sum_params += n
            temp = params.reshape([1, n])
            global_params = torch.cat([global_params, temp], axis=1)
        global_params = torch.abs(global_params)
        global_params = global_params.sort()[0]
        global_prune_threshold = global_params[0][int((1-prune_frac_distribution)*(sum_params))]
        for params in model.parameters():
            params[torch.abs(params)<global_prune_threshold] = 0


def mask_vector(model, prune_frac):
    mask_vector = torch.zeros([1,1])
    sum_params = 0
    with torch.no_grad():
        for params in model.parameters():
            n = params.numel()
            sum_params += n
            temp = params.reshape([1, n])
            mask_vector = torch.cat([mask_vector, temp], axis=1)
        mask_vector_temp = torch.abs(mask_vector)
        mask_vector_temp = mask_vector_temp.sort()[0]
        prune_threshold = mask_vector_temp[0][int((1-prune_frac)*(sum_params))]
        print(str(prune_frac)+' threshold:', prune_threshold)
        mask_vector[torch.abs(mask_vector)<prune_threshold] = 0
        mask_vector[torch.abs(mask_vector)>=prune_threshold] = 1
    return mask_vector


__all__  = ['pruning_model', 'pruning_model_random', 'prune_model_custom', 'remove_prune',
            'extract_mask', 'reverse_mask', 'check_sparsity', 'check_sparsity_dict']


# Pruning operation
def prune_structured_l1(model, ratio,  module_name):
    for name, m in model.named_modules():
        if name == module_name and isinstance(m, nn.Conv2d):
            prune.ln_structured(m, 'weight', amount=ratio, n=1, dim=1)

def prune_structured_l1_erk(model, ratio):
    sum_list = 0
    total_param_num = 0
    layer_param_num =[]
    s_yiwan_array =[]
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            # parameters_to_prune.append((m,'weight'))
            total_param_num =  total_param_num +float(m.weight.nelement())
            layer_param_num.append(float(m.weight.nelement()))
            sum_list = sum_list + 1
            s_yiwan_array.append(1-(m.in_channels + m.out_channels + m.kernel_size[0] + m.kernel_size[1]) / (m.in_channels * m.out_channels * m.kernel_size[0] * m.kernel_size[1]))
    coeff = ratio * total_param_num / (np.sum(np.array(layer_param_num) * np.array(s_yiwan_array)))
    layer_ratio = coeff * np.array(s_yiwan_array)
    idx = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.ln_structured(m, 'weight', amount=layer_ratio[idx], n=1, dim=1)
            idx = idx + 1
            

def pruning_model(model, px):

    print('Apply Unstructured L1 Pruning Globally (all conv layers)')
    parameters_to_prune =[]
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            parameters_to_prune.append((m,'weight'))

    parameters_to_prune = tuple(parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )


def pruning_model_random(model, px):

    print('Apply Unstructured Random Pruning Globally (all conv layers)')
    parameters_to_prune =[]
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            parameters_to_prune.append((m,'weight'))

    parameters_to_prune = tuple(parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.RandomUnstructured,
        amount=px,
    )


def prune_model_custom(model, mask_dict):

    print('Pruning with custom mask (all conv layers)')
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            mask_name = name+'.weight_mask'
            if mask_name in mask_dict.keys():
                prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'])
            else:
                print('Can not fing [{}] in mask_dict'.format(mask_name))


def remove_prune(model):
    
    print('Remove hooks for multiplying masks (all conv layers)')
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.remove(m,'weight')


# Mask operation function
def extract_mask(model_dict):

    new_dict = {}
    for key in model_dict.keys():
        if 'mask' in key:
            new_dict[key] = copy.deepcopy(model_dict[key])

    return new_dict


def reverse_mask(mask_dict):

    new_dict = {}
    for key in mask_dict.keys():

        new_dict[key] = 1 - mask_dict[key]

    return new_dict


# Mask statistic function
def check_sparsity(model):
    
    sum_list = 0
    zero_sum = 0

    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            sum_list = sum_list+float(m.weight.nelement())
            zero_sum = zero_sum+float(torch.sum(m.weight == 0))  

    if zero_sum:
        remain_weight_ratie = 100*(1-zero_sum/sum_list)
        print(zero_sum)
        print(sum_list)
        print('* remain weight ratio = ', 100*(1-zero_sum/sum_list),'%')
    else:
        print('no weight for calculating sparsity')
        remain_weight_ratie = None

    return remain_weight_ratie


def check_sparsity_dict(state_dict):
    
    sum_list = 0
    zero_sum = 0

    for key in state_dict.keys():
        if 'mask' in key:
            sum_list += float(state_dict[key].nelement())
            zero_sum += float(torch.sum(m.weight == 0))  

    if zero_sum:
        remain_weight_ratie = 100*(1-zero_sum/sum_list)
        print('* remain weight ratio = ', 100*(1-zero_sum/sum_list),'%')
    else:
        print('no weight for calculating sparsity')
        remain_weight_ratie = None

    return remain_weight_ratie


