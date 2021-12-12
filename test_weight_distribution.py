import os
import data
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn

from neural_network import mlp_network, resnet, vgg, preact_resnet


parser = argparse.ArgumentParser(description='SFW DNN Training')
################################ basic settings ################################
parser.add_argument('--data', default='cifar10', type=str, help='type of dataset (default: cifar10)')
parser.add_argument('--arch', default='ResNet18', type=str, help='model architecture (default: resnet18)')
parser.add_argument('--optimizer', default='SFW', type=str, help='optimizer to train the model (default: SFW)')
parser.add_argument('--constraint', default='k_sparse_constraints', type=str, help='model architecture (default: k_sparse_constraints)')

################################ SFW settings ################################
parser.add_argument('--lr', default=1.0, type=float, help='initial learning rate (default: 1.0)')
parser.add_argument('--lr_scheme', default='dynamic_change', type=str, help='learning rate changing scheme (default: dynamically change per 5 epochs')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum parameter (default: 0.9)')
parser.add_argument('--weight_decay', default=0, type=float, help='weight decay parameter (default: 0.0)')
parser.add_argument('--k_sparseness', default=10, type=int, help='K in K-sparse polytope constraint (default: 10)')
parser.add_argument('--k_frac', default=0.05, type=float, help='K fraction in K-sparse polytope constraint (default: 5%)')
parser.add_argument('--tau', default=15, type=int, help='diameter parameter of K-sparse polytope constraint (default: 15)')
parser.add_argument('--mode', default='initialization', type=str, help='rescale method of constraint diamete (default: initialization)')
parser.add_argument('--rescale', default='gradient', type=str, help='rescale method of learning rate (default: gradient)')
parser.add_argument('--sfw_init', default=0, type=int, help='whether use SFW_Init scheme (default: 0)')

################################ other settings ################################
parser.add_argument('--train_batchsize', default=128, type=int, help='train batchsize')
parser.add_argument('--test_batchsize', default=128, type=int, help='test batchsize')
parser.add_argument('--epoch_num', default=180, type=int, help='number of training epochs (default: 180)')
parser.add_argument('--color_channel', default=3, type=int, help='number of color channels (default: 3)')
parser.add_argument('--gpu', default=-1, type=int, help='GPU id, -1 for CPU')


def load_data(args):
    if args.data == 'cifar10':
        train_data, test_data = data.load_cifar10_data(
            args.dir_path, args.train_batchsize, args.test_batchsize)
    elif args.data =='mnist':
        train_data, test_data = data.load_mnist_data(
            args.dir_path, args.train_batchsize, args.test_batchsize)
    elif args.data == 'cifar100':
        train_data, test_data = data.load_cifar100_data(    
            args.dir_path, args.train_batchsize, args.test_batchsize)
    elif args.data == 'svhn':
        train_data, test_data = data.load_svhn_data(
            args.dir_path, args.train_batchsize, args.train_batchsize)
    elif args.data == 'tiny':
        train_data, test_data = data.tiny_loader(
            args.dir_path + '/data/tiny_imagenet_200', args.train_batchsize, args.train_batchsize)
    else:
        print('wrong data option')
        train_data, test_data = None
    return train_data, test_data


def build_model(args):
    # define model 
    if args.arch == 'ResNet18': 
        if args.data == 'cifar100':
            model = resnet.ResNet18(color_channel=args.color_channel, num_classes=100)
        if args.data == 'cifar10':
            model = resnet.ResNet18(color_channel=args.color_channel, num_classes=10)
        if args.data == 'svhn':
            model = resnet.ResNet18(color_channel=args.color_channel, num_classes=10)
        if args.data == 'tiny':
            model = resnet.ResNet18(color_channel=args.color_channel, num_classes=200)
    elif args.arch == 'VGG16':
        if args.data == 'cifar100':
            model = vgg.VGG16(color_channel=args.color_channel, num_classes=100)
        if args.data == 'cifar10':
            model = vgg.VGG16(color_channel=args.color_channel, num_classes=10)
        if args.data == 'svhn':
            model = vgg.VGG16(color_channel=args.color_channel, num_classes=10)
        if args.data == 'tiny':
            model = vgg.VGG16(color_channel=args.color_channel, num_classes=200)
    elif args.arch == 'Mlp': # only for mnist
        model = mlp_network.MlpNetwork(input_size=784, output_size=10)
    else:
        print('wrong model option')
        model = None
    
    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # load parameters
    model.load_state_dict(torch.load(model_path))

    return model, loss_function


def print_model_parameters(model, weight_threshold_distribution):
    # print number of non-zero parameters
    count_active_weights = 0
    sum_params = 0
    for params in model.parameters():
        sum_params += params.numel()
        temp = torch.abs(params).detach().cpu().numpy()
        count_active_weights += np.size(temp[temp>weight_threshold_distribution])
    print('total:', sum_params, 'threshold:', weight_threshold_distribution, 'activated:', count_active_weights) 
    return count_active_weights


def run_weight_distribution():
    args = parser.parse_args()
    args.device = torch.device(
        'cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu'
    )
    args.dir_path = os.getcwd()    

    # initialize logger
    logger = logging.getLogger(args.data + '_' + args.arch + '_' + args.optimizer + '_' + args.constraint)
    logger.setLevel(logging.INFO)
    logger_dir = args.dir_path + '/saved_logs/weight_distribution/'
    logger_dir += "data=" + args.data + "_" + "model=" + args.arch + "_" + \
                "optimizer=" + args.optimizer + "_" + "constraint=" + args.constraint + "_" + \
                "learning_rate=" + str(args.lr) + "_" + "learning_rate_scheme=" + args.lr_scheme + "_" + \
                "momentum=" + str(args.momentum) + "_" + "weight_decay=" + str(args.weight_decay) + "_" + \
                "k_sparseness=" + str(args.k_sparseness) + "_" + "k_frac=" + str(args.k_frac) + "_" + \
                "tau=" + str(args.tau) + "_" + "mode=" + args.mode + "_" + \
                "rescale=" + args.rescale + "sfw_init=" + str(args.sfw_init)
    logger_dir += '.log'
    logger_handler = logging.FileHandler(logger_dir)
    logger.addHandler(logger_handler)

    # log configuration
    logger.info('Configuration' + " " + 
                'Data:' + " " + args.data + " " +
                'Model:' + " " + args.arch + " " +
                'Optimizer:' + " " + args.optimizer + " " +
                'Constraint:' + " " + args.constraint + " " +
                'Train_Batchsize:' + " " + str(args.train_batchsize) + " " +
                'Test_Batchsize:' + " " + str(args.test_batchsize) + " " +
                'Epoch_Number:' + " " + str(args.epoch_num) + " " +
                'Learning_Rate:' + " " + str(args.lr) + " " +
                'Learning_Rate_Scheme:' + " " + str(args.lr_scheme) + " " +
                'Momentum:' + " " + str(args.momentum) + " " +
                'Weight_Decay:' + " " + str(args.weight_decay) + " " +
                'Color_Channel:' + " " + str(args.color_channel) + " " +
                'K_sparseness:' + " " + str(args.k_sparseness) + " " +
                'K_frac:' + " " + str(args.k_frac) + " " +
                'Tau:' + " " + str(args.tau) + " " +
                'Mode:' + " " + args.mode + " " +
                'Rescale:' + " " + args.rescale + " " +
                'SFW_Init' + " " + str(args.sfw_init))
    
    # model path
    model_path = args.dir_path + '/saved_models/' \
            + 'data-' + args.data + '_' + 'model-' + args.arch + '_' \
            + 'optimizer-' + args.optimizer + '_' + 'constraints-' + args.constraint + '_' \
            + 'epoch_num-' + str(args.epoch_num) + '_' + 'lr-' + str(args.lr) + '_'\
            + 'lr_scheme-' + args.lr_scheme + '_' + 'momentum-' + str(args.momentum) + '_'\
            + 'weight_decay-' + str(args.weight_decay) + '_' + 'color_channel-' + str(args.color_channel) + '_'\
            + 'k_sparseness-' + str(args.k_sparseness) + '_' + 'k_frac-' + str(args.k_frac) + '_'\
            + 'tau-' + str(args.tau)  + '_' + 'sfw_init-' + str(args.sfw_init) + '.pt'


    print('-------build model-------')
    model, loss_function = load_model(args, model_path)
    model.to(args.device)

    params_sum = print_model_parameters(model, 0)
    logger.info('Total parameters' + " " + str(params_sum))

    weight_threshold_distribution = 10
    weight_count_array = []

    while weight_threshold_distribution > 1e-10:
        count_active_weights = print_model_parameters(model, weight_threshold_distribution)
        logger.info('Number of weights' + " " + ">" + str(weight_threshold_distribution) + ": " + str(count_active_weights))
        weight_count_array.append(count_active_weights)
        weight_threshold_distribution = weight_threshold_distribution/10
    
    weight_count_density = np.zeros(len(weight_count_array)+1)
    weight_count_density[0] = weight_count_array[0]
    for i in range(1,len(weight_count_array)):
        weight_count_density[i] = weight_count_array[i] - weight_count_array[i-1]
    weight_count_density[-1] = params_sum - weight_count_array[-1]
    print('weight_density:', weight_count_density)
    logger.info('Weight distribution density:' + str(weight_count_density))


if __name__ == '__main__':
    run_weight_distribution()
   

