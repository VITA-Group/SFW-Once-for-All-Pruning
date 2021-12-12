import os
import data
import time
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from neural_network import mlp_network, resnet, vgg, preact_resnet
from frankwolfe.pytorch import optimizers, constraints
from utils.gradient_utils import gradinit_sfw


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
    elif args.arch == 'VGG16':
        if args.data == 'cifar100':
            model = vgg.VGG16(color_channel=args.color_channel, num_classes=100)
        if args.data == 'cifar10':
            model = vgg.VGG16(color_channel=args.color_channel, num_classes=10)
        if args.data == 'svhn':
            model = preact_resnet.PreActResNet18(color_channel=args.color_channel, num_classes=10)
    elif args.arch == 'PreActResNet18':
        if args.data == 'cifar100':
            model = preact_resnet.PreActResNet18(color_channel=args.color_channel, num_classes=100)
        if args.data == 'cifar10':
            model = preact_resnet.PreActResNet18(color_channel=args.color_channel, num_classes=10)
        if args.data == 'svhn':
            model = preact_resnet.PreActResNet18(color_channel=args.color_channel, num_classes=10)
    elif args.arch == 'Mlp':
        model = mlp_network.MlpNetwork(input_size=784, output_size=10)
    else:
        print('wrong model option')
        model = None
    
    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # define constraints 
    if args.constraint == 'l2_constraints':
        constraints_list = constraints.create_lp_constraints(model, ord=2, value=args.tau, mode=args.mode)
    elif args.constraint == 'k_sparse_constraints':
        constraints_list = constraints.create_k_sparse_constraints(model, 
        K=args.k_sparseness, K_frac=args.k_frac, value=args.tau, mode=args.mode)
    elif args.constraint == 'unconstraints':
        constraints_list = constraints.create_unconstraints(model)
    else:
        print('wrong constraints option')
        optimizer = None
    constraints.make_feasible(model, constraints_list)

    # define optimizer
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,  
        momentum=args.momentum, weight_decay = args.weight_decay)
    elif args.optimizer == 'SFW':
        optimizer = optimizers.SFW(
            model.parameters(), learning_rate=args.lr,  
            momentum=args.momentum, rescale=args.rescale)
    else:
        print('wrong optimizer option')
        optimizer = None

    return model, loss_function, constraints_list, optimizer


def sfw_gradinit(model, trainloader, constraint, config, make_feasible=False):    
    gradinit_sfw(model, trainloader, constraint, config)
    if make_feasible:
        constraints.make_feasible(model, constraint)


def print_model_parameters(model, weight_threshold):

    # print number of non-zero parameters given a threshold
    count_active_weights = 0
    sum_params = 0
    for params in model.parameters():
        sum_params += params.numel()
        temp = torch.abs(params).detach().cpu().numpy()
        count_active_weights += np.size(temp[temp>weight_threshold])
    print('total:', sum_params, 'threshold:', weight_threshold, 'activated:', count_active_weights) 
    return count_active_weights


def train_batch(trainloader, testloader, model, loss_function, constraints_list, optimizer, args, logger):

    # go through epochs
    loss_list = []
    for epoch in range(args.epoch_num):
        print('=======Epoch=======', epoch + 1)
        model.train()

        # learning rate decay
        if args.lr_scheme == 'keep':
            pass
        elif args.lr_scheme == 'decrease_3':
            if epoch == int(args.epoch_num / 3):
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] / 10
                print('divide current learning rate by 10', '\n','Current learning rate:', g['lr'] )
            elif epoch == int(args.epoch_num * 2 / 3):
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] / 10
                print('divide current learning rate by 10', '\n','Current learning rate:',g['lr'] )
        elif args.lr_scheme == 'decrease_3_180':
            if epoch == 90:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] / 10
                print('divide current learning rate by 10', '\n','Current learning rate:', g['lr'] )
            elif epoch == 130:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] / 10
                print('divide current learning rate by 10', '\n','Current learning rate:',g['lr'] )
        elif args.lr_scheme ==  'dynamic_change':
            if epoch == int(args.epoch_num / 3):
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] / 10
                print('divide current learning rate by 10', '\n','Current learning rate:', g['lr'] )
            elif epoch == int(args.epoch_num * 2 / 3):
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] / 10
                print('divide current learning rate by 10', '\n','Current learning rate:',g['lr'] )
            if epoch > 20 and epoch%5 == 0:
                loss_list_5epoch = loss_list[(epoch-6):(epoch-1)]
                loss_list_10epoch = loss_list[(epoch-11):(epoch-1)]
                avg_loss_5epoch = np.mean(loss_list_5epoch)
                avg_loss_10epoch = np.mean(loss_list_10epoch)
                if avg_loss_5epoch > avg_loss_10epoch:
                    for g in optimizer.param_groups:
                        g['lr'] =  g['lr'] * 0.7
                    print('multiply current learning rate by 0.7', '\n','Current learning rate: ',g['lr'])
                if avg_loss_5epoch < avg_loss_10epoch:
                    for g in optimizer.param_groups:
                        g['lr'] =  g['lr'] * 1.06
                    print('multiply current learning rate by 1.06', '\n','Current learning rate: ',g['lr'])

        # train
        print('------training------')
        for steps, (x_batch, y_batch) in enumerate(trainloader):
            
            # preparing training data
            loss_steps_list = []
            length = len(trainloader)
            x_batch, y_batch = x_batch.to(args.device), y_batch.to(args.device)
            if args.data == 'mnist' and args.arch == 'Mlp':
                x_batch = x_batch.reshape(-1,28*28) 
            optimizer.zero_grad()

            # forward and backward
            outputs = model(x_batch)
            loss = loss_function(outputs, y_batch)
            loss.backward()
            if args.optimizer == 'SGD':
                optimizer.step()
            elif args.optimizer == 'SFW':
                optimizer.step(constraints_list)
            else:
                break
            
            # print loss and accuracy of a batch
            _, predicted = torch.max(outputs.data, 1)
            correct = predicted.eq(y_batch.data).sum()
            accuracy = 100. * correct / len(x_batch)
            print('epoch:', epoch + 1, 'step:', steps + 1 + epoch * length, 
                        'batch_loss:', loss.item(), 'batch_accuracy:', accuracy)
            loss_steps_list.append(loss.item())

            # logging
            logger.info('Train' + " " + 
                'Steps:' + " " + str(steps + 1 + epoch * length) + " " + 
                'Epoch:' + " " + str(epoch + 1) + " " + 
                'Batch_Loss:' + " " + str(loss.item()) + " " +
                'Batch_Accuracy' + " " + str(accuracy.item()))

        print('------testing------')
        with torch.no_grad():
            model.eval()

            # test accuracy
            correct = 0
            total = 0
            for x_batch, y_batch in testloader:
                x_batch, y_batch = x_batch.to(args.device), y_batch.to(args.device)
                if args.data == 'mnist' and args.arch == 'Mlp':
                    x_batch = x_batch.reshape(-1,28*28) 
                outputs = model(x_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum()
                accuracy = 100. * correct / total
            print("Accuracy:", 100. * correct /total)
            print("Current time: ", time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
            logger.info('Test' + " " + 
                'Epoch:' + " " + str(epoch + 1) + " " + 
                'Test_Accuracy:' + " " + str(accuracy.item()) + " " +
                'Current_Time:' + str(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())))

            # weight distribution
            weight_threshold = 10
            while weight_threshold > 1e-10:
                count_active_weights = print_model_parameters(model, weight_threshold)
                logger.info('Weight_Distribution' + " " + ">" + str(weight_threshold) + " " + str(count_active_weights))
                weight_threshold = weight_threshold/10

        epoch_mean_loss = np.mean(loss_steps_list)
        loss_list.append(epoch_mean_loss)


def run_train_models():
    args = parser.parse_args()
    args.device = torch.device(
        'cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu'
    )
    args.dir_path = os.getcwd()

    # initialize logger
    logger = logging.getLogger(args.data + '_' + args.arch + '_' + args.optimizer + '_' + args.constraint)
    logger.setLevel(logging.INFO)
    logger_dir = args.dir_path + '/saved_logs/SFW_one_shot_prune/'
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

    print('-------load data-------') 
    train_data, test_data = load_data(args)

    print('-------build model-------')
    model, loss_function, constraints_list, optimizer = build_model(args)
    model.to(args.device)
    if args.sfw_init == 1:
        sfw_gradinit(model, train_data, constraints_list, args)
    

    print('-------train model-------')
    train_batch(train_data, test_data, model, loss_function, constraints_list, optimizer, args, logger)

    print('-------save model-------')
    torch.save(model.state_dict(), model_path) 


if __name__ == '__main__':
    run_train_models()
