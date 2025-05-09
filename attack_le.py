import copy

import torch
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributed import rpc
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import datasets, transforms
import pickle
import PIL.Image as Image
import os
from tqdm import tqdm
import argparse
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from  torch.distributed import rpc
import torch.multiprocessing as mp

import sys
sys.path.append(".")
from cfg import results_path
from tools import *
from algorithm.prune import global_prune, check_sparsity, extract_mask, custom_prune, remove_prune
from algorithm.zoo import cge_weight_allocate_to_process, cge_calculation, network_synchronize
from copy import deepcopy
from data import prepare_dataset
from models.tools import time_consumption_per_layer
from models.distributed_model import DistributedCGEModel

# run command on first session in terminal
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# numpy 1.26.4

class Args:
    def __init__(self, p, mu):
        self.sparsity_folder = "Layer_Sparsity"
        self.network = "lenet" # lenet, resnet20
        self.zero = True
        self.sparsity = p # p
        self.sparsity_ckpt = f"zo_grasp_{self.sparsity}"
        self.lr = 0.1
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.nesterov = True
        self.epoch = 3
        self.warmup_epochs = 3
        self.scheduler = "cosine"
        self.gpus = [0]
        self.process_per_gpu = 2
        self.zoo_step_size = mu # mu
        self.master_port = "29600"
        self.master_addr = "localhost"
        self.score = "layer_wise_random"
        self.mask_shuffle_interval = 5
        self.log = True


# mew: smootheness = cgs_step_size
# mew: cge_estimate
# step size:
# p: precentile start with higher value, will decrease privacy
# find the starting value of p
# high p

def cge_cal_and_return(remote_networks, network, gpus, process_per_gpu, x, y, cge_step_size):
    # cge_step_size = args.zoo_step_size
    params_dict = {
        name: p for name, p in network.named_parameters() if p.requires_grad
    }
    device = next(network.parameters()).device
    x_rref, y_rref = rpc.RRef(x), rpc.RRef(y)
    grads_signal = []
    for gpu in gpus:
        for i in range(process_per_gpu):
            grads_signal.append(remote_networks[f"{gpu}-{i}"].rpc_async(timeout=0).calculate_grads(x_rref, y_rref, cge_step_size))
    grads = []
    for g in grads_signal:
        grads.append(g.wait())
    grads = torch.cat(grads, dim=0).to(device)

    # dlg_grads = {} # gradients for dlg attack
    grad_list = []
    for name_id, (name, param) in enumerate(params_dict.items()):
        param.grad = torch.zeros_like(param)
        grads_indices_and_values = grads[grads[:, 0]==name_id, 1:]
        param_grad_flat = param.grad.flatten()
        param_grad_flat[grads_indices_and_values[:, 0].long()] = grads_indices_and_values[:, 1]

        # dlg_grads[name] = param.grad.clone()
        grad_list.append(param.grad.clone())
    # return dlg_grads

    return grad_list

@torch.no_grad()
def cge(func, params_dict, mask_dict, step_size, net, gt_data, gt_label, loss_func, base=None):
    if base is None:
        # base = func(params_dict, net, gt_data, gt_label)
        base = func(params_dict, net, gt_data, gt_label, loss_func)
    grads_dict = {}
    for key, param in params_dict.items():
        if 'orig' in key:
            mask_key = key.replace('orig', 'mask')
            mask_flat = mask_dict[mask_key].flatten()
        else:
            mask_flat = torch.ones_like(param).flatten()
        directional_derivative = torch.zeros_like(param)
        directional_derivative_flat = directional_derivative.flatten()
        for idx in mask_flat.nonzero().flatten():
            perturbed_params_dict = deepcopy(params_dict)
            p_flat = perturbed_params_dict[key].flatten()
            p_flat[idx] += step_size
            # directional_derivative_flat[idx] = (func(perturbed_params_dict, net, gt_data, gt_label) - base) / step_size
            directional_derivative_flat[idx] = (func(perturbed_params_dict, net, gt_data, gt_label, loss_func) - base) / step_size
        grads_dict[key] = directional_derivative.to(param.device)
    return list(grads_dict.values())

@torch.no_grad()
def f(params_dict, network, x, y, loss_func):
    state_dict_backup = network.state_dict()
    network.load_state_dict(params_dict, strict=False)
    loss = loss_func(network(x), y).detach().item()
    network.load_state_dict(state_dict_backup)
    return loss

@torch.no_grad()
def loss_func(params_dict, net, gt_data, gt_label):
    with torch.no_grad():
        for name, param in net.named_parameters():
            param.copy_(params_dict[name])
    out = net(gt_data)
    loss = F.cross_entropy(out, gt_label)
    return loss

def init_model(device, args, class_num, hidden, channel):
    # class_num = 10 # cifar10
    # hidden = 588
    # channel = 1
    if args.network == "resnet20":
        from models.resnet_s import resnet20, param_name_to_module_id_rn20
        # param_name_to_module_id = param_name_to_module_id_rn20
        network_init_func = resnet20
        network_kwargs = {
            'channel': channel,
            'num_classes': class_num
        }

    elif args.network == "lenet": # lenet
        from models.lenet import lenet, param_name_to_module_id_lenet
        network_init_func = lenet
        network_kwargs = {
            'channel': channel,
            'hidden': hidden,
            'num_classes': class_num
        }
    else:
        raise NotImplementedError
    net = network_init_func(**network_kwargs).to(device)
    net.apply(weights_init).to(device)
    # net = network_init_func(**network_kwargs)

    # Optimizer
    # optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    #                             momentum=args.momentum, nesterov=args.nesterov)

    net.train()
    return net

def net_prep(net, args):
    class_num = 10
    hidden = 588
    channel = 1
    # prep
    if args.network == "resnet20":
        from models.resnet_s import resnet20, param_name_to_module_id_rn20
        network_init_func = resnet20
        param_name_to_module_id = param_name_to_module_id_rn20
    elif args.network == "lenet":
        from models.lenet import lenet, param_name_to_module_id_lenet
        network_init_func = lenet
        param_name_to_module_id = param_name_to_module_id_lenet
    else:
        raise NotImplementedError
    network_kwargs = {
        'channel': channel,
        'hidden': hidden,
        'num_classes': class_num  # cifar10
    }
    useRemoteNet = False
    remote_networks = {}
    if useRemoteNet:
        os.makedirs('.cache', exist_ok=True)
        cache_file_path = f'.cache/pruned_model_{args.master_port}.pth'
        torch.save(net.state_dict(), cache_file_path)
        for gpu in args.gpus:  # gpus
            for i in range(args.process_per_gpu):
                remote_networks[f"{gpu}-{i}"] = rpc.remote(f"{gpu}-{i}", DistributedCGEModel,
                                                           args=(f"cuda:{gpu}",
                                                                 partial(network_init_func, **network_kwargs),
                                                                 F.cross_entropy, param_name_to_module_id,
                                                                 cache_file_path, False))

    # ReGenerate Mask
    sparsity_ckpt = torch.load(os.path.join(args.sparsity_folder, args.network, args.sparsity_ckpt + '.pth'),
                               map_location=f"cuda:{args.gpus[-1]}") if args.sparsity_ckpt is not None else None
    state_dict_to_restore = net.state_dict()
    if 0. < args.sparsity < 1.:
        global_prune(net, args.sparsity, args.score, class_num, None, zoo_sample_size=192,
                     zoo_step_size=5e-3, layer_wise_sparsity=sparsity_ckpt)
    elif args.sparsity == 0:
        pass
    else:
        raise ValueError('sparsity not valid')
    assert abs(args.sparsity - (1 - check_sparsity(net, if_print=False) / 100)) < 0.01, check_sparsity(net,
                                                                                                       if_print=False)
    current_mask = extract_mask(net.state_dict())

    # gets error can not find [layer_._.conv_.weight_mask] in mask_dict, skipping
    if useRemoteNet:
        cge_weight_allocate_to_process(remote_networks, net, args.gpus, args.process_per_gpu,
                                       param_name_to_module_id,
                                       time_consumption_per_layer(args.network))
    remove_prune(net)
    net.load_state_dict(state_dict_to_restore)
    return net, remote_networks

def weights_init(m):
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())

def prep_data(gt_data, gt_label):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2471, 0.2435, 0.2616]
    transform = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    gt_data = transform(gt_data)
    data = TensorDataset(gt_data, gt_label)
    data_loader = DataLoader(data, batch_size=1 , shuffle=True, num_workers=0, pin_memory=False)
    return {
        "train": data_loader,
        "test": data_loader,
    }, 10

def train(network, args, remote_networks, gt_data, gt_label, save_path):
    logger = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    # dataset = data.TensorDataset(
    #     gt_data.unsqueeze(0),  # Add batch dimension
    #     gt_label.unsqueeze(0)  # Add batch dimension
    # )
    # # loaders = DataLoader(dataset, batch_size=1, shuffle=False)
    from models.resnet_s import resnet20, param_name_to_module_id_rn20
    param_name_to_module_id = param_name_to_module_id_rn20

    device = f"cuda:{args.gpus[-1]}"
    loaders, class_num = prep_data(gt_data, gt_label)
    sparsity_ckpt = torch.load(os.path.join(args.sparsity_folder, args.network, args.sparsity_ckpt + '.pth'),
                               map_location=device) if args.sparsity_ckpt is not None else None
    optimizer = torch.optim.SGD(network.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                momentum=args.momentum, nesterov=args.nesterov)

    epoch = 0
    while epoch < args.epoch:
        epoch += 1
        if (epoch-1) % args.mask_shuffle_interval == 0:
            # ReGenerate Mask
            state_dict_to_restore = network.state_dict()
            if 0. < args.sparsity < 1.:
                global_prune(network, args.sparsity, args.score, class_num, loaders['train'], zoo_sample_size=192, zoo_step_size=5e-3, layer_wise_sparsity=sparsity_ckpt)
            elif args.sparsity == 0:
                pass
            else:
                raise ValueError('sparsity not valid')
            assert abs(args.sparsity - (1 - check_sparsity(network, if_print=False) / 100)) < 0.01, check_sparsity(network, if_print=False)
            current_mask = extract_mask(network.state_dict())
            cge_weight_allocate_to_process(remote_networks, network, args.gpus, args.process_per_gpu, param_name_to_module_id, time_consumption_per_layer(args.network))
            remove_prune(network)
            network.load_state_dict(state_dict_to_restore)
        # Train
        network.train()
        acc = AverageMeter()
        loss = AverageMeter()
        pbar = tqdm(loaders['train'], total=len(loaders['train']),
                desc=f"Epo {epoch} Training", ncols=160)
        for i, (x, y) in enumerate(pbar):
            # need to move to cpu
            x = x.cpu()
            y = y.cpu()
            if epoch <= args.warmup_epochs:
                warmup_lr(optimizer, epoch-1, i+1, len(loaders['train']), args.warmup_epochs, args.lr)
            x_cuda, y_cuda = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                fx = network(x_cuda, return_interval = False)
                loss_batch = F.cross_entropy(fx, y_cuda).cpu()
            lr = optimizer.param_groups[0]['lr']
            cge_calculation(remote_networks, network, args.gpus, args.process_per_gpu, x, y, lr if args.zoo_step_size == -1 else args.zoo_step_size)
            optimizer.step()
            network_synchronize(remote_networks, network, args.gpus, args.process_per_gpu)
            acc.update(torch.argmax(fx, 1).eq(y_cuda).float().mean().item(), y.size(0))
            loss.update(loss_batch.item(), y.size(0))
            if epoch > args.warmup_epochs:
                scheduler.step()
            pbar.set_postfix_str(f"Lr {lr:.2e} Acc {100*acc.avg:.2f}%")
        if args.log:
            logger.add_scalar("train/acc", acc.avg, epoch)
            logger.add_scalar("train/loss", loss.avg, epoch)

        # Test
        network.eval()
        pbar = tqdm(loaders['test'], total=len(loaders['test']), desc=f"Epo {epoch} Testing", ncols=120)
        acc = AverageMeter()
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                fx = network(x)
            acc.update(torch.argmax(fx, 1).eq(y).float().mean(), y.size(0))
            pbar.set_postfix_str(f"Acc {100*acc.avg:.2f}%")
        if args.log:
            logger.add_scalar("test/acc", acc.avg, epoch)
    return network

def main(args):
    dataset = 'MNIST'
    root_path = '.'
    data_path = os.path.join(root_path, '../data').replace('\\', '/')
    # save_path = os.path.join(root_path, f'results/iDLG_lenet_{dataset}/iDLG_{dataset}_p{args.sparsity}_mu{args.zoo_step_size}').replace('\\', '/')
    save_path = os.path.join(root_path, f'results/iDLG_lenet_{dataset}/iDLG_{dataset}').replace('\\', '/')
    result_path = os.path.join(root_path, f"results/iDLG_lenet_{dataset}/results.csv")

    lr = 1
    num_dummy = 1
    Iteration = 300
    num_exp = 200

    use_cuda = torch.cuda.is_available()
    device = f'cuda:{args.gpus[0]}' if use_cuda else 'cpu'
    # device = "cpu"

    tt = transforms.Compose([transforms.ToTensor()])
    tp = transforms.Compose([transforms.ToPILImage()])

    print(dataset, 'root_path:', root_path)
    print(dataset, 'data_path:', data_path)
    print(dataset, 'save_path:', save_path)

    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    ''' load data '''
    if dataset == 'MNIST':
        shape_img = (28, 28)
        num_classes = 10
        channel = 1
        hidden = 588
        dst = datasets.MNIST(data_path, download=False)

    elif dataset == 'cifar100':
        shape_img = (32, 32)
        num_classes = 100
        channel = 3
        hidden = 768
        dst = datasets.CIFAR100(data_path, download=True)

    elif dataset == 'cifar10':
        shape_img = (32, 32)
        num_classes = 10
        channel = 3
        hidden = 768
        dst = datasets.CIFAR10(data_path, download=True)

    elif dataset == 'lfw':
        shape_img = (32, 32)
        num_classes = 5749
        channel = 3
        hidden = 768
        lfw_path = os.path.join(root_path, '../data/lfw')
        dst = lfw_dataset(lfw_path, shape_img)

    else:
        exit('unknown dataset')

    # num_exp = len(dst)
    idx_shuffle = np.random.default_rng(123).permutation(len(dst))
    ''' train DLG and iDLG '''
    for idx_net in range(0, num_exp):
        print("init model")
        net = init_model(device, args, num_classes, hidden, channel)
        print("done init model")

        print('running %d|%d experiment' % (idx_net, num_exp))
        net = net.to(device)
        # idx_shuffle = np.arange(len(dst))
        # for method in ['DLG', 'iDLG']:
        for method in ['iDLG']:
            print('%s, Try to generate %d images' % (method, num_dummy))

            criterion = nn.CrossEntropyLoss().to(device)
            imidx_list = []

            for imidx in range(num_dummy):
                idx = idx_shuffle[idx_net]
                imidx_list.append(idx)
                tmp_datum = tt(dst[idx][0]).float().to(device)
                # tmp_datum = tt(dst[idx][0]).float()
                tmp_datum = tmp_datum.view(1, *tmp_datum.size())
                tmp_label = torch.Tensor([dst[idx][1]]).long().to(device)
                # tmp_label = torch.Tensor([dst[idx][1]]).long()
                tmp_label = tmp_label.view(1, )
                if imidx == 0:
                    gt_data = tmp_datum
                    gt_label = tmp_label
                else:
                    gt_data = torch.cat((gt_data, tmp_datum), dim=0)
                    gt_label = torch.cat((gt_label, tmp_label), dim=0)
                gt_data_cpu = gt_data.to("cpu")
                gt_label_cpu = tmp_label.to("cpu")

            # print("prep model")
            # if args.zero:
            #     net, remote_networks = net_prep(net, args)
            # print("done prep model")

            # print("Training model")
            # # train model
            # net = train(net, args, remote_networks, gt_data, gt_label, save_path)
            # print("Done training")

            print("computing grads")
            # distributed cge
            if args.zero:
                # dy_dx = cge_cal_and_return(remote_networks, net, args.gpus, args.process_per_gpu, gt_data_cpu, gt_label_cpu, args.zoo_step_size)
                # cge
                params_dict = {
                    name: p for name, p in net.named_parameters() if p.requires_grad
                }
                mask_dict = {
                    name: p for name, p in net.named_buffers() if 'mask' in name
                }
                dy_dx = cge(f, params_dict, mask_dict, args.zoo_step_size, net, gt_data, gt_label, F.cross_entropy)
            else:
                # compute original gradient
                out = net(gt_data)
                y = criterion(out, gt_label)
                dy_dx = torch.autograd.grad(y, net.parameters())

            # original_dy_dx = list((dy_dx[_].detach().clone() for _ in dy_dx))
            original_dy_dx = [grad.detach().clone() for grad in dy_dx]
            print("grads computed")

            # generate dummy data and label
            dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
            # dummy_data = torch.randn(gt_data.size()).requires_grad_(True)
            # dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)
            # using ground truth label instead
            dummy_label = gt_label.to(device)


            if method == 'DLG':
                optimizer = torch.optim.LBFGS([dummy_data.float(), dummy_label.float()], lr=lr)
            elif method == 'iDLG':
                optimizer = torch.optim.LBFGS([dummy_data, ], lr=lr)
                # predict the ground-truth label
                label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape(
                    (1,)).requires_grad_(False)

            history = []
            history_iters = []
            losses = []
            mses = []
            train_iters = []

            print('lr =', lr)
            for iters in range(Iteration):
                if args.zero and False:
                    def closure():
                        optimizer.zero_grad()

                        pred = net(dummy_data)
                        dummy_loss = criterion(pred, label_pred)

                        params_dict = {
                            name: p for name, p in net.named_parameters() if p.requires_grad
                        }
                        mask_dict = {
                            name: p for name, p in net.named_buffers() if 'mask' in name
                        }
                        dummy_dy_dx = cge(f, params_dict, mask_dict, args.zoo_step_size, net, dummy_data, dummy_label,
                                    F.cross_entropy)

                        # grad_diff = 0
                        # for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                        #     grad_diff += ((gx - gy) ** 2).sum()
                        # # Cant use backward
                        # # dummy data never changes
                        #
                        # # you need to let the dummy_data can be derived
                        # # grad_x=torch.autograd.grad(grad_diff, dummy_data,create_graph=True, retain_graph=True,allow_unused=True)
                        # grad_diff.backward()
                        # # print("here:",grad_x)
                        # # exit()

                        diffs = [dg - og for dg, og in zip(dummy_dy_dx, original_dy_dx)]
                        grad_diff = sum((d ** 2).sum() for d in diffs)

                        def first_grads(*params):
                            # allow_unused so we don’t error if a param is untouched
                            raw = torch.autograd.grad(dummy_loss, params,
                                                      create_graph=False,
                                                      allow_unused=True)
                            # replace any None with a zero‐tensor of the right shape
                            return tuple(
                                g if g is not None else torch.zeros_like(p)
                                for g, p in zip(raw, params)
                            )

                        # now compute the Hessian–vector product of dummy_loss w.r.t. params along 2*diffs
                        _, hvp = torch.autograd.functional.hvp(
                            first_grads,
                            tuple(net.parameters()),
                            tuple(2 * d for d in diffs),
                            strict=False
                        )

                        # stash the resulting per-parameter hvp into their .grad
                        for p, g in zip(net.parameters(), hvp):
                            p.grad = g

                        print(grad_diff)
                        return grad_diff
                else:
                    def closure():
                        optimizer.zero_grad()
                        pred = net(dummy_data)
                        if method == 'DLG':
                            dummy_loss = - torch.mean(
                                torch.sum(torch.softmax(dummy_label.float(), -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
                            # dummy_loss = criterion(pred, gt_label)
                        elif method == 'iDLG':
                            dummy_loss = criterion(pred, label_pred)

                        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

                        grad_diff = 0
                        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                            grad_diff += ((gx - gy) ** 2).sum()
                        grad_diff.backward()
                        return grad_diff

                optimizer.step(closure)
                current_loss = closure().item()
                train_iters.append(iters)
                losses.append(current_loss)
                mses.append(torch.mean((dummy_data - gt_data) ** 2).item())

                if iters % int(Iteration / 30) == 0:
                    current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
                    print(current_time, iters, 'loss = %.8f, mse = %.8f' % (current_loss, mses[-1]))
                    history.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
                    history_iters.append(iters)

                    for imidx in range(num_dummy):
                        # print(imidx_list, imidx_list[imidx])
                        plt.figure(figsize=(12, 8))
                        plt.subplot(3, 10, 1)
                        plt.imshow(tp(gt_data[imidx].cpu()))
                        for i in range(min(len(history), 29)):
                            plt.subplot(3, 10, i + 2)
                            plt.imshow(history[i][imidx])
                            plt.title('iter=%d' % (history_iters[i]))
                            plt.axis('off')
                        if method == 'DLG':
                            plt.savefig('%s/DLG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                            plt.close()
                        elif method == 'iDLG':
                            plt.savefig('%s/iDLG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                            plt.close()

                    if current_loss < 0.000001:  # converge
                        break

            if method == 'DLG':
                loss_DLG = losses
                label_DLG = torch.argmax(dummy_label, dim=-1).detach().item()
                mse_DLG = mses
            elif method == 'iDLG':
                loss_iDLG = losses
                label_iDLG = label_pred.item()
                mse_iDLG = mses

        print('imidx_list:', imidx_list)
        if method == 'DLG':
            print('loss_DLG:', loss_DLG[-1], 'loss_iDLG:', loss_iDLG[-1])
            print('mse_DLG:', mse_DLG[-1], 'mse_iDLG:', mse_iDLG[-1])
            print('gt_label:', gt_label.detach().cpu().data.numpy(), 'lab_DLG:', label_DLG, 'lab_iDLG:', label_iDLG)
        if method == 'iDLG':
            print('loss_iDLG:', loss_iDLG[-1])
            print('mse_iDLG:', mse_iDLG[-1])
            print('gt_label:', gt_label.detach().cpu().data.numpy(), 'lab_iDLG:', label_iDLG)

        print('----------------------\n\n')
        # index, loss, mse
        csv = open(result_path, 'a')
        csv.write(f"{imidx_list[0]},{loss_iDLG[-1]},{mse_iDLG[-1]}\n")
        csv.close()

def init_process(rank, world_size, args):
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    if rank == 0:
        rpc.init_rpc(
                f"master", rank=rank, world_size=world_size,
                rpc_backend_options=rpc.TensorPipeRpcBackendOptions(num_worker_threads=args.process_per_gpu*world_size+1, rpc_timeout=0.)
            )
        main(args)
    else:
        gpu = args.gpus[(rank-1)//args.process_per_gpu]
        i = (rank-1) % args.process_per_gpu
        rpc.init_rpc(
                f"{gpu}-{i}", rank=rank, world_size=world_size,
                rpc_backend_options=rpc.TensorPipeRpcBackendOptions(num_worker_threads=args.process_per_gpu*world_size+1, rpc_timeout=0.)
            )
    rpc.shutdown()

if __name__ == '__main__':
    p_values = [0.1, 0.6, 0.9] # 1e-10
    default_mu = 5e-3
    delta_mu_values = [10, 5, 1, 1e-1, 1e-2, 1e-3,1e-4,1e-5]
    num_mus = len(delta_mu_values)
    mu_values = []
    for i in range(num_mus):
        mu_value = default_mu + delta_mu_values[i]
        mu_values.append(mu_value)
        mu_values.append(-mu_value)

    # p_values = [0.99]
    # mu_values = [5e-8]
    p_values = [0.9]
    mu_values = [5e-6]

    for p_value in p_values:
        for mu_value in mu_values:
            args = Args(p_value, mu_value)
            world_size = 1 + len(args.gpus) * args.process_per_gpu
            mp.spawn(init_process, args=(world_size, args), nprocs=world_size, join=True)
    # main()