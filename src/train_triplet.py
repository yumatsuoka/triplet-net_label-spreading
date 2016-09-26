#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train triplet net and get feature vectors
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import time
import argparse
# from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
import numpy as np

import chainer
from chainer import cuda, optimizers, serializers

import dump_vec
import triplet_net


def train_and_dump(model, optimizer, labeled_data_dict, unlabeled_data_dict,\
        xp, batchsize, epoch, plot_dim, gpu, outputdim, d_name):
    x_train = labeled_data_dict['data']
    y_train = labeled_data_dict['target']
    loss_list = np.empty(epoch)
    
    for itr in six.moves.range(1, epoch + 1):
    # for itr in tqdm(six.moves.range(1, epoch + 1)):
        print('epoch', itr)
        xall_a, xall_p, xall_f = data_feed(x_train, y_train)
        n_train = len(xall_a)
        perm = np.random.permutation(n_train)
        sum_train_loss = 0
        for i in six.moves.range(0, n_train, batchsize):
            x_a = xall_a[perm[i:i + batchsize]]
            x_p = xall_p[perm[i:i + batchsize]]
            x_f = xall_f[perm[i:i + batchsize]]

            x_a = xp.asarray(x_a, dtype=xp.float32)
            x_p = xp.asarray(x_p, dtype=xp.float32)
            x_f = xp.asarray(x_f, dtype=xp.float32)
            real_batchsize = len(x_a)
            
            optimizer.zero_grads()
            loss = model(x_a, x_p, x_f)
            loss.backward()
            optimizer.update()

            sum_train_loss += float(cuda.to_cpu(loss.data)) * real_batchsize
        print('train mean loss={}'.format(sum_train_loss / n_train))
        loss_list[epoch-1] = sum_train_loss / n_train

    # print('dump model & optimizer')
    # serializers.save_hdf5('../dump/triplet.model', model)
    # serializers.save_hdf5('../dump/triplet.state', optimizer)
    
    print('Make loss graph')
    plt.clf()
    plt.xlabel('weight update')
    plt.ylabel('loss')
    plt.plot(loss_list)
    plt.savefig('../dump/{}_loss.png'.format(d_name))

    print('dump feature vector')
    dump_vec.dump_feature_vector(model, '../dump/{}_label'.format(d_name),\
            labeled_data_dict, outputdim, batchsize, xp, gpu)
    dump_vec.dump_feature_vector(model, '../dump/{}_unlabel'.format(d_name),\
            unlabeled_data_dict, outputdim, batchsize, xp, gpu, plot_dim)


def get_mnist(n_data):
    mnist = fetch_mldata('MNIST original')
    r_data = mnist['data'].astype(np.float32)
    r_label = mnist['target'].astype(np.int32)
    # former 60,000 samples which is training data in MNIST.
    perm = np.random.permutation(60000)
    data = r_data[perm]
    label = r_label[perm]
    # split the data to training data(labeled data) and test data(unlabeled data)
    ld_dict = {'data':data[:n_data].reshape((n_data, 1, 28, 28)) / 255.0,\
            'target':label[:n_data]}
    unld_dict = {'data':data[n_data:].reshape((60000-n_data, 1, 28, 28)) / 255.0,\
            'target':label[n_data:]}
    
    return ld_dict, unld_dict



def data_feed(train_data, train_label):
    n_class = 10
    nl = len(train_label)
    nl_class = [len(np.where(train_label==c)[0]) for c in range(n_class)]
    xa = np.asarray([train_data[idx] for c in range(n_class) 
                                     for i in range(nl-nl_class[c]) 
                                     for idx in np.random.permutation(np.where(train_label==c)[0])])
    xp = np.asarray([train_data[idx] for c in range(n_class) 
                                     for i in range(nl-nl_class[c]) 
                                     for idx in np.random.permutation(np.where(train_label==c)[0])])
    xf = np.asarray([train_data[idx] for c in range(n_class) 
                                     for i in range(nl_class[c]) 
                                     for idx in np.random.permutation(np.where(train_label!=c)[0])])
    
    return xa, xp, xf


if __name__ == '__main__':

    st = time.clock()
    s_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=-1, type=int)
    parser.add_argument('--epoch', default=40, type=int)
    parser.add_argument('--batchsize', default=100, type=int)
    parser.add_argument('--initmodel', default=0, type=int)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--outputdim', default=2, type=int)
    parser.add_argument('--n_train', default=1000, type=int)
    parser.add_argument('--plot_dim', default=100, type=int)
    parser.add_argument('--d_name', default='hoge', type=str)
    args = parser.parse_args()

    print('Create model')
    model = triplet_net.Triplet_net(args.outputdim)

    print('Check gpu')
    if args.gpu >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if args.gpu >= 0 else np
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()
    
    print('Load dataset')
    ld_dict, unld_dict = get_mnist(args.n_train)

    print('Setup optimizer')
    optimizer = optimizers.Adam(alpha=0.0002)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001))

    if args.initmodel:
        model_dir = '../dump/{}_triplet.model'.format(args.d_name)
        serializers.load_hdf5(model_dir, model)
    if args.resume:
        state_dir = '../dump/{}_triplet.state'.format(args.d_name)
        serializers.load_hdf5(state_dir, optimizer)
    
    print('training and test')
    train_and_dump(model, optimizer, ld_dict, unld_dict, xp, args.batchsize,\
            args.epoch, args.plot_dim, args.gpu, args.outputdim, args.d_name)
    print('end')
    print('elapsed time[m]:', (time.clock() - st)/60.0)
