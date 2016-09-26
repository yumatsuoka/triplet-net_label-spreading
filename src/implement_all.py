#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

#######
# Siamese Net + Label-Spreadingを実行
#
# parameters
# gpu:gpuを使う場合はgpuのidを指定, -1でCPUモード
# od: 特徴抽出部で作成する特徴ベクトルの次元数
# n_train: ラベル付与済み画像数
# epoch: 学習epoch数,パラメータのアップデート回数=epoch*n_train/batchsize
# itr: このsiamese net+label-spreadingの処理を繰り返す回数
######

gpu=-1
od = 2
epoch=40
n_train=1000
batchsize=100

itr = 1

for i in range(itr):
    d_name = "mnist_n:"+str(n_train)+"_od:"+str(od)+"_itr:0"+str(i)
    os.system( "echo {}".format(d_name) )
    # train siamese net and get feature vector
    os.system( "python train_triplet.py --epoch {} --outputdim {} --gpu {} --n_train {} --d_name {} --batchsize".format(epoch, od, gpu, n_train, d_name, batchsize)) 
    #implement label-spreading
    os.system( "python a_ls.py --d_name {}".format(d_name))
