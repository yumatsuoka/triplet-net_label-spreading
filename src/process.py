#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random

gpu=1
rs = int(random.uniform(1, 1000))
plot_dim = 100
m_nolabel=1

itr = 10
#epoch=10
n_train = 600
bs=32
momentum=0.9
lr=0.0001
wd=0.00001
od = 100

for epoch in [20, 40]:
    for i in range(itr):
        rs = int(random.uniform(1, 1000))
        os.system( "echo od:{} n_train:{} epoch:{}".format(od, n_train, epoch) )
        os.system( "python implement.py --train 1 --data train --model triplet --epoch {} --outputdim {} --gpu {} --lr {} --n_train {} --rand_seed {} --wd {} --batchsize {}".format(epoch, od, gpu, lr, n_train, rs, wd, bs) ) 

#テスト鉀
#学習しているデータの徴ベクトルの作成
        os.system( "python implement.py  --train 0 --data train --model triplet --initmodel 1 --resume 1 --outputdim {} --gpu {} --n_train {} --rand_seed {}".format(od, gpu, n_train, rs) )

#学習していないデータの徴ベクトルの作成
        os.system( "python implement.py  --train 0 --data test --model triplet --initmodel 1 --resume 1 --outputdim {} --gpu {} --n_train {} --rand_seed {} --m_nolabel {} --plot_dim {}".format(od, gpu, n_train, rs, m_nolabel, plot_dim) )

        os.system(" python apply_label-spreading.py")
