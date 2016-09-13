#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
make triplet net using chainer
"""

import chainer
import chainer.functions as F
import chainer.links as L

class Triplet_net(chainer.Chain):

    def __init__(self, output_dim):
        super(Triplet_net, self).__init__(
            conv1=L.Convolution2D(1, 32, 3, pad=1),
            conv2=L.Convolution2D(32, 32, 3, pad=1),
            conv3=L.Convolution2D(32, 64, 3, pad=1),
            conv4=L.Convolution2D(64, 64, 3, pad=1),
            fc1=L.Linear(3136, output_dim),
        )
        self.train = True

    def clear(self):
        self.loss = None

    def __call__(self, anchor, positive, negative):
        self.clear()
        y_a = self.forward_one(anchor)
        y_p = self.forward_one(positive)
        y_n = self.forward_one(negative)

        self.loss = F.triplet(y_a, y_p, y_n)
        return self.loss

    def forward_one(self, x_data):
        x = chainer.Variable(x_data, volatile='off')
        h = F.max_pooling_2d(F.relu(
            F.local_response_normalization(self.conv1(x))), 3, stride=2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(F.relu(
            F.local_response_normalization(self.conv3(h))), 3, stride=2)
        h = F.relu(self.conv4(h))
        y = self.fc1(h)
        return y
