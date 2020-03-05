# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Average LPIPS."""

import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib

from metrics import metric_base
from training import misc

#----------------------------------------------------------------------------

# Normalize batch of vectors.
def normalize(v):
    return v / tf.sqrt(tf.reduce_sum(tf.square(v), axis=-1, keepdims=True))

# Spherical interpolation of a batch of vectors.
def slerp(a, b, t):
    a = normalize(a)
    b = normalize(b)
    d = tf.reduce_sum(a * b, axis=-1, keepdims=True)
    p = t * tf.math.acos(d)
    c = normalize(b - d * a)
    d = a * tf.math.cos(p) + c * tf.math.sin(p)
    return normalize(d)

#----------------------------------------------------------------------------

class LPIPS(metric_base.MetricBase):
    def __init__(self, num_samples, epsilon, space, sampling, crop, minibatch_per_gpu, Gs_overrides, **kwargs):
        assert space in ['z', 'w']
        assert sampling in ['full', 'end']
        super().__init__(**kwargs)
        self.num_samples = num_samples
        self.epsilon = epsilon
        self.space = space
        self.sampling = sampling
        self.crop = crop
        self.minibatch_per_gpu = minibatch_per_gpu
        self.Gs_overrides = Gs_overrides

    def _evaluate(self, Gs, Gs_kwargs, num_gpus, rho):
        Gs_kwargs = dict(Gs_kwargs)
        Gs_kwargs.update(self.Gs_overrides)
        minibatch_size = num_gpus * self.minibatch_per_gpu

        # Construct TensorFlow graph.
        distance_expr = []
        Gs_clone = Gs.clone()
        distance_measure = misc.load_pkl('http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/vgg16_zhang_perceptual.pkl')
        latents = tf.random_normal([self.minibatch_per_gpu] + Gs_clone.input_shape[1:])
        labels = self._get_random_labels_tf(self.minibatch_per_gpu)
        fakes = Gs_clone.get_output_for(latents, labels, np.array([rho]), **Gs_kwargs)


        distances = []
        #for i in range(100):
        #    fake = tflib.run(fakes)
        #    print(i)
           
        for idx, real in enumerate(self._iterate_reals(minibatch_size=self.minibatch_per_gpu)):
            if idx > self.num_samples: break
            fake = tflib.run(fakes)
            dist = distance_measure.run(real, fake)
            distances.append(dist)
        self._report_result(np.mean(np.concatenate(distances)))


#----------------------------------------------------------------------------
