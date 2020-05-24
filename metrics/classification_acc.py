# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Frechet Inception Distance (FID)."""

import os
import numpy as np
import scipy
import tensorflow as tf
import dnnlib.tflib as tflib

from metrics import metric_base
from training import misc
from training import dataset

import metrics.mobilenet_v2 as mobilenet_v2
from metrics.mobilenet_v2 import mobilenet_v2_140
from tensorflow.contrib import slim


#----------------------------------------------------------------------------
# TODO pass in dset dize somehow
class CAS(metric_base.MetricBase):
    def __init__(self, minibatch_per_gpu, img_size=[224, 224], train_steps=10000, val_steps=10000, **kwargs):
        super().__init__(**kwargs)
        self.minibatch_per_gpu = minibatch_per_gpu
        self.img_size = img_size

        self._train_dataset_obj = None
        self._val_dataset_obj = None
        self.train_steps = train_steps
        self.val_steps = val_steps
        # Construct classifier graph
        self.ckpt = '/work/newriver/erobb/pickles/mobilenet/mobilenet_v2_1.4_224.ckpt'
        self.train_scope = 'MobilenetV2'
        self.exclude_scope = 'MobilenetV2/Logits'
        self.graph_exists = False

    def _get_partitioned_dataset_obj(self):
        # TODO cache these for speeds
        self._train_dataset_obj = dataset.load_dataset(data_dir=self._data_dir, **dict(self._dataset_args, max_images=self.dset_size // 2))
        self._val_dataset_obj = dataset.load_dataset(data_dir=self._data_dir, **dict(self._dataset_args, skip_images=self.dset_size // 2, max_images=self.dset_size // 2))
        return self._train_dataset_obj, self._val_dataset_obj


    def _get_minibatch_tf(self, dataset):
        c = dataset.shape[0]
        data_fetch_ops = []
        dataset.configure(self.minibatch_per_gpu)

        reals, labels = dataset.get_minibatch_tf()
        reals = tf.image.resize(tf.transpose(tf.reshape(reals, [-1] + dataset.shape),
                                                   [0, 2, 3, 1]), self.img_size)
        reals = tf.cast(reals, tf.float32)
        drange_net = [-1, 1]
        reals = misc.adjust_dynamic_range(reals, dataset.dynamic_range, drange_net)
        return reals, labels        



    def _create_keras_classifier(self):
        mobinet = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                                       include_top=False,
                                                       weights='imagenet')
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        self.logits = tf.keras.layers.Dense(self.num_class)
        self.net = tf.keras.Sequential([
          mobinet,
          global_average_layer,
          self.logits
        ])
        self.img = tf.placeholder(tf.float32, [self.minibatch_per_gpu, 224, 224, 3])
        self.pred = self.net(self.img)
        #base_learning_rate = 0.0001
        # TODO
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                            0.001,
                            decay_steps=2 * self.dset_size,
                            decay_rate=0.94,
                            staircase=True)
        self.net.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])


    def _create_slim_classifier(self, is_training=True):

        self.img = tf.placeholder(tf.float32, [self.minibatch_per_gpu, 224, 224, 3])
        self.lab = tf.placeholder(tf.float32, [self.minibatch_per_gpu, self.num_class])

        with slim.arg_scope(mobilenet_v2.training_scope(weight_decay=4e-05)):
            self.logits, self.end_points = mobilenet_v2_140(self.img, num_classes=self.num_class, is_training=is_training)

        self.loss = tf.losses.softmax_cross_entropy(self.lab, self.logits)
        self.global_step = tf.get_variable('global_step', trainable=False, initializer=0)
        self.learning_rate = tf.train.exponential_decay(
            0.01,
            self.global_step,
            2 * self.dset_size,
            0.94,
            staircase=True,
            name='exponential_decay_learning_rate')
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, 
                                                    decay=0.9,
                                                    momentum=0.9,
                                                    epsilon=1.0)


        mobi_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.train_scope)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.loss, var_list=mobi_vars, global_step=self.global_step)

        exclude_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.exclude_scope)
        rms_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='MobilenetV2/.*/RMSProp')
        self.restore_vars = set(mobi_vars) - set(exclude_vars)
        self.saver = tf.train.Saver(self.restore_vars)
        self.init = tf.initialize_variables(exclude_vars + rms_vars + [self.global_step])
        self.graph_exists = True


    def _get_keras_cas(self, train_img, train_lab, val_img, val_lab):
        self.net.fit(train_img, train_lab, steps_per_epoch=self.train_steps, epochs=1)
        loss, acc = self.net.evaluate(val_img, val_lab, steps=self.val_steps)
        return acc


    def _get_slim_cas(self, train_img, train_lab, val_img, val_lab):
    
        steps = self.train_steps // self.minibatch_per_gpu
        sess = tf.get_default_session()
        sess.run(self.init)
        self.saver.restore(sess, self.ckpt)

        # Train
        for i in range(self.train_steps):
            _img, _lab = sess.run([train_img, train_lab])
            _, _loss, _step, _lr = sess.run([self.train_op, self.loss, self.global_step, self.learning_rate], 
                                            feed_dict={self.img: _img, self.lab: _lab})
            if i % 100 == 0: print(_loss)

        # Evaluate
        correct = 0
        ct = self.val_steps * self.minibatch_per_gpu
        for _ in range(self.val_steps):
            _img, _lab = sess.run([val_img, val_lab])
            _e = sess.run(self.end_points, feed_dict={self.img: _img, self.lab: _lab})
            pred = np.argmax(_e['Predictions'], axis=1)
            _lab = np.argmax(_lab, axis=1)
            correct += np.sum(pred == _lab)
        print('Eval Reals Acc', correct / ct)
        return correct / ct


    def _evaluate(self, Gs, Gs_kwargs, num_gpus, rho):
        val_dset = dataset.load_dataset(data_dir=self._data_dir, **dict(self._dataset_args, max_images=None))
        val_imgs, val_labels = self._get_minibatch_tf(val_dset)
        self.dset_size = val_dset.get_length()
        self.num_class = val_labels.shape[-1]
        train_dset = dataset.load_dataset(data_dir=self._data_dir, **dict(self._dataset_args_train, max_images=None))
        train_imgs, train_labels = self._get_minibatch_tf(train_dset)
        # Get generated images
        latents = tf.random_normal([self.minibatch_per_gpu] + Gs.input_shape[1:])
        fake_labels = self._get_random_labels_tf(self.minibatch_per_gpu)
        fake_imgs = Gs.get_output_for(latents, fake_labels, np.array([rho]), **Gs_kwargs)
        fake_imgs = tf.image.resize(tf.transpose(fake_imgs, [0, 2, 3, 1]), self.img_size)
        self._create_keras_classifier()
        acc = self._get_keras_cas(fake_imgs, fake_labels, val_imgs, val_labels)
        self._report_result(acc)
