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

#----------------------------------------------------------------------------

class CAS(metric_base.MetricBase):
    def __init__(self, num_images, minibatch_per_gpu, dset_size=3000, img_size=[224, 224], train_steps=10000, val_steps=10000, **kwargs):
        super().__init__(**kwargs)
        self.num_images = num_images
        self.minibatch_per_gpu = minibatch_per_gpu
        self.img_size = img_size

        self._train_dataset_obj = None
        self._val_dataset_obj = None
        self.train_steps = train_steps
        self.val_steps = val_steps
        self.dset_size = dset_size


    def test(self, mobinet, reals, labels, n=100):
        from PIL import Image
        import time
        test_dir = '/work/newriver/erobb/tmp/%d' % int(time.time())
        os.mkdir(test_dir)
        for i in range(n):
            imgs, labs = tflib.run([reals, labels])
            img = np.uint8((1 + imgs[0]) * 127.5)
            lab = np.where(labs[0])[0][0]
            p = mobinet.predict(imgs[:1])
            p = np.argmax(1 / (1 + np.exp(-p)))
            Image.fromarray(img).save('%s/%d_l%d_p%d.png' % (test_dir, i, lab, p))


    def _get_partitioned_dataset_obj(self):
        # TODO cache these for speeds
        self._train_dataset_obj = dataset.load_dataset(data_dir=self._data_dir, **dict(self._dataset_args, max_images=self.dset_size // 2))
        self._val_dataset_obj = dataset.load_dataset(data_dir=self._data_dir, **dict(self._dataset_args, skip_images=self.dset_size // 2, max_images=self.dset_size // 2))
        return self._train_dataset_obj, self._val_dataset_obj


    def _get_minibatch_tf(self, dataset):
        c = dataset.shape[0]
        data_fetch_ops = []

        reals, labels = dataset.get_minibatch_tf()
        reals = tf.image.resize(tf.transpose(tf.reshape(reals, [-1] + dataset.shape),
                                                   [0, 2, 3, 1]), self.img_size)
        reals = tf.cast(reals, tf.float32)
        drange_net = [-1, 1]
        reals = misc.adjust_dynamic_range(reals, dataset.dynamic_range, drange_net)
        dataset.configure(self.minibatch_per_gpu)
        return reals, labels        


    def _create_classifier(self, n_classes):
        mobinet = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                                       include_top=False,
                                                       weights='imagenet')
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer = tf.keras.layers.Dense(n_classes)
        mobinet = tf.keras.Sequential([
          mobinet,
          global_average_layer,
          prediction_layer
        ])
        base_learning_rate = 0.0001
        mobinet.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        return mobinet


    def _evaluate(self, Gs, Gs_kwargs, num_gpus, rho):
        np.random.seed(123)
        tf.set_random_seed(123)
        os.environ['PYTHONHASHSEED']='123'
        import random as rn; rn.seed(123)
        # Split the evaluation set into train / val partitions for the classifier.
        train_set, val_set = self._get_partitioned_dataset_obj()
        reals, reals_labels = self._get_minibatch_tf(train_set)
        val_reals, val_labels = self._get_minibatch_tf(val_set)
        # Create classifier
        assert reals_labels.shape[1] > 0
        reals_cls = self._create_classifier(reals_labels.shape[1])
        steps = self.num_images // self.minibatch_per_gpu

        # Train on reals
        print('Training classifier on reals...')
        reals_cls.fit(reals, reals_labels, steps_per_epoch=steps, epochs=1)
        print('Eval acc...')
        _, reals_acc = reals_cls.evaluate(val_reals, val_labels, steps=steps)

        # Get generated images
        latents = tf.random_normal([self.minibatch_per_gpu] + Gs.input_shape[1:])
        fakes_labels = self._get_random_labels_tf(self.minibatch_per_gpu)
        # TODO should roughly lie in [-1, 1] but should I rescale it for the classifier? See if it makes a difference.
        fakes = Gs.get_output_for(latents, fakes_labels, np.array([rho]), **Gs_kwargs)
        fakes = tf.image.resize(tf.transpose(fakes, [0, 2, 3, 1]), self.img_size)

        # Train on gen
        print('Training classifier on fakes...')
        fakes_cls = self._create_classifier(fakes_labels.shape[1])
        fakes_cls.fit(fakes, fakes_labels, steps_per_epoch=steps, epochs=1)
        print('Eval acc...')
        _, fakes_acc = fakes_cls.evaluate(val_reals, val_labels, steps=steps)

        self.test(reals_cls, reals, reals_labels)
        self.test(fakes_cls, fakes, fakes_labels)
        self._report_result(reals_acc - fakes_acc)
