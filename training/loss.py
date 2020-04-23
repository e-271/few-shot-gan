# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Loss functions."""

import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary

#----------------------------------------------------------------------------
# Logistic loss from the paper
# "Generative Adversarial Nets", Goodfellow et al. 2014

def G_logistic(G, D, opt, training_set, minibatch_size):
    _ = opt
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)[:,0:1]
    loss = -tf.nn.softplus(fake_scores_out) # log(1-sigmoid(fake_scores_out)) # pylint: disable=invalid-unary-operand-type
    return loss, None

def G_logistic_ns(G, D, opt, training_set, minibatch_size):
    _ = opt
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)[:,0:1]
    loss = tf.nn.softplus(-fake_scores_out) # -log(sigmoid(fake_scores_out))
    return loss, None

def D_logistic(G, D, opt, training_set, minibatch_size, reals, labels):
    _ = opt, training_set
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = D.get_output_for(reals, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = tf.nn.softplus(fake_scores_out) # -log(1-sigmoid(fake_scores_out))
    loss += tf.nn.softplus(-real_scores_out) # -log(sigmoid(real_scores_out)) # pylint: disable=invalid-unary-operand-type
    return loss, None

#----------------------------------------------------------------------------
# R1 and R2 regularizers from the paper
# "Which Training Methods for GANs do actually Converge?", Mescheder et al. 2018

def D_logistic_r1(G, D, opt, training_set, minibatch_size, reals, labels, gamma=10.0):
    _ = opt, training_set
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    rho = np.array([1])
    fake_images_out = G.get_output_for(latents, labels, rho, is_training=True)[0]
    real_scores_out = D.get_output_for(reals, labels, is_training=True)[0]
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)[0]
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = tf.nn.softplus(fake_scores_out) # -log(1-sigmoid(fake_scores_out))
    loss += tf.nn.softplus(-real_scores_out) # -log(sigmoid(real_scores_out)) # pylint: disable=invalid-unary-operand-type

    with tf.name_scope('GradientPenalty'):
        real_grads = tf.gradients(tf.reduce_sum(real_scores_out), [reals])[0]
        gradient_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1,2,3])
        gradient_penalty = autosummary('Loss/gradient_penalty', gradient_penalty)
        reg = gradient_penalty * (gamma * 0.5)
    return loss, reg

def D_logistic_r2(G, D, opt, training_set, minibatch_size, reals, labels, gamma=10.0):
    _ = opt, training_set
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = D.get_output_for(reals, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = tf.nn.softplus(fake_scores_out) # -log(1-sigmoid(fake_scores_out))
    loss += tf.nn.softplus(-real_scores_out) # -log(sigmoid(real_scores_out)) # pylint: disable=invalid-unary-operand-type

    with tf.name_scope('GradientPenalty'):
        fake_grads = tf.gradients(tf.reduce_sum(fake_scores_out), [fake_images_out])[0]
        gradient_penalty = tf.reduce_sum(tf.square(fake_grads), axis=[1,2,3])
        gradient_penalty = autosummary('Loss/gradient_penalty', gradient_penalty)
        reg = gradient_penalty * (gamma * 0.5)
    return loss, reg

#----------------------------------------------------------------------------
# WGAN loss from the paper
# "Wasserstein Generative Adversarial Networks", Arjovsky et al. 2017

def G_wgan(G, D, opt, training_set, minibatch_size):
    _ = opt
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)[:,0:1]
    loss = -fake_scores_out
    return loss, None

def D_wgan(G, D, opt, training_set, minibatch_size, reals, labels, wgan_epsilon=0.001):
    _ = opt, training_set
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = D.get_output_for(reals, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = fake_scores_out - real_scores_out
    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
        loss += epsilon_penalty * wgan_epsilon
    return loss, None

#----------------------------------------------------------------------------
# WGAN-GP loss from the paper
# "Improved Training of Wasserstein GANs", Gulrajani et al. 2017

def D_wgan_gp(G, D, opt, training_set, minibatch_size, reals, labels, wgan_lambda=10.0, wgan_epsilon=0.001, wgan_target=1.0):
    _ = opt, training_set
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = D.get_output_for(reals, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = fake_scores_out - real_scores_out
    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon

    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tflib.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        mixed_scores_out = D.get_output_for(mixed_images_out, labels, is_training=True)
        mixed_scores_out = autosummary('Loss/scores/mixed', mixed_scores_out)
        mixed_grads = tf.gradients(tf.reduce_sum(mixed_scores_out), [mixed_images_out])[0]
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        mixed_norms = autosummary('Loss/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
        reg = gradient_penalty * (wgan_lambda / (wgan_target**2))
    return loss, reg

#----------------------------------------------------------------------------
# Non-saturating logistic loss with path length regularizer from the paper
# "Analyzing and Improving the Image Quality of StyleGAN", Karras et al. 2019

def G_logistic_ns_pathreg(G, D, opt, training_set, minibatch_size, pl_minibatch_shrink=2, pl_decay=0.01, pl_weight=2.0):
    _ = opt
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    rho = np.array([1])
    fake = G.get_output_for(latents, labels, rho, is_training=True, return_dlatents=True)
    fake_images_out, fake_dlatents_out = fake[0], fake[-1]
    # fake_images_out, fake_dlatents_out = G.get_output_for(latents, labels, rho, is_training=True, return_dlatents=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)[0][:,0:1]
    loss = tf.nn.softplus(-fake_scores_out) # -log(sigmoid(fake_scores_out))

    # Path length regularization.
    with tf.name_scope('PathReg'):

        # Evaluate the regularization term using a smaller minibatch to conserve memory.
        if pl_minibatch_shrink > 1:
            pl_minibatch = minibatch_size // pl_minibatch_shrink
            pl_latents = tf.random_normal([pl_minibatch] + G.input_shapes[0][1:])
            pl_labels = training_set.get_random_labels_tf(pl_minibatch)
            fake = G.get_output_for(pl_latents, pl_labels, rho, is_training=True, return_dlatents=True)
            fake_images_out, fake_dlatents_out = fake[0], fake[-1]
            # fake_images_out, fake_dlatents_out = G.get_output_for(pl_latents, pl_labels, rho, is_training=True, return_dlatents=True)

        # Compute |J*y|.
        pl_noise = tf.random_normal(tf.shape(fake_images_out)) / np.sqrt(np.prod(G.output_shape[2:])) # N x 3 x 256 x 256
        pl_grads = tf.gradients(tf.reduce_sum(fake_images_out * pl_noise), [fake_dlatents_out])[0] # N x 14 x 512 (synthesis network output w from Nx512 z)?

        pl_lengths = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(pl_grads), axis=2), axis=1)) # N x 1??
        pl_lengths = autosummary('Loss/pl_lengths', pl_lengths)

        # Track exponential moving average of |J*y|.
        with tf.control_dependencies(None):
            pl_mean_var = tf.Variable(name='pl_mean', trainable=False, initial_value=0.0, dtype=tf.float32)
        pl_mean = pl_mean_var + pl_decay * (tf.reduce_mean(pl_lengths) - pl_mean_var) # Scalar. If this is EMA where is the / N... Passed in from elsewhere??
        pl_update = tf.assign(pl_mean_var, pl_mean)

        # Calculate (|J*y|-a)^2.
        with tf.control_dependencies([pl_update]):
            pl_penalty = tf.square(pl_lengths - pl_mean) # E_ij[||J*y||_2 - a]
            pl_penalty = autosummary('Loss/pl_penalty', pl_penalty)

        # Apply weight.
        #
        # Note: The division in pl_noise decreases the weight by num_pixels, and the reduce_mean
        # in pl_lengths decreases it by num_affine_layers. The effective weight then becomes:
        #
        # gamma_pl = pl_weight / num_pixels / num_affine_layers
        # = 2 / (r^2) / (log2(r) * 2 - 2)
        # = 1 / (r^2 * (log2(r) - 1))
        # = ln(2) / (r^2 * (ln(r) - ln(2))
        #
        reg = pl_penalty * pl_weight

    return loss, reg



#----------------------------------------------------------------------------
# Adaptive regularization losses

def G_logistic_ns_pathreg_adareg(G, D, opt, training_set, minibatch_size, pl_minibatch_shrink=2, pl_decay=0.01, pl_weight=2.0, rho=0.0):
    loss, reg = G_logistic_ns_pathreg(G, D, opt, training_set, minibatch_size, pl_minibatch_shrink, pl_decay, pl_weight)
    ada_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=G.components['synthesis'].scope + '/.*/adapt')
    ada_reg = 0
    for var in ada_vars:
        ada_reg += rho * tf.reduce_sum(var)
    ada_reg = autosummary('Loss/adareg/G', ada_reg)
    reg += ada_reg
    return loss, reg


#----------------------------------------------------------------------------
# Jacobian clamping 

def norm(x):
    return tf.sqrt(tf.reduce_sum(tf.square(x), axis=list(range(1, len(x.shape)))))

# TODO(me): May need to input the noise images to make consistent between these.
def G_logistic_ns_pathreg_jc(G, D, opt, training_set, minibatch_size, pl_minibatch_shrink=2, pl_decay=0.01, pl_weight=2.0, epsilon=0.01, lambda_min=1.0, lambda_max=2.0):
    loss, reg = G_logistic_ns_pathreg(G, D, opt, training_set, minibatch_size, pl_minibatch_shrink, pl_decay, pl_weight)

    # Jacobian clamping regularization.
    with tf.name_scope('JCReg'):
        # Evaluate the regularization term using a smaller minibatch to conserve memory.
        jc_minibatch = minibatch_size // pl_minibatch_shrink
        jc_latents1 = tf.random_normal([jc_minibatch] + G.input_shapes[0][1:])
        jc_noise = tf.random_normal([jc_minibatch] + G.input_shapes[0][1:])
        jc_latents2 = epsilon * jc_noise / tf.reshape(norm(jc_noise), (-1, 1)) + jc_latents1
        jc_labels = training_set.get_random_labels_tf(jc_minibatch)
        rho = np.array([1])
        fake_images_out1, fake_dlatents_out1 = G.get_output_for(jc_latents1, jc_labels, rho, randomize_noise=False, style_mixing_prob=None, is_training=True, return_dlatents=True)
        fake_images_out2, fake_dlatents_out2 = G.get_output_for(jc_latents2, jc_labels, rho, randomize_noise=False, style_mixing_prob=None, is_training=True, return_dlatents=True)

        #jc_norm = tf.norm(fake_images_out1 - fake_images_out2) / tf.norm(fake_dlatents_out1 - fake_dlatents_out2)
        jc_norm = norm(fake_images_out1 - fake_images_out2) / norm(fake_dlatents_out1 - fake_dlatents_out2)
        jc_reg = tf.square(lambda_max - tf.maximum(lambda_max, jc_norm))
        jc_reg += tf.square(lambda_min - tf.minimum(lambda_min, jc_norm))
        reg += jc_reg
        jc_norm = autosummary('Loss/jc_norm', jc_norm)
        jc_reg = autosummary('Loss/jc_reg', jc_reg)

    return loss, reg




#--------------------------------
# Diversity regularization from https://arxiv.org/pdf/1901.09024.pdf
def G_logistic_ns_pathreg_div(G, D, opt, training_set, minibatch_size, pl_minibatch_shrink=2, pl_decay=0.01, pl_weight=2.0, tau=0.01):
    loss, reg = G_logistic_ns_pathreg(G, D, opt, training_set, minibatch_size, pl_minibatch_shrink, pl_decay, pl_weight)

    # Jacobian clamping regularization.
    with tf.name_scope('Diversity'):
        # Evaluate the regularization term using a smaller minibatch to conserve memory.
        div_minibatch = minibatch_size // pl_minibatch_shrink
        div_latents1 = tf.random_normal([div_minibatch] + G.input_shapes[0][1:])
        div_latents2 = tf.random_normal([div_minibatch] + G.input_shapes[0][1:])
        div_labels = training_set.get_random_labels_tf(div_minibatch)
        rho = np.array([1])
        fake_images_out1, fake_dlatents_out1 = G.get_output_for(div_latents1, div_labels, rho, is_training=True, return_dlatents=True)
        fake_images_out2, fake_dlatents_out2 = G.get_output_for(div_latents2, div_labels, rho, is_training=True, return_dlatents=True)

        div_norm = norm(fake_images_out1 - fake_images_out2) / norm(fake_dlatents_out1 - fake_dlatents_out2)
        div_reg = tf.minimum(div_norm, tau)
        div_norm = autosummary('Loss/div_norm', div_norm)
        div_reg = autosummary('Loss/div_reg', div_reg)
        reg -= div_reg # maximize div_reg

    return loss, reg



#----------------------------------------------------------------------------
# Gradient sparsity regularizaion 

# Gini index for 1D tf.tensor
def gini_index(x):
    rowstack, colstack = tf.meshgrid(x, x)
    md = tf.reduce_mean(tf.sqrt((rowstack - colstack)**2))
    am = tf.reduce_mean(x) 
    rmd = md / am
    gini = rmd / 2
    return gini


def G_logistic_ns_gsreg(G, D, opt, training_set, minibatch_size, gs_minibatch_shrink=2, gs_decay=0.1, gs_weight=3.0):
    _ = opt
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    rho = np.array([1])
    fake_images_out, fake_dlatents_out = G.get_output_for(latents, labels, rho, is_training=True, return_dlatents=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)[:,0:1]
    loss = tf.nn.softplus(-fake_scores_out) # -log(sigmoid(fake_scores_out))

    # Path length regularization.
    with tf.name_scope('GradSparsityReg'):

        # Evaluate the regularization term using a smaller minibatch to conserve memory.
        if gs_minibatch_shrink > 1:
            gs_minibatch = minibatch_size // gs_minibatch_shrink
            gs_latents = tf.random_normal([gs_minibatch] + G.input_shapes[0][1:])
            gs_labels = training_set.get_random_labels_tf(gs_minibatch)
            fake_images_out, fake_dlatents_out = G.get_output_for(gs_latents, gs_labels, rho, is_training=True, return_dlatents=True)

        # Compute |J*y|.a
        # TODO why is there noise. Do we want noise?
        gs_noise = tf.random_normal(tf.shape(fake_images_out)) / np.sqrt(np.prod(G.output_shape[2:])) # N x 3 x 256 x 256
        gs_grads = tf.gradients(tf.reduce_sum(fake_images_out * gs_noise), [fake_dlatents_out])[0] # N x 14 x 512 (synthesis network output w from Nx512 z)?

        # TODO Not to sure about just flattening this guy.
        gs_sparsity = gini_index(tf.reshape(gs_grads, [-1])) #tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(gs_grads), axis=2), axis=1))
        gs_sparsity = autosummary('Loss/gs_sparsity', gs_sparsity)

        # TODO not sure we really want this fancy buisiness, unless we use the initial value of sparsity (seems smart) or a large decay like 0.1 (also)
        # Track exponential moving average of |J*y|.
        with tf.control_dependencies(None):
            gs_mean_var = tf.Variable(name='gs_mean', trainable=False, initial_value=0.0, dtype=tf.float32)
        gs_mean = gs_mean_var + gs_decay * (tf.reduce_mean(gs_sparsity) - gs_mean_var) # If this is EMA where is the / N... Passed in from elsewhere??
        gs_update = tf.assign(gs_mean_var, gs_mean)

        # Calculate (|J*y|-a)^2.
        with tf.control_dependencies([gs_update]):
            gs_penalty = tf.square(gs_sparsity - gs_mean) # E_ij[||J*y||_2 - a]
            gs_penalty = autosummary('Loss/gs_penalty', gs_penalty)

        reg = gs_penalty * gs_weight

    return loss, reg

#----------------------------------------------------------------------------
# Adaptive weight regulariation

def D_logistic_r1_adareg(G, D, opt, training_set, minibatch_size, reals, labels, gamma=10.0, rho=0.0):
    loss, reg = D_logistic_r1(G, D, opt, training_set, minibatch_size, reals, labels, gamma)
    ada_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=D.scope + '/.*/adapt')
    ada_reg = 0
    for var in ada_vars:
        ada_reg += rho * tf.reduce_sum(var)
    ada_reg = autosummary('Loss/adareg/D', ada_reg)
    reg += ada_reg
    return loss, reg

#----------------------------------------------------------------------------
# Cosine similarity loss
def D_logistic_r1_cos(G, D, opt, training_set, minibatch_size, reals, labels, gamma=10.0):
    _ = opt, training_set
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    rho = np.array([1])
    fake_images_out = G.get_output_for(latents, labels, rho, is_training=True)
    real_scores_out = D.get_output_for(reals, labels, is_training=True)[:,1:2]
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)[:,0:1]
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = tf.nn.softplus(fake_scores_out) # -log(1-sigmoid(fake_scores_out))
    loss += tf.nn.softplus(-real_scores_out) # -log(sigmoid(real_scores_out)) # pylint: disable=invalid-unary-operand-type

    with tf.name_scope('GradientPenalty'):
        real_grads = tf.gradients(tf.reduce_sum(real_scores_out), [reals])[0]
        gradient_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1,2,3])
        gradient_penalty = autosummary('Loss/gradient_penalty', gradient_penalty)
        reg = gradient_penalty * (gamma * 0.5)
    return loss, reg



#----------------------------------------------------------------------------
# Autoencoder loss

def AE_l2(MI, G, opt, training_set, minibatch_size):
    _ = opt
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    rho1 = np.array([1])
    rho0 = np.array([0])
    fake0 = G.get_output_for(latents, labels, rho0, randomize_noise=False, style_mixing_prob=None, is_training=True)[0]
    fake1 = G.get_output_for(latents, labels, rho1, randomize_noise=False, style_mixing_prob=None, is_training=True)[0]
    fake0_recon = MI.get_output_for(fake1, labels)
    loss = tf.reduce_mean((fake0_recon - fake0)**2, axis=[1,2,3]) # MSE
    loss = autosummary('Loss/enc_l2', loss)
    return loss, fake1, fake0, fake0_recon


# TODO HOLLLLLD UP
# Check on this, see if it has the artifact with "If false".
def G_logistic_ns_pathreg_ae(G, D, MI, opt, training_set, minibatch_size, pl_minibatch_shrink=2, pl_decay=0.01, pl_weight=2.0, tau=0.01, ae_loss_weight=0.0):
    loss, reg = G_logistic_ns_pathreg(G, D, opt, training_set, minibatch_size, pl_minibatch_shrink, pl_decay, pl_weight)
    # Mutual information.
    ae_loss = tf.zeros([minibatch_size, 1])
    with tf.name_scope('MutualInfo'):
        latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
        labels = training_set.get_random_labels_tf(minibatch_size)
        rho1 = np.array([1])
        rho0 = np.array([0])
        fake0 = G.get_output_for(latents, labels, rho0, randomize_noise=False, style_mixing_prob=None, is_training=True)[0]
        fake1 = G.get_output_for(latents, labels, rho1, randomize_noise=False, style_mixing_prob=None, is_training=True)[0]
        fake0_recon = MI.get_output_for(fake1, labels)
        ae_loss = tf.reduce_mean((fake0_recon - fake0)**2, axis=[1,2,3]) # MSE
        loss += ae_loss_weight * tf.reshape(ae_loss, [-1, 1])
        ae_loss = autosummary('Loss/ae_loss', ae_loss)
    return loss, reg, ae_loss


#----------------------------------------------------------------------------
# Feature Autoencoder loss


# TODO these
def FeatAE_l2(MI, G, D, opt, training_set, minibatch_size):
    print('MI feature loss...')
    _ = opt
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    rho1 = np.array([1])
    rho0 = np.array([0])
    gfake0 = G.get_output_for(latents, labels, rho0, randomize_noise=False, style_mixing_prob=None, is_training=True)
    dfake0 = D.get_output_for(gfake0[0], labels, rho0, is_training=True)
    gfake1 = G.get_output_for(latents, labels, rho1, randomize_noise=False, style_mixing_prob=None, is_training=True)
    dfake1 = D.get_output_for(gfake1[0], labels, rho1, is_training=True)
    feats0 = gfake0[1:] + dfake0[1:]
    feats1 = gfake1[1:] + dfake1[1:]
    feats0_pred = MI.get_output_for(gfake1[0], labels, feats_in=feats1)
    gloss, dloss = tf.zeros([]), tf.zeros([])
    loss = tf.zeros([])
    for fp0, f0 in zip(feats0_pred, feats0): 
        #loss += norm(fp0 - f0)
        ll = tf.reduce_mean(tf.square(fp0-f0), axis=list(range(1, len(f0.shape))))
        if 'Recon/D' in fp0.name: 
            gloss += ll
        elif 'Recon/G' in fp0.name: 
            dloss += ll
    loss = gloss + dloss
    return loss, gloss, dloss # feats1, feats0, feats0_pred


def G_logistic_ns_pathreg_ft(G, D, MI, opt, training_set, minibatch_size, pl_minibatch_shrink=2, pl_decay=0.01, pl_weight=2.0, tau=0.01, mi_weight=0.000001):
    print('G feature loss...')
    loss, reg = G_logistic_ns_pathreg(G, D, opt, training_set, minibatch_size, pl_minibatch_shrink, pl_decay, pl_weight)
    # TODO split to G only
    _, mi_loss,_ = FeatAE_l2(MI, G, D, opt, training_set, minibatch_size)
    loss += mi_weight * mi_loss
    return loss, reg, mi_loss


def D_logistic_r1_ft(G, D, MI, opt, training_set, minibatch_size, reals, labels, gamma=10.0, mi_weight=0.000001):
    print('D feature loss...')
    loss, reg = D_logistic_r1(G, D, opt, training_set, minibatch_size, reals, labels, gamma)
    # TODO split to D only
    _, _, mi_loss = FeatAE_l2(MI, G, D, opt, training_set, minibatch_size)
    loss += mi_weight * mi_loss
    return loss, reg, mi_loss


#----------------------------------------------------------------------------
