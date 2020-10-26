# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import argparse
import copy
import os
import sys
import json

import dnnlib
from dnnlib import EasyDict

from metrics.metric_defaults import metric_defaults

#----------------------------------------------------------------------------

_valid_configs = [
    # Table 1
    'config-a', # Baseline StyleGAN
    'config-b', # + Weight demodulation
    'config-c', # + Lazy regularization
    'config-d', # + Path length regularization
    'config-e', # + No growing, new G & D arch.
    'config-f', # + Large networks (default)

    # Domain adaptation
    'config-ada-ss',
    'config-ada-sv',
    'config-ada-sv-flat',
    'config-ada-pc',
    'config-ada-pc-flat',
    'config-fd',
    'config-da',
    'config-da-fd',

    #'config-a-gb',
    'config-b-g',
    'config-b-b',
    'config-b-gb', 
    'config-c-g',
    'config-c-b',
    'config-c-gb',
    'config-d-gb',

    # Table 2
    'config-e-Gorig-Dorig',   'config-e-Gorig-Dresnet',   'config-e-Gorig-Dskip',
    'config-e-Gresnet-Dorig', 'config-e-Gresnet-Dresnet', 'config-e-Gresnet-Dskip',
    'config-e-Gskip-Dorig',   'config-e-Gskip-Dresnet',   'config-e-Gskip-Dskip',
]

#----------------------------------------------------------------------------

def run(g_loss, g_loss_kwargs, d_loss, d_loss_kwargs, dataset_train, dataset_eval, data_dir, result_dir, config_id, num_gpus, total_kimg, gamma, mirror_augment, metrics, resume_pkl, resume_kimg, resume_pkl_dir, max_images, lrate_base, img_ticks, net_ticks, skip_images):

    if g_loss_kwargs != '': g_loss_kwargs = json.loads(g_loss_kwargs)
    else: g_loss_kwargs = {}
    if d_loss_kwargs != '': d_loss_kwargs = json.loads(d_loss_kwargs)
    else: d_loss_kwargs = {}

    train     = EasyDict(run_func_name='training.training_loop.training_loop') # Options for training loop.
    G         = EasyDict(func_name='training.networks_stylegan2.G_main')       # Options for generator network.
    D         = EasyDict(func_name='training.networks_stylegan2.D_stylegan2')  # Options for discriminator network.
    G_opt     = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                  # Options for generator optimizer.
    D_opt     = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                  # Options for discriminator optimizer.
    G_loss    = EasyDict(func_name='training.loss.' + g_loss, **g_loss_kwargs) #G_logistic_ns_gsreg')      # Options for generator loss.
    D_loss    = EasyDict(func_name='training.loss.' + d_loss, **d_loss_kwargs) # Options for discriminator loss.
    sched     = EasyDict()                                                     # Options for TrainingSchedule.
    grid      = EasyDict(size='1080p', layout='random')                           # Options for setup_snapshot_image_grid().
    sc        = dnnlib.SubmitConfig()                                          # Options for dnnlib.submit_run().
    tf_config = {'rnd.np_random_seed': 1000}                                   # Options for tflib.init_tf().


    train.total_kimg = total_kimg
    train.mirror_augment = mirror_augment
    train.image_snapshot_ticks = img_ticks
    train.network_snapshot_ticks = net_ticks
    G.scale_func = 'training.networks_stylegan2.apply_identity'
    D.scale_func = None
    sched.G_lrate_base = sched.D_lrate_base = lrate_base #0.002
    # TODO: Changed this to 16 to match DiffAug
    sched.minibatch_size_base = 16
    sched.minibatch_gpu_base = 4
    D_loss.gamma = 10
    metrics = [metric_defaults[x] for x in metrics]
    desc = 'stylegan2'
    sched.tick_kimg_base = 1
    sched.tick_kimg_dict = {} #{8:28, 16:24, 32:20, 64:16, 128:12, 256:8, 512:6, 1024:4}): # Resolution-specific overrides.


    desc += '-' + dataset_train.split('/')[-1]
    # Get dataset paths
    t_path = dataset_train.split('/')
    e_path = dataset_eval.split('/')
    if len(t_path) > 1:
        dataset_train = t_path[-1]
        train.train_data_dir = os.path.join(data_dir, '/'.join(t_path[:-1]))
    if len(e_path) > 1:
        dataset_eval = e_path[-1]
        train.eval_data_dir = os.path.join(data_dir, '/'.join(e_path[:-1]))
    dataset_args = EasyDict(tfrecord_dir=dataset_train)
    # Limit number of training images during train (not eval)
    dataset_args['max_images'] = max_images
    if max_images: desc += '-%dimg' % max_images
    dataset_args['skip_images'] = skip_images
    dataset_args_eval = EasyDict(tfrecord_dir=dataset_eval)
    desc += '-' + dataset_eval

    assert num_gpus in [1, 2, 4, 8]
    sc.num_gpus = num_gpus

    assert config_id in _valid_configs
    desc += '-' + config_id

    if mirror_augment: desc += '-aug'

    # Infer pretrain checkpoint from target dataset
    if not resume_pkl:
        if any(ds in dataset_train.lower() for ds in ['hat', 'obama', 'celeba', 'rem', 'portrait']):
            resume_pkl = 'ffhq-config-f.pkl'
        if any(ds in dataset_train.lower() for ds in ['gogh', 'temple', 'tower', 'medici', 'bridge']):
            resume_pkl = 'church-config-f.pkl'
        if any(ds in dataset_train.lower() for ds in ['bus', 'boat', 'bike']):
            resume_pkl = 'car-config-f.pkl'
    resume_pkl = os.path.join(resume_pkl_dir, resume_pkl)
    train.resume_pkl = resume_pkl
    train.resume_kimg = resume_kimg

    train.resume_with_new_nets = True # Recreate with new parameters
    # Adaptive parameters
    if 'ada' in config_id:
        G['train_scope'] = D['train_scope'] = '.*adapt' # Freeze old parameters
        if 'ss' in config_id:
            G['adapt_func'] = D['adapt_func'] = 'training.networks_stylegan2.apply_adaptive_scale_shift'
        if 'sv' or 'pc' in config_id: # [:9] == 'config-sv' or config_id[:9] == 'config-pc':
            G['map_svd'] = G['syn_svd'] = D['svd'] = True
            # Flatten over spatial dimension
            if 'flat' in config_id:
                G['spatial'] = D['spatial'] = True
            # Do PCA by centering before SVD
            if 'pc' in config_id:
                G['svd_center'] = D['svd_center'] = True
            G['svd_config'] = D['svd_config'] = 'S' 
            if 'U' in config_id:
                G['svd_config'] += 'U'
                D['svd_config'] += 'U' 
            if 'V' in config_id:
                G['svd_config'] += 'V'
                D['svd_config'] += 'V' 
    # FreezeD
    D['freeze'] = 'fd' in config_id #freeze_d

    if gamma is not None:
        D_loss.gamma = gamma

    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    kwargs = EasyDict(train)

    kwargs.update(G_args=G, D_args=D,
                  G_opt_args=G_opt, D_opt_args=D_opt,
                  G_loss_args=G_loss, D_loss_args=D_loss,)
    kwargs.update(dataset_args=dataset_args, dataset_args_eval=dataset_args_eval, sched_args=sched, grid_args=grid, metric_arg_list=metrics, tf_config=tf_config)
    kwargs.submit_config = copy.deepcopy(sc)
    kwargs.submit_config.run_dir_root = result_dir
    kwargs.submit_config.run_desc = desc
    dnnlib.submit_run(**kwargs)

#----------------------------------------------------------------------------

def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def _parse_comma_sep(s):
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

_examples = '''examples:

  # Train StyleGAN2 using the FFHQ dataset
  python %(prog)s --num-gpus=8 --data-dir=~/datasets --config=config-f --dataset=ffhq --mirror-augment=true

valid configs:

  ''' + ', '.join(_valid_configs) + '''

valid metrics:

  ''' + ', '.join(sorted([x for x in metric_defaults.keys()])) + '''

'''

def main():
    parser = argparse.ArgumentParser(
        description='Train StyleGAN2.',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    parser.add_argument('--data-dir', help='Dataset root directory', required=True)
    parser.add_argument('--dataset-train', help='Training dataset', required=True)
    parser.add_argument('--dataset-eval', help='Evaluation dataset (defaults to training dataset)', default=None)
    parser.add_argument('--config', help='Training config (default: %(default)s)', default='config-pc-all', required=True, dest='config_id', metavar='CONFIG')
    parser.add_argument('--g-loss', help='Import path to generator loss function.', default='G_logistic_ns_pathreg', required=False)
    parser.add_argument('--d-loss', help='Import path to generator loss function.', default='D_logistic_r1', required=False)
    parser.add_argument('--g-loss-kwargs', help='JSON-formatted keyword arguments for generator loss function.', default='', required=False)
    parser.add_argument('--d-loss-kwargs', help='JSON-formatted keyword arguments for discriminator loss function.', default='', required=False)
    parser.add_argument('--max-images', help='Maximum number of images to pull from dataset.', default=None, type=int)
    parser.add_argument('--skip-images', help='Number of images to skip, set negative for random seed', default=None, type=int)
    parser.add_argument('--num-gpus', help='Number of GPUs (default: %(default)s)', default=1, type=int, metavar='N')
    parser.add_argument('--total-kimg', help='Training length in thousands of images (default: %(default)s)', metavar='KIMG', default=100, type=int)
    parser.add_argument('--gamma', help='R1 regularization weight (default is config dependent)', default=None, type=float)
    parser.add_argument('--mirror-augment', help='Mirror augment (default: %(default)s)', default=True, metavar='BOOL', type=_str_to_bool)
    parser.add_argument('--metrics', help='Comma-separated list of metrics or "none" (default: %(default)s)', default='fid1k,ppgs1k', type=_parse_comma_sep)
    parser.add_argument('--resume-pkl', help='Network pickle name', default='')
    parser.add_argument('--resume-pkl-dir', help='Directory of network pickles', default='pickles', metavar='DIR')
    parser.add_argument('--lrate-base', help='Base learning rate for G and D', default=0.002, type=float)
    parser.add_argument('--resume-kimg', help='kimg to resume from, affects scheduling', default=0, type=int)
    parser.add_argument('--img-ticks', help='How often to save images', default=1, type=int)
    parser.add_argument('--net-ticks', help='How often to save network snapshots', default=4, type=int)

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print ('Error: dataset root directory does not exist.')
        sys.exit(1)

    if args.config_id not in _valid_configs:
        print ('Error: --config value must be one of: ', ', '.join(_valid_configs))
        sys.exit(1)

    for metric in args.metrics:
        if metric not in metric_defaults:
            print ('Error: unknown metric \'%s\'' % metric)
            sys.exit(1)

    run(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------

