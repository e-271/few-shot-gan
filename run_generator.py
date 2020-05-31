# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import os
import sys
from dnnlib import EasyDict
from training import dataset
from training import networks_stylegan2

import pretrained_networks
import tensorflow as tf

#----------------------------------------------------------------------------

# def generate_images(network_pkl, seeds, truncation_psi, layer_toggle, layer_dset, layer_ddir):
def generate_images(network_pkl, seeds, truncation_psi, layer_toggle):
    # For residual adapter layerwise contribution plots
    # Instructions for hackage:
    # vi +552 training/networks_stylegan2.py
    # rho_in * (latent_idx == <layer>)

    # For layerwise visualization of PCA-GAN
    # if layer_toggle:
    #     print('Loading networks from "%s"...' % network_pkl)
    #     rG, rD, rGs = pretrained_networks.load_networks(network_pkl)
    #
    #     G_lambda_mask = {var: np.ones(rGs.vars[var].shape[-1]) for var in rGs.vars if 'SVD/s' in var}
    #     # G_lambda_mask = {var: 1./tflib.run(rGs.vars[var]) for var in rGs.vars if 'SVD/s' in var}
    #     D_lambda_mask = {'D/' + var: np.ones(rD.vars[var].shape[-1]) for var in rD.vars if 'SVD/s' in var}
    #
    #     G_args         = EasyDict(func_name='training.networks_stylegan2.G_main')       # Options for generator network.
    #     D_args         = EasyDict(func_name='training.networks_stylegan2.D_stylegan2')  # Options for discriminator network.
    #     G_args['spatial'] = D_args['spatial'] = True
    #     G_args['map_svd'] = G_args['syn_svd'] = D_args['svd'] = True
    #     G_args['svd_center'] = D_args['svd_center'] = True
    #     G_args['factorized'] = D_args['factorized'] = True
    #     G_args['lambda_mask'] = G_lambda_mask
    #     D_args['lambda_mask'] = D_lambda_mask
    #
    #     dset="anime25" # layer_dset # "anime25"
    #     dataset_args = EasyDict(tfrecord_dir=dset)
    #     data_dir="/work/newriver/jiaruixu/datasets/mini/faces" # layer_ddir # "/mnt/slow_ssd/erobb/datasets"
    #     # Load training set.
    #     dnnlib.tflib.init_tf()
    #     training_set = dataset.load_dataset(data_dir=dnnlib.convert_path(data_dir), verbose=True, **dataset_args)
    #     print('Constructing networks...')
    #     _G = tflib.Network('G', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size, **G_args)
    #     _D = tflib.Network('D', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size, **D_args)
    #     Gs = _G.clone('Gs')
    #
    #     _G.copy_vars_from(rG);
    #     _D.copy_vars_from(rD);
    #     Gs.copy_vars_from(rGs)
    # else:
    #     print('Loading networks from "%s"...' % network_pkl)
    #     _G, _D, Gs = pretrained_networks.load_networks(network_pkl)

    # For layerwise visualization of PCA-GAN
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)

    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False

    # Gs_lambda_mask = {var: np.ones(Gs.vars[var].shape[-1]) for var in Gs.vars if 'SVD/s' in var}
    Gs_lambda_mask = {var: 1./tflib.run(Gs.vars[var[:-1]+'adapt/lambda']) for var in Gs.vars if 'SVD/s' in var}
    # Gs_kwargs['lambda_mask'] = Gs_lambda_mask

    G_lambda_lists = [var for var in Gs.vars if 'SVD/s' in var]

    if truncation_psi is not None:
        Gs_kwargs.truncation_psi = truncation_psi

    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        rho = np.array([1])

        if layer_toggle == 1:
            images = Gs.run(z, None, rho, lambda_mask=Gs_lambda_mask, **Gs_kwargs) # [minibatch, height, width, channel]
            layer_fakes_map = [images]
            layer_fakes_syn = [images]
            for name in G_lambda_lists:
                # Get variables under name scope as tensor
                # lambdas = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)[0]
                # # Get values as numpy arrays
                # lambda_values = tflib.run(lambdas)
                # Gs_lambda_mask[name] = 1. / lambda_values
                print(name)
                Gs_lambda_mask[name] = np.ones(Gs.vars[name].shape[-1])
                images = Gs.run(z, None, rho, lambda_mask=Gs_lambda_mask, **Gs_kwargs) # [minibatch, height, width, channel]

                save_name = name.replace('/', '_')
                PIL.Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path('%s_seed%04d.jpg' % (save_name, seed)))

                if 'G_mapping' in name:
                    layer_fakes_map.append(images)
                else:
                    layer_fakes_syn.append(images)
                # sz=10
                #
                # terp_start, terp_stop = z, rnd.randn(1, *Gs.input_shape[1:])
                # terp_latent = np.linspace(terp_start, terp_stop, sz)
                # terp_fakes = []
                # for j in range(sz):
                #     terp_fake = Gs.run(terp_latent[j], None, rho, lambda_mask=Gs_lambda_mask, **Gs_kwargs)
                #     terp_fakes.append(terp_fake)
                # terp_fakes=np.concatenate(terp_fakes, 2)
                # print(terp_fakes.shape)
                # PIL.Image.fromarray(terp_fakes[0], 'RGB').save(dnnlib.make_run_dir_path('terp_latent_%s_seed%04d.jpg' % (save_name, seed)))

                # set the value back to 1 / lambdas
                Gs_lambda_mask[name] = 1./ tflib.run(Gs.vars[name[:-1]+'adapt/lambda'])

            layer_fakes_map=np.concatenate(layer_fakes_map, 2)
            PIL.Image.fromarray(layer_fakes_map[0], 'RGB').save(dnnlib.make_run_dir_path('G_mapping_seed%04d.jpg' % (seed)))
            layer_fakes_syn=np.concatenate(layer_fakes_syn, 2)
            PIL.Image.fromarray(layer_fakes_syn[0], 'RGB').save(dnnlib.make_run_dir_path('G_synthesis_seed%04d.jpg' % (seed)))
        elif layer_toggle == 2:
            # toggle mapping layers / synthesis layers
            # do not use adaptive lambdas
            images = Gs.run(z, None, rho, lambda_mask=Gs_lambda_mask, **Gs_kwargs)
            layer_fakes = [images]
            # keep mapping adaptive lambdas
            for name in G_lambda_lists:
                if 'G_mapping' in name:
                    print(name)
                    Gs_lambda_mask[name] = np.ones(Gs.vars[name].shape[-1])
            images = Gs.run(z, None, rho, lambda_mask=Gs_lambda_mask, **Gs_kwargs) # [minibatch, height, width, channel]
            layer_fakes.append(images)
            # kepp synthesis adaptive lambdas
            Gs_lambda_mask = {var: 1./tflib.run(Gs.vars[var[:-1]+'adapt/lambda']) for var in Gs.vars if 'SVD/s' in var}
            for name in G_lambda_lists:
                if 'G_synthesis' in name:
                    print(name)
                    Gs_lambda_mask[name] = np.ones(Gs.vars[name].shape[-1])
            images = Gs.run(z, None, rho, lambda_mask=Gs_lambda_mask, **Gs_kwargs) # [minibatch, height, width, channel]
            layer_fakes.append(images)
            # use all adaptive lambdas
            Gs_lambda_mask = {var: np.ones(Gs.vars[var].shape[-1]) for var in Gs.vars if 'SVD/s' in var}
            images = Gs.run(z, None, rho, lambda_mask=Gs_lambda_mask, **Gs_kwargs) # [minibatch, height, width, channel]
            layer_fakes.append(images)
            # concatenate
            layer_fakes=np.concatenate(layer_fakes, 2)
            PIL.Image.fromarray(layer_fakes[0], 'RGB').save(dnnlib.make_run_dir_path('G_seed%04d.jpg' % (seed)))
        elif layer_toggle == 3:
            # toggle mapping layers / synthesis layers
            # do not use adaptive lambdas
            images = Gs.run(z, None, rho, lambda_mask=Gs_lambda_mask, **Gs_kwargs)
            layer_fakes = [images]
            # keep mapping adaptive lambdas
            for name in G_lambda_lists:
                if 'G_mapping' in name:
                    if 'Dense0' in name or 'Dense1' in name or 'Dense2' in name or 'Dense3' in name:
                        print(name)
                        Gs_lambda_mask[name] = np.ones(Gs.vars[name].shape[-1])
            images = Gs.run(z, None, rho, lambda_mask=Gs_lambda_mask, **Gs_kwargs) # [minibatch, height, width, channel]
            layer_fakes.append(images)
            #
            Gs_lambda_mask = {var: 1./tflib.run(Gs.vars[var[:-1]+'adapt/lambda']) for var in Gs.vars if 'SVD/s' in var}
            for name in G_lambda_lists:
                if 'G_mapping' in name:
                    if 'Dense4' in name or 'Dense5' in name or 'Dense6' in name or 'Dense7' in name:
                        print(name)
                        Gs_lambda_mask[name] = np.ones(Gs.vars[name].shape[-1])
            images = Gs.run(z, None, rho, lambda_mask=Gs_lambda_mask, **Gs_kwargs) # [minibatch, height, width, channel]
            layer_fakes.append(images)
            # kepp synthesis adaptive lambdas
            Gs_lambda_mask = {var: 1./tflib.run(Gs.vars[var[:-1]+'adapt/lambda']) for var in Gs.vars if 'SVD/s' in var}
            for name in G_lambda_lists:
                if 'G_synthesis' in name and 'Conv' in name:
                    print(name)
                    Gs_lambda_mask[name] = np.ones(Gs.vars[name].shape[-1])
            images = Gs.run(z, None, rho, lambda_mask=Gs_lambda_mask, **Gs_kwargs) # [minibatch, height, width, channel]
            layer_fakes.append(images)
            # kepp synthesis adaptive lambdas
            Gs_lambda_mask = {var: 1./tflib.run(Gs.vars[var[:-1]+'adapt/lambda']) for var in Gs.vars if 'SVD/s' in var}
            for name in G_lambda_lists:
                if 'G_synthesis' in name and 'Conv0_up' in name:
                    print(name)
                    Gs_lambda_mask[name] = np.ones(Gs.vars[name].shape[-1])
            images = Gs.run(z, None, rho, lambda_mask=Gs_lambda_mask, **Gs_kwargs) # [minibatch, height, width, channel]
            layer_fakes.append(images)
            # kepp synthesis adaptive lambdas
            Gs_lambda_mask = {var: 1./tflib.run(Gs.vars[var[:-1]+'adapt/lambda']) for var in Gs.vars if 'SVD/s' in var}
            for name in G_lambda_lists:
                if 'G_synthesis' in name and 'Conv1' in name:
                    print(name)
                    Gs_lambda_mask[name] = np.ones(Gs.vars[name].shape[-1])
            images = Gs.run(z, None, rho, lambda_mask=Gs_lambda_mask, **Gs_kwargs) # [minibatch, height, width, channel]
            layer_fakes.append(images)
            # use all adaptive lambdas
            Gs_lambda_mask = {var: np.ones(Gs.vars[var].shape[-1]) for var in Gs.vars if 'SVD/s' in var}
            images = Gs.run(z, None, rho, lambda_mask=Gs_lambda_mask, **Gs_kwargs) # [minibatch, height, width, channel]
            layer_fakes.append(images)
            # concatenate
            layer_fakes=np.concatenate(layer_fakes, 2)
            PIL.Image.fromarray(layer_fakes[0], 'RGB').save(dnnlib.make_run_dir_path('G_seed%04d.jpg' % (seed)))
        else:
            # not toggle layers
            images = Gs.run(z, None, rho, **Gs_kwargs) # [minibatch, height, width, channel]
            PIL.Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path('seed%04d.jpg' % seed))

            # sz=10
            # terp_fakes = []
            # terp_rhos = np.linspace(0,1,sz)
            # i = 0
            # for j in range(sz): # col
            #    terp_fake = Gs.run(z, None, terp_rhos[j:j+1], **Gs_kwargs)
            #    terp_fakes.append(terp_fake)
            # terp_fakes = np.concatenate(terp_fakes, 2)
            # print(terp_fakes.shape)
            # PIL.Image.fromarray(terp_fakes[0], 'RGB').save(dnnlib.make_run_dir_path('terp_rho_seed%04d.jpg' % seed))

            terp_start, terp_stop = z, rnd.randn(1, *Gs.input_shape[1:])
            terp_latent = np.linspace(terp_start, terp_stop, sz)
            terp_fakes = []
            for j in range(sz):
                terp_fake = Gs.run(terp_latent[j], None, rho, **Gs_kwargs)
                terp_fakes.append(terp_fake)
            terp_fakes=np.concatenate(terp_fakes, 2)
            print(terp_fakes.shape)
            PIL.Image.fromarray(terp_fakes[0], 'RGB').save(dnnlib.make_run_dir_path('terp_latent_seed%04d.jpg' % seed))


#----------------------------------------------------------------------------

def style_mixing_example(network_pkl, row_seeds, col_seeds, truncation_psi, col_styles, minibatch_size=4):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    w_avg = Gs.get_var('dlatent_avg') # [component]

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = minibatch_size

    print('Generating W vectors...')
    all_seeds = list(set(row_seeds + col_seeds))
    all_z = np.stack([np.random.RandomState(seed).randn(*Gs.input_shape[1:]) for seed in all_seeds]) # [minibatch, component]
    all_w = Gs.components.mapping.run(all_z, None) # [minibatch, layer, component]
    all_w = w_avg + (all_w - w_avg) * truncation_psi # [minibatch, layer, component]
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))} # [layer, component]

    print('Generating images...')
    all_images = Gs.components.synthesis.run(all_w, **Gs_syn_kwargs) # [minibatch, height, width, channel]
    image_dict = {(seed, seed): image for seed, image in zip(all_seeds, list(all_images))}

    print('Generating style-mixed images...')
    for row_seed in row_seeds:
        for col_seed in col_seeds:
            w = w_dict[row_seed].copy()
            w[col_styles] = w_dict[col_seed][col_styles]
            image = Gs.components.synthesis.run(w[np.newaxis], **Gs_syn_kwargs)[0]
            image_dict[(row_seed, col_seed)] = image

    print('Saving images...')
    for (row_seed, col_seed), image in image_dict.items():
        PIL.Image.fromarray(image, 'RGB').save(dnnlib.make_run_dir_path('%d-%d.jpg' % (row_seed, col_seed)))

    print('Saving image grid...')
    _N, _C, H, W = Gs.output_shape
    canvas = PIL.Image.new('RGB', (W * (len(col_seeds) + 1), H * (len(row_seeds) + 1)), 'black')
    for row_idx, row_seed in enumerate([None] + row_seeds):
        for col_idx, col_seed in enumerate([None] + col_seeds):
            if row_seed is None and col_seed is None:
                continue
            key = (row_seed, col_seed)
            if row_seed is None:
                key = (col_seed, col_seed)
            if col_seed is None:
                key = (row_seed, row_seed)
            canvas.paste(PIL.Image.fromarray(image_dict[key], 'RGB'), (W * col_idx, H * row_idx))
    canvas.save(dnnlib.make_run_dir_path('grid.jpg'))

#----------------------------------------------------------------------------

def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

_examples = '''examples:

  # Generate ffhq uncurated images (matches paper Figure 12)
  python %(prog)s generate-images --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --seeds=6600-6625 --truncation-psi=0.5

  # Generate ffhq curated images (matches paper Figure 11)
  python %(prog)s generate-images --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --seeds=66,230,389,1518 --truncation-psi=1.0

  # Generate uncurated car images (matches paper Figure 12)
  python %(prog)s generate-images --network=gdrive:networks/stylegan2-car-config-f.pkl --seeds=6000-6025 --truncation-psi=0.5

  # Generate style mixing example (matches style mixing video clip)
  python %(prog)s style-mixing-example --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --row-seeds=85,100,75,458,1500 --col-seeds=55,821,1789,293 --truncation-psi=1.0
'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='''StyleGAN2 generator.

Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    parser_generate_images = subparsers.add_parser('generate-images', help='Generate images')
    parser_generate_images.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_generate_images.add_argument('--seeds', type=_parse_num_range, help='List of random seeds', required=True)
    parser_generate_images.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser_generate_images.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    parser_generate_images.add_argument('--layer-toggle', type=int, help='Which adaptive layer to toggle', default=None, metavar='DIR')
    # parser_generate_images.add_argument('--layer-dset', help='Dataset name, needed for layer plots', default=None, metavar='DIR')
    # parser_generate_images.add_argument('--layer-ddir', help='Dataset dir, needed for layer plots', default=None, metavar='DIR')


    parser_style_mixing_example = subparsers.add_parser('style-mixing-example', help='Generate style mixing video')
    parser_style_mixing_example.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_style_mixing_example.add_argument('--row-seeds', type=_parse_num_range, help='Random seeds to use for image rows', required=True)
    parser_style_mixing_example.add_argument('--col-seeds', type=_parse_num_range, help='Random seeds to use for image columns', required=True)
    parser_style_mixing_example.add_argument('--col-styles', type=_parse_num_range, help='Style layer range (default: %(default)s)', default='0-6')
    parser_style_mixing_example.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser_style_mixing_example.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    args = parser.parse_args()
    kwargs = vars(args)
    subcmd = kwargs.pop('command')

    if subcmd is None:
        print ('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)

    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = os.path.dirname(kwargs['network_pkl']) + '/gen_%s' % kwargs['network_pkl'].split('/')[-1].split('.')[0].split('-')[-1]
    kwargs.pop('result_dir')
    sc.run_desc = subcmd

    func_name_map = {
        'generate-images': 'run_generator.generate_images',
        'style-mixing-example': 'run_generator.style_mixing_example'
    }
    dnnlib.submit_run(sc, func_name_map[subcmd], **kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
