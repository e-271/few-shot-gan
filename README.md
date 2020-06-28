## Few-shot GAN Official TensorFlow Code

![Teaser image](./docs/stylegan2-teaser-1024x256.png)

**Few-shot Domain Transfer for Generative Adversarial Networks**<br>
Esther Robb, Jiarui Xu, Vincent Chu, Abhishek Kumar, Jia-Bin Huang<br>

Paper: <br>
Website: https://e-271.github.io/few-shot-gan<br>

Abstract:

We present Few-shot GAN, a novel approach to adapt a pretrained GAN model to generate new images with only a few training images from a target domain.
Inspired by component analysis techniques, which have shown to relax training data requirements, we repurpose PCA decomposition on the pretrained GAN's weights to evolve the pretrained GAN into a target-specific GAN.
Instead of finetuning the entire GAN or just batch statistics as in alternative methods, our method learns to adjust the obtained principal components to reshape the weight subspace, and strikes a balance between parameter efficiency and diversity.
To detect multi-mode overfitting in the few-shot training, we develop a novel metric to monitor latent space smoothness during the training process.
Unlike most GANs that often require large-scale training datasets (usually tens or hundreds of thousands of images), the proposed method achieves high-quality, diverse image synthesis in a data-efficient setting (as few as 25 images).
We demonstrate the effectiveness of our approach on both in-domain and cross-domain transfer settings, including face image personalization and image stylization (e.g. churches &rarr; Van Gogh landscapes and faces &rarr; painting/anime portrait).
We demonstrate qualitative and quantitative results against competing methods across several datasets.

## Requirements

Our code is build on StyleGAN2 with no additional requirements. You can follow their directions here: 

[StyleGAN2 Requirements](https://github.com/NVlabs/stylegan2#requirements).

## Preparing datasets

To prepare a few-shot dataset from a folder containing images:

```
python dataset_tool.py create_from_images /path/to/target/tfds /path/to/source/folder --resolution 1024
```

If you want to evaluate FID, you may want to create a small dataset for few-shot training and a larger dataset for evaluation.


## Training networks

Our networks start with pretrained checkpoint pickle from vanilla StyleGAN2 `config-f`, which can be downloaded from here:

[StyleGAN2 Checkpoints](https://drive.google.com/corp/drive/folders/1yanUI9m4b4PWzR0eurKNq6JR1Bbfbh6L)

To adapt a pretrained checkpoint to a new dataset, use the following commands.

PCA (our method):
```
data_root=/path/to/data/root
train_dir=relative/path/to/train
eval_dir=relative/path/to/eval
pretrain_pickle=/path/to/stylegan2/pickle

python run_training.py \
--data-dir=$data_root \
--dataset-train=$train_dir \
--dataset-eval=$eval_dir \
--resume-pkl=$pretrain_pickle \
--max-images=25 \
--config=config-pc-all\
--lrate-base=0.003 \
```

Transfer GAN:
```
python run_training.py \
--data-dir=$data_root \
--dataset-train=$train_dir \
--dataset-eval=$eval_dir \
--resume-pkl=$pretrain_pickle \
--max-images=25 \
--config=config-f\
--lrate-base=0.0003 \
```

FreezeD:
```
python run_training.py \
--data-dir=$data_root \
--dataset-train=$train_dir \
--dataset-eval=$eval_dir \
--resume-pkl=$pretrain_pickle \
--max-images=25 \
--config=config-f\
--lrate-base=0.0003 \
--freeze-d=1
```

Scale & Shift GAN:
```
python run_training.py \
--data-dir=$data_root \
--dataset-train=$train_dir \
--dataset-eval=$eval_dir \
--resume-pkl=$pretrain_pickle \
--max-images=25 \
--config=config-ss\
--lrate-base=0.003 \
```


## Image generation

To generate additional samples from a pretrained model:

```
python run_generator.py generate-images --network=/path/to/network/pickle --seeds=0-100
```

## Pretrained networks

We provide some pretrained network checkpoints in Drive: 

https://drive.google.com/drive/folders/1uRwA-HspeoQF9k-6AmotEtCH7tsFTjHI?usp=sharing

## License

As a modification of the official StyleGAN2 code, this work inherits the Nvidia Source Code License-NC. To view a copy of this license, visit https://nvlabs.github.io/stylegan2/license.html

## Citation

```
@article{robb2020fewshotgan,
  title   = {Few-shot Domain Transfer for Generative Adversarial Networks.},
  author  = {Esther Robb and Jiarui Xu and Vincent Chu and Abhishek Kumar and Jia-Bin Huang},
  journal = {arXiv},
  volume  = {},
  year    = {2020},
}
```

## Acknowledgements

We thank Mark Sandler and Andrey Zhmoginov for their input and support.
