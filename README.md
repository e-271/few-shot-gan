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

Requirements match those of [StyleGAN2](https://github.com/NVlabs/stylegan2).

## Preparing datasets


## Training networks


## License

This work is made available under the Nvidia Source Code License-NC. To view a copy of this license, visit https://nvlabs.github.io/stylegan2/license.html

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
