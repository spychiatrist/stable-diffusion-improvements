# Stable Diffusion Improvements

## Goals

The focus of this fork of [Stable Diffusion](https://github.com/CompVis/latent-diffusion) is to improve:

1. **Workflow**
   - Sensible GUI with shortcuts and launch arguments at your fingertips
   - Quicker iteration times (model stays in-memory, among other improvements)
2. **Determinism**
   - Enhanced determinism of seeding per-sample, singly-indexable subsamples
   - Sequential seed stepping per iteration
3. **Usability**
   - Accessibility through UI
   - More concise and fool-proof installation

![interactive_usagedemo](assets/gifs/usage.gif)

  
## Installation

### Requirements

- Modern NVIDIA GPU with > 10GB VRAM

### Windows

Install [Miniconda3](https://docs.conda.io/en/latest/miniconda.html), a minimized Python virtual environment manager for Windows.

From the Start Menu, run **Anaconda Prompt (miniconda3)**.  This is the shell that you should execute all further shell commands within.

Create the environment (automatically fetches dependencies) from the base of this repository:

```
conda env create -f environment.yaml
conda activate ldm
```

Place your obtained `model.ckpt` file in new folder: `.\stable-diffusion-improvements\models\ldm\stable-diffusion-v1`

## Usage

Using the interactive mode is straightforward.  Simply call (with `ldm` conda env active):

```
python scripts\txt2img.py --interactive
```

Any classic arguments passed to txt2img.py will show up in the interactive view as parameter textbox/checkbox defaults.  Await the activation of the `Generate` button, and create to your heart's content.

More info and features coming soon, so keep your repository up to date.

## Acknowledgements
*From Stability.ai:*

>*Stable Diffusion was made possible thanks to a collaboration with [Stability AI](https://stability.ai/) and [Runway](https://runwayml.com/) and builds upon our previous work:*

[**High-Resolution Image Synthesis with Latent Diffusion Models**](https://ommer-lab.com/research/latent-diffusion-models/)<br/>
[Robin Rombach](https://github.com/rromb)\*,
[Andreas Blattmann](https://github.com/ablattmann)\*,
[Dominik Lorenz](https://github.com/qp-qp)\,
[Patrick Esser](https://github.com/pesser),
[Björn Ommer](https://hci.iwr.uni-heidelberg.de/Staff/bommer)<br/>

**CVPR '22 Oral**

which is available on [GitHub](https://github.com/CompVis/latent-diffusion). PDF at [arXiv](https://arxiv.org/abs/2112.10752). Please also visit our [Project page](https://ommer-lab.com/research/latent-diffusion-models/).

![txt2img-stable2](assets/stable-samples/txt2img/merged-0006.png)
[Stable Diffusion](#stable-diffusion-v1) is a latent text-to-image diffusion
model.
Thanks to a generous compute donation from [Stability AI](https://stability.ai/) and support from [LAION](https://laion.ai/), we were able to train a Latent Diffusion Model on 512x512 images from a subset of the [LAION-5B](https://laion.ai/blog/laion-5b/) database. 
Similar to Google's [Imagen](https://arxiv.org/abs/2205.11487), 
this model uses a frozen CLIP ViT-L/14 text encoder to condition the model on text prompts.
With its 860M UNet and 123M text encoder, the model is relatively lightweight and runs on a GPU with at least 10GB VRAM.
See [this section](#stable-diffusion-v1) below and the [model card](https://huggingface.co/CompVis/stable-diffusion).

