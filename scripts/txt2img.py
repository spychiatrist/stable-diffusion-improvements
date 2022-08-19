import argparse, os, sys, glob
from ast import arg
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image,ImageTk
import PySimpleGUI as sg
import tkinter as tk
import threading
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

opt = None

def main():
    global window
    global opt
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--interactive",
        action='store_true',
        help="run in interactive mode",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--seed_offset",
        type=int,
        default=0,
        help="how many samples forward should the seed state emulate",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    if opt.interactive:
        threading.Thread(target=ui_thread).start()
        sem_generate.acquire()

    seed_everything(opt.seed)
    
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():


                while True:
                    #await mutex to proceed
                    sem_generate.acquire()

                    os.makedirs(opt.outdir, exist_ok=True)
                    outpath = opt.outdir

                    batch_size = opt.n_samples
                    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
                    

                    sample_path = os.path.join(outpath, "samples")
                    os.makedirs(sample_path, exist_ok=True)
                    base_count = len(os.listdir(sample_path))
                    grid_count = len(os.listdir(outpath)) - 1

                    start_code = None
                    if opt.fixed_code:
                        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)


                    if not opt.from_file:
                        prompt = opt.prompt
                        assert prompt is not None
                        data = [batch_size * [prompt]]

                    else:
                        print(f"reading prompts from {opt.from_file}")
                        with open(opt.from_file, "r") as f:
                            data = f.read().splitlines()
                            n_rows = len(data)
                            data = list(chunk(data, batch_size))

                    

                    tic = time.time()
                    all_samples = list()
                    for n in trange(opt.n_iter, desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):

                            seed_everything(opt.seed)
                            uc = None
                            if opt.scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)
                            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                            samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                            conditioning=c,
                                                            batch_size=opt.n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=opt.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=opt.ddim_eta,
                                                            seed_offset=opt.seed_offset,
                                                            x_T=start_code)

                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                            for x_sample in x_samples_ddim:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                if opt.interactive:
                                    window.write_event_value("-IMAGE-", img)
                                if not opt.skip_save:
                                    img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                                    base_count += 1

                            if not opt.skip_grid:
                                all_samples.append(x_samples_ddim)

                            opt.seed += 1

                    if not opt.skip_grid:
                        # additionally, save as grid
                        grid = torch.stack(all_samples, 0)
                        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                        grid = make_grid(grid, nrow=n_rows)

                        # to image
                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                        grid_count += 1

                    toc = time.time()

                    if not opt.interactive:
                        break


    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")

window = None

def make_ui():
    global opt
    global window
    print("Starting UI...")

    def TextLabel(text): return sg.Text(text+':', justification='r', size=(15,1))
    def ThumbnailImage(key): return sg.Image(size=(64,64), subsample=4, key=key, p=5, background_color=sg.theme_button_color()[1], enable_events=True)

    layout_settings = [
        [sg.Text('Parameters', font='Any 13')],
        [TextLabel('Prompt'), sg.Input(key='prompt', default_text=opt.prompt)],
        [TextLabel('Seed'), sg.Input(key='seed', default_text=opt.seed)],
        [TextLabel('Seed Sampler Offset'), sg.Input(key='seed_offset', default_text=opt.seed_offset)],
        [TextLabel('Sampler Substeps'), sg.Input(key='ddim_steps', default_text=opt.ddim_steps)],
        [TextLabel('Iterations'), sg.Input(key='n_iter', default_text=opt.n_iter)],
        [TextLabel('Samples'), sg.Input(key='n_samples', default_text=opt.n_samples)],
        [TextLabel('Guidance Scale'), sg.Input(key='scale', default_text=opt.scale)],
        [TextLabel('Sampler Eta'), sg.Input(key='ddim_eta', default_text=opt.ddim_eta)],
        [sg.Button('Generate', size=(30,4))],]

    layout_imageviewer = [
        [sg.Text('Results', font='Any 13')],
        [
            ThumbnailImage('hist0'), 
            ThumbnailImage('hist1'), 
            ThumbnailImage('hist2'), 
            ThumbnailImage('hist3'), 
            ThumbnailImage('hist4'), 
            ThumbnailImage('hist5'), 
            ThumbnailImage('hist6'), 
            ],
        [sg.Image(size=(512,512), key='Image', background_color=sg.theme_button_color()[1] )],
    ]

    layout = [
        [sg.Col(layout_settings, p=0), sg.VSeparator(), sg.Col(layout_imageviewer, p=0)]
    ]

    window = sg.Window('txt2img Interactive', layout, size=(1440,900))




sem_generate = threading.Semaphore(1)


def ui_thread():
    global opt

    make_ui()

    imglist = []
    imgKeys = [
                'hist0',
                'hist1',
                'hist2',
                'hist3',
                'hist4',
                'hist5',
                'hist6',
                ]

    while True:

        event, values = window.read()

        if event == 'Generate': # generate button event
            opt.prompt =        values['prompt']

            opt.seed =          int(values['seed'])
            opt.seed_offset =   int(values['seed_offset'])
            opt.ddim_steps =    int(values['ddim_steps'])
            opt.n_iter =        int(values['n_iter'])
            opt.n_samples =     int(values['n_samples'])

            opt.ddim_eta =      float(values['ddim_eta'])
            opt.scale =         float(values['scale'])
            sem_generate.release()

        elif event == '-IMAGE-': # new image from backend event
            imglist.insert(0, values[event])
            for i, img in enumerate(imglist):
                if i == 0: 
                    window['Image'].update(data=ImageTk.PhotoImage(img))
                elif i >= len(imgKeys):
                    break
                window[imgKeys[i]].update(data=ImageTk.PhotoImage(img.resize((64,64), resample=Image.Resampling.BICUBIC)))

        elif event in imgKeys: # clicked a thumbnail
            i = imgKeys.index(event)
            if len(imglist) > i:
                window['Image'].update(data=ImageTk.PhotoImage(imglist[i]))
            

        if event == sg.WIN_CLOSED:
            os._exit(1)



if __name__ == "__main__":
    main()
