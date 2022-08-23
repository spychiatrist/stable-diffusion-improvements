import argparse, os, sys, glob
from ctypes import alignment
from ast import arg
import torch
import hashlib as hl
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
        default=40,
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
        default=1,
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
        default=1,
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
        default=9.0,
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
                    if opt.interactive:
                        window.write_event_value("-READY-", None)
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

                    def interactiveCallback( i ):
                        window.write_event_value('-ITER-', i)

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
                                                            x_T=start_code,
                                                            callback=interactiveCallback if opt.interactive else None)

                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                            for i, x_sample in enumerate(x_samples_ddim):
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                if opt.interactive:
                                    _metadata = (img, opt.__dict__.copy(), i)
                                    window.write_event_value("-IMAGE-", _metadata)
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

window: sg.Window = None

def make_ui():
    global opt
    global window
    print("Starting UI...")

    def TextLabel(text): return sg.Text(text+':', justification='r', size=(15,1))
    def ThumbnailImage(key): 
        _l = [[sg.Image(size=(64,64), subsample=4, key=key, p=2, background_color=sg.theme_background_color(), enable_events=True)]]
        return sg.Frame("", _l, p=1, background_color=sg.theme_button_color()[1], key=key+'_f' )

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
        [
            sg.Checkbox('Skip Sample Save',key='skip_save', default=opt.skip_save),
            sg.Checkbox('Skip Grid Save',key='skip_grid', default=opt.skip_grid)
        ],
        [sg.Button('Generate', size=(20,2), disabled=True), sg.ProgressBar(k='GenerateProgress', max_value=100, s=(30, 20), orientation='h')],
        [sg.HSeparator()],
        [sg.Text('Viewer Options', font='Any 13')],
        [
            sg.Checkbox('Auto-focus view on new image', key='_auto_focus', default=True),
        ],
        ]

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
        [sg.Text('Sample:', justification='l', expand_x=True, size=(60,4), key='SampleInfo')],
        [sg.Button('Clear All', size=(8,1), key='-CLEARALL-'), sg.Push(), sg.Button('Save', size=(12,1), key='-SAVE-'), sg.Button('Save-All', size=(12,1), key='-SAVEALL-')], 
    ]

    layout = [
        [sg.Col(layout_settings, p=0), sg.VSeparator(), sg.Col(layout_imageviewer, p=0)]
    ]

    window = sg.Window('txt2img Interactive', layout, size=(1440,900), finalize=True)




sem_generate = threading.Semaphore(1)

curr_sample_i = 0

def ui_thread():
    global opt
    global curr_sample_i
    make_ui()

    datalist = []
    imgKeys = [
                'hist0',
                'hist1',
                'hist2',
                'hist3',
                'hist4',
                'hist5',
                'hist6',
                ]
    itercount = 0

    blankImg = Image.new('RGB', (512, 512), sg.theme_button_color()[1])

    def SetSampleAndInfo(index):
        global curr_sample_i
        if index < len(datalist) and index >= 0:
            _img, _options, _samplenum = datalist[index]
            window['SampleInfo'].update(f"Sample: seed {_options['seed']}:{_options['seed_offset']}, sample {_samplenum}.\n{_options['ddim_steps']} substeps, g_scale: {_options['scale']}\n\"{_options['prompt']}\"")
            window['Image'].update(data=ImageTk.PhotoImage(_img))
            curr_sample_i = index
        else:
            window['SampleInfo'].update('Sample: None')
            window['Image'].update(data=ImageTk.PhotoImage(blankImg), size=(512,512))
            
        for i, imgKey in enumerate(imgKeys):
            if i == curr_sample_i:
                window[imgKey+'_f'].update(value="S")
            else:
                window[imgKey+'_f'].update(value="")


    def SaveImage(index):
        _img, _options, _i = datalist[index]

        if '_saved' in _options:
            return
        _options['_saved'] = True
        prompthash = hl.sha256(bytes(_options['prompt'], 'utf-8')).hexdigest()
        _path = os.path.join(_options['outdir'], "interactive", prompthash[:16])
        os.makedirs(_path, exist_ok=True)
        _file = os.path.join(_path, f"{_options['seed']:08}-{_options['seed_offset']:02}-{_i}.png")
        _img.save(_file)


    window.bind('<Right>', '-R-ARROW-')
    window.bind('<Left>', '-L-ARROW-')
    window.bind('<Escape>', '-ESC-')
    window.bind('<Control-s>', '-SAVE-')
    window.bind('<Control-S>', '-SAVEALL-')

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

            opt.skip_grid =     values['skip_grid']
            opt.skip_save =     values['skip_save']

            itercount = 0

            window['Generate'].update(disabled=True)
            window.force_focus()
            sem_generate.release()

        elif event == '-READY-':
            window['Generate'].update(disabled=False)


        elif event == '-IMAGE-': # new image from backend event
            datalist.insert(0, values[event])
            for i, data in enumerate(datalist):
                
                if i >= len(imgKeys):
                    break
                window[imgKeys[i]].update(data=ImageTk.PhotoImage(data[0].resize((64,64), resample=Image.Resampling.BICUBIC)))
            if values['_auto_focus']: 
                SetSampleAndInfo(0)
            else:
                SetSampleAndInfo(curr_sample_i + 1)

        elif event in imgKeys: # clicked a thumbnail
            _i = imgKeys.index(event)
            SetSampleAndInfo(_i)

        elif event == '-ITER-':

            window['GenerateProgress'].update(current_count=values[event], max=opt.ddim_steps)
            
        elif event == '-SAVE-':
            if len(datalist) > 0:
                SaveImage(curr_sample_i)

        elif event == '-CLEARALL-':
            datalist.clear()
            for key in imgKeys:
                window[key].update(data=ImageTk.PhotoImage(blankImg.resize((64,64), resample=Image.Resampling.BICUBIC)), size=(64,64))
            SetSampleAndInfo(0)

        elif event == '-SAVEALL-':
            for i in reversed(range(0, len(datalist))):
                SaveImage(i) #save in reverse order of history to get newest versions of overwrites last.

        # events that can happen only with global focus (i.e. focus is not a textbox)
        elif window.find_element_with_focus() == None:
        
            if event == '-R-ARROW-':
                i_attempt = max(0, min(len(datalist) - 1, curr_sample_i + 1))
                SetSampleAndInfo(i_attempt)

            elif event == '-L-ARROW-':
                i_attempt = max(0, curr_sample_i - 1)
                SetSampleAndInfo(i_attempt)


        elif event == '-ESC-':
            window.force_focus()
        
        


        if event == sg.WIN_CLOSED:
            os._exit(1)


    



if __name__ == "__main__":
    main()
