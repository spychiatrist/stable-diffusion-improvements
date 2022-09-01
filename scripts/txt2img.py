import argparse, os, sys, glob
import json
#from ctypes import alignment
#from ast import arg
import hashlib as hl
#import numpy as np
from PIL import Image,ImageTk
from PIL.PngImagePlugin import PngInfo
import PySimpleGUI as sg
#import tkinter as tk
import threading
#from torchvision.utils import make_grid
#import time
#from torch import autocast
#from contextlib import contextmanager, nullcontext
from itertools import product



opt = None
g_init_img = None

window:sg.Window = None

def main():
    global opt
    global window
    opt = parse_args()
    

    window = make_ui()

    
    sem_generate.acquire()
    if not opt.debug_ui:
        threading.Thread(target=backend_loop).start()
    ui_thread(window)


def make_ui():
    global opt
    print("Starting UI...")

    programName = 'Stable Diffusion Interactive'

    sampler_choices = ['k_lms', 'k_euler_a', 'ddim', 'k_dpm_2_a', 'k_dpm_2', 'k_euler', 'k_heun', 'plms']
    def TextLabel(text, base_key, vis=True): return sg.Text(text+':', justification='r', size=(18,1), visible=vis, k='label_' + base_key)

    def InputRow(text, base_key, default_text, tooltip, groupable=False, group_default=False, vis=True, input_size_v=1):
        ret = [
            TextLabel(text, base_key, vis=vis),
        ]
        if input_size_v==1:
            ret.append(sg.Input(key=base_key, default_text=default_text, tooltip=tooltip, visible=vis, size=(80,1)))
        else:
            ret.append(sg.Multiline(key=base_key, default_text=default_text, tooltip=tooltip, no_scrollbar=True, visible=vis, size=(80,input_size_v)))
        if groupable:
            ret.append(GroupCheckbox(base_key, group_default))
        return ret


    def ThumbnailImage(key): 
        _l = [[sg.Image(size=(64,64), subsample=4, key=key, p=2, background_color=sg.theme_button_color()[1], enable_events=True)]]
        return sg.Frame(" ", _l, p=1, background_color=sg.theme_button_color()[1], key=key+'_f' )

    l_to_block_focus = []

    def GroupCheckbox(base_key, default:bool):
        cb = sg.Checkbox('', key='g_'+base_key, default=default, tooltip='Group by (for up/down instance navigation)')
        l_to_block_focus.append(cb)
        return cb

    layout_settings = [
        # [sg.Text('Parameters', font='Any 13')],
        [TextLabel('Sampler', 'SamplerCombo'), sg.Combo(sampler_choices, default_value=sampler_choices[0], enable_events=True, k='SamplerCombo' ), sg.Radio('txt->img', 'SamplerProcess', k='tti', default=True), sg.Radio('img->img', 'SamplerProcess', k='iti')],

        InputRow('Prompt'               , 'prompt'      , opt.prompt      , 'Description of the image you would like to see.' , input_size_v=3                                                                                          , groupable=True, group_default=True),
        InputRow('Seed'                 , 'seed'        , opt.seed        , 'Seed for first image generation.'                                                                                                                          , groupable=True, group_default=True),
        InputRow('Seed Sampler Offset'  , 'seed_offset' , opt.seed_offset , 'Numeric offset into batch.  Effectively, number of samples to skip.  Useful if you want to re-run a single image generation nested within a sample batch.' , groupable=True, group_default=True),
        InputRow('Sampler Substeps'     , 'ddim_steps'  , opt.ddim_steps  , 'Number of substeps per batch.'                                                                                                                             , groupable=True                    ),
        InputRow('Guidance Scale'       , 'scale'       , opt.scale       , 'Unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))'                                                                , groupable=True                    ),
        InputRow('Iterations'           , 'n_iter'      , opt.n_iter      , 'Number of batches per generation.'                                                                                                                                                             ),
        InputRow('Batch Size (Samples)' , 'n_samples'   , opt.n_samples   , 'Number of samples per batch.'                                                                                                                                                                  ),
        InputRow('Strength'             , 'strength'    , opt.strength    , 'Image-to-Image strength (0.0, 1.0)'                                                                                                                        , vis=False                         ),
        InputRow('Sampler Eta'          , 'ddim_eta'    , opt.ddim_eta    , 'Deprecated: ddim eta'                                                                                                                                      , vis=False                         ),

        [
            sg.Checkbox('Skip Sample Save',key='skip_save', default=opt.skip_save, tooltip='If checked, program will not save each sample automatically.'),
            sg.Checkbox('Skip Grid Save',key='skip_grid', default=opt.skip_grid, tooltip='If checked, program will not save grid collages of each batch.')
        ],
        [sg.Button('Generate', size=(20,2), disabled=True), sg.ProgressBar(k='GenerateProgress', max_value=100, s=(30, 20), orientation='h')],
        [sg.HSeparator()],
        [sg.Text('Viewer Options', font='Any 13')],
        [
            sg.Checkbox('Auto-focus', key='_auto_focus', default=True, tooltip='Whether or not to set the image viewer\'s focus to fresh images as they are generated.'),
            sg.Checkbox('Visualize Substeps (Slow)', key='_visualize_sub', default=False, tooltip='Whether or not to set the image viewer\'s focus to fresh images as they are generated.'),
        ],
        [sg.VPush()],
        ]

    layout_imageviewer = [
        # [sg.Text('Results', font='Any 13')],
        [
            ThumbnailImage('hist0'), 
            ThumbnailImage('hist1'), 
            ThumbnailImage('hist2'), 
            ThumbnailImage('hist3'), 
            ThumbnailImage('hist4'), 
            ThumbnailImage('hist5'), 
            ThumbnailImage('hist6'), 
            sg.Frame(" ", 
                [[sg.Text('', size=(4,1), justification='l', key='ThumbOverflowText')]], 
                expand_x=True, expand_y=True, p=1, background_color=sg.theme_button_color()[1], key='ThumbOverflowText_f', s=(68, 68) , vertical_alignment='center', element_justification='center'),
            ],
        [sg.HSeparator()],
        [sg.Image(size=(512,512), key='Image', enable_events=True, background_color=sg.theme_button_color()[1] )],
        [sg.Text('Sample:', justification='l', expand_x=True, size=(60,4), key='SampleInfo')],
        [
            sg.Button('Clear All', size=(8,1), key='-CLEARALL-', tooltip='Clears sample viewer history.  If samples are not saved to disk, this will permanently erase them.'), 
            sg.Button('Clear (Del)', size=(8,1), key='-CLEAR-', tooltip='Clears currently viewed sample.'),  
            sg.Push(), 
            sg.Checkbox('Embed params', key='-SAVE-Embed-', tooltip='Whether to embed parameters in the metadata of saved images. Required for reloading saved images in this viewer.', default=True),
            sg.Button('Save', size=(12,1), key='-SAVE-', tooltip='Save currently viewed sample.'), 
            sg.Button('Save-All', size=(12,1), key='-SAVEALL-', tooltip='Save all samples in history, even those overflowing the thumbnail history.')
        ],
        [
            sg.Button('Set Params', size=(8,1), key='-RESET-PARAMS-', tooltip='Set the generation parameters (on the left) to whatever generated the image shown.'), 
            sg.Checkbox('Single-shot?', key='-RESET-PARAMS-SingleShot-', tooltip='Check this box to ignore prior sample num/iteration num in favor of 1.', default=True), sg.Push()
        ],
        [sg.Push(), sg.FileBrowse(k='FileOpen', target='FileOpen', change_submits=True)],
    ]
    layout_canvaseditor = [
        [],
    ]
    layout = [
        [sg.Frame('Generation Parameters', layout=layout_settings, size=(800, 10), expand_y=True, vertical_alignment='top', p=0), 
            sg.VSeparator(), 
            sg.Frame('Sample Browser', layout=layout_imageviewer, p=0),
            sg.VSeparator(), 
            sg.Frame('Canvas Editor', layout=layout_canvaseditor, p=0)],
        [sg.StatusBar(f'Welcome to {programName}', expand_x=True, key='StatusBar')]
    ]

    window = sg.Window(programName, layout, finalize=True)

    for cb in l_to_block_focus:
        cb.block_focus(True)

    return window


sbar_colors = {'l': sg.theme_text_color(), 'w': '#a6943c', 'e': '#b03e3e',}

sem_generate = threading.Semaphore(1)

curr_sample_i = 0

def ui_thread(window:sg.Window):
    global opt
    global curr_sample_i

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

    def LogStatusBar(msg:str, level:str='i'):
        window['StatusBar'].update(value=msg, text_color=sbar_colors[level])

    def SetSampleAndInfo(index):
        global curr_sample_i
        if index < len(datalist) and index >= 0:
            _img, _options, _samplenum = datalist[index]
            window['SampleInfo'].update(
                f"{_options['process']} Sampler: {_options['sampler']} seed {_options['seed']}:{_options['seed_offset']}, sample {_samplenum}\n\
{_options['ddim_steps']} substeps, g_scale: {_options['scale']}\n\
\"{_options['prompt']}\"")
            window['Image'].update(data=ImageTk.PhotoImage(_img))
            curr_sample_i = index
        else:
            window['SampleInfo'].update('Sample: None')
            window['Image'].update(data=ImageTk.PhotoImage(blankImg), size=(512,512))
        
        window['ThumbOverflowText_f'].update(value="S")
        for i, imgKey in enumerate(imgKeys):
            if i == curr_sample_i:
                window[imgKey+'_f'].update(value="S")
                window['ThumbOverflowText_f'].update(value=" ")
            else:
                window[imgKey+'_f'].update(value=" ")


    def SaveImage(index):
        _img, _options, _i = datalist[index]

        if '_saved' in _options:
            return
        _options['_saved'] = True
        prompthash = hl.sha256(bytes(_options['prompt'], 'utf-8')).hexdigest()
        _path = os.path.join(_options['outdir'], "interactive", prompthash[:16])
        os.makedirs(_path, exist_ok=True)
        _file = os.path.join(_path, f"{_options['seed']:08}-{_options['seed_offset']:02}-{_i}.png")

        metadata = PngInfo()

        if values['-SAVE-Embed-']:
            metadata.add_text("sdParams", json.dumps(_options))
            metadata.add_text("sdSubsample", str(_i))

        metadata.add_text('sd_interactive', 'Made with Stable Diffusion Interactive.  See https://github.com/spychiatrist/stable-diffusion-improvements')

        _img.save(_file, pnginfo=metadata)

    def LoadImage(path):
        _img = Image.open(path)
        if 'sdParams' not in _img.text:
            LogStatusBar('Image lacks parameters metadata.  Select \'Embed params\' when saving outputs from this program.', 'e')
            return
        _options = json.loads(_img.text['sdParams'])
        _i = int(_img.text['sdSubsample'])
        _metadata = (_img, _options, _i)
        window.write_event_value("-IMAGE-", _metadata)

    def UpdateThumbnails():
        for i, k in enumerate(imgKeys):
            data = blankImg.resize((64,64), resample=Image.Resampling.BICUBIC)
            if i < len(datalist):
                data=datalist[i][0].resize((64,64), resample=Image.Resampling.BICUBIC)
            window[k].update(data=ImageTk.PhotoImage(data))

        nThumbs = len(imgKeys)
        nData = len(datalist)
        if nData <= nThumbs:
            window['ThumbOverflowText'].update(value='')
        else:
            window['ThumbOverflowText'].update(value=f'+{nData - nThumbs}')

    def SetThreadActionsDisabled(disabled:bool=False):
        window['Generate'].update(disabled=disabled)
        window['SamplerCombo'].update(disabled=disabled)
        window.force_focus()

    def GenerateYield():
        global curr_sample_i
        global g_init_img

        if values['tti']: #text to image params:
            opt.process = 'tti'
        else:
            opt.process = 'iti'
            g_init_img = datalist[curr_sample_i][0]


        opt.prompt =        values['prompt']
        opt.skip_grid =       values['skip_grid']
        opt.skip_save =       values['skip_save']
        opt._visualize_sub =  values['_visualize_sub']
        opt.n_iter =        int(values['n_iter'])
        opt.n_samples =     int(values['n_samples'])
        opt.seed_offset =   int(values['seed_offset'])

        l_scale =           ParseNumericParam( (values['scale']), float)
        l_strength =        ParseNumericParam( (values['strength']), float)
        l_ddim_eta =        ParseNumericParam( (values['ddim_eta']), float)
        l_ddim_steps =      ParseNumericParam( (values['ddim_steps']), int)
        l_seed =            ParseNumericParam( (values['seed']), int)
        
        perm_iterator = product(l_scale, l_strength, l_ddim_eta, l_ddim_steps, l_seed)

        for opt_it in perm_iterator:
            opt.scale, opt.strength, opt.ddim_eta, opt.ddim_steps, opt.seed = opt_it
            sem_generate.release()      
            yield


    def ParseNumericParam(input:str, castfn): 
        s = input.split(':')
        if len(s) == 2:
            s1 = s[1].split('(')
            if len(s1) == 2:
                s1[1] = s1[1].strip(')')
                base = castfn(s[0])
                iters = int(s1[0])
                offset = castfn(s1[1])
                return [base + i*offset for i in range(iters)]
        return [castfn(s[0])]
    window.bind('<Up>', '-U-ARROW-')
    window.bind('<Down>', '-D-ARROW-')
    window.bind('<Right>', '-R-ARROW-')
    window.bind('<Left>', '-L-ARROW-')
    window.bind('<Escape>', '-ESC-')
    window.bind('<Return>', 'Generate')
    window.bind('<Delete>', '-CLEAR-')
    window.bind('<Control-s>', '-SAVE-')
    window.bind('<Control-S>', '-SAVEALL-')

    UpdateThumbnails()

    SetSampleAndInfo(0)

    generator_params_yield = None

    while True:

        event, values = window.read()

        if event == 'Generate': # generate button event
            generator_params_yield = GenerateYield()
            SetThreadActionsDisabled(True)
            next(generator_params_yield)


        elif event == '-READY-':
            if generator_params_yield != None:
                try:
                    next(generator_params_yield)
                except StopIteration:
                    generator_params_yield = None
                    SetThreadActionsDisabled(False)
            else:
                SetThreadActionsDisabled(False)  

        elif event == 'SamplerCombo':
            opt.sampler = values[event]
            opt.change_model = True

            SetThreadActionsDisabled(True)
            sem_generate.release()



        elif event == '-IMAGE-': # new image from backend event
            datalist.insert(0, values[event])
            UpdateThumbnails()
            if values['_auto_focus']: 
                SetSampleAndInfo(0)
            else:
                SetSampleAndInfo(curr_sample_i + 1)

        elif event in imgKeys: # clicked a thumbnail
            _i = imgKeys.index(event)
            window.force_focus()
            SetSampleAndInfo(_i)

        elif event == 'Image':
            window.force_focus()



        elif event == '-ITER-':
            imgs, _i = values[event]
            if values['_visualize_sub'] and imgs != None: 
                #for img in imgs:
                window['Image'].update(data=ImageTk.PhotoImage(imgs[0]))

            window['GenerateProgress'].update(current_count=_i, max=opt.ddim_steps-1)
            
        elif event == '-SAVE-':
            if len(datalist) > 0:
                SaveImage(curr_sample_i)
        
        elif event == 'FileOpen':
            LoadImage(values['FileOpen'])

        elif event == '-CLEARALL-':
            datalist.clear()
            UpdateThumbnails()
            SetSampleAndInfo(0)

        elif event == '-CLEAR-':
            if curr_sample_i < len(datalist):
                del datalist[curr_sample_i]
                UpdateThumbnails()
                i_attempt = min(len(datalist) - 1, curr_sample_i)
                SetSampleAndInfo(i_attempt)
                


        elif event == '-RESET-PARAMS-':

            if len(datalist) > 0:
                _, _opt, _samplenum = datalist[curr_sample_i]

                window['prompt'      ].update(value=str(_opt['prompt'      ]))
                window['seed'        ].update(value=str(_opt['seed'        ]))
                window['seed_offset' ].update(value=str(_opt['seed_offset' ] + (_samplenum if values['-RESET-PARAMS-SingleShot-'] else 0)))
                window['ddim_steps'  ].update(value=str(_opt['ddim_steps'  ]))
                window['n_iter'      ].update(value=str(1 if values['-RESET-PARAMS-SingleShot-'] else _opt['n_iter'      ]))
                window['n_samples'   ].update(value=str(1 if values['-RESET-PARAMS-SingleShot-'] else _opt['n_samples'   ]))
                window['ddim_eta'    ].update(value=str(_opt['ddim_eta'    ]))
                window['scale'       ].update(value=str(_opt['scale'       ]))

                if values['SamplerCombo'] != _opt['sampler']:
                    window['SamplerCombo'].update( value=_opt['sampler'])
                    window.write_event_value('SamplerCombo', _opt['sampler'])

        elif event == '-SAVEALL-':
            for i in reversed(range(0, len(datalist))):
                SaveImage(i) #save in reverse order of history to get newest versions of overwrites last.

        # events that can happen only with global focus (i.e. focus is not a textbox)
        elif window.find_element_with_focus() == None:
            
            def opts_eq(d1, d2):
                _, _o1, _sn1 = d1
                _, _o2, _sn2 = d2
                def _cmp_key(key):
                    return _o1[key] == _o2[key]
                if values['g_seed'       ] and not  _cmp_key('seed'                         ): return False
                if values['g_seed_offset'] and not (_cmp_key('seed_offset') and _sn1 == _sn2): return False
                if values['g_ddim_steps' ] and not  _cmp_key('ddim_steps'                   ): return False
                if values['g_scale'      ] and not  _cmp_key('scale'                        ): return False
                if values['g_prompt'     ] and not  _cmp_key('prompt'                       ): return False
                return True
                    

            if event == '-D-ARROW-':
                #Search forward for next previous sample with matching settings
                if curr_sample_i > 0:
                    curr_data = datalist[curr_sample_i]
                    for i, data in enumerate(datalist[curr_sample_i+1:]):
                        if opts_eq(curr_data, data):
                            SetSampleAndInfo(curr_sample_i + 1 + i)
                            break

            if event == '-U-ARROW-':
                if curr_sample_i > 0:
                    curr_data = datalist[curr_sample_i]
                    for i, data in enumerate(datalist[curr_sample_i-1::-1]):
                        if opts_eq(curr_data, data):
                            SetSampleAndInfo(curr_sample_i - 1 - i)
                            break

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

def backend_loop():
    global window

    from ldm.simplet2i import T2I
    from pytorch_lightning import seed_everything

    t2i = T2I(
        width=opt.W,
        height=opt.H,
        sampler_name=opt.sampler,
        weights=opt.ckpt,
        full_precision=(opt.precision == "full"),
        config=opt.config,
        # grid  = opt.grid,
        # this is solely for recreating the prompt
        latent_diffusion_weights=opt.laion400m,
        embedding_path=opt.embedding_path,
        device=opt.device,
    )

    t2i.load_model()
    
    opt.change_model = False

    while True:
        #await mutex to proceed
        window.write_event_value("-READY-", None)
        sem_generate.acquire()

        if opt.change_model:
            t2i.sampler_name = opt.sampler
            t2i._set_sampler()
            opt.change_model = False
            continue

        os.makedirs(opt.outdir, exist_ok=True)
        outpath = opt.outdir

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)



        def substep_callback( x0,  i ):
            imgs = None
            if opt._visualize_sub:
                imgs = t2i._samples_to_images(x0)
            window.write_event_value('-ITER-', (imgs, i))

        def iter_callback(img:Image, seed, i):
            opt.seed=seed
            _metadata = (img, opt.__dict__.copy(), i)
            window.write_event_value("-IMAGE-", _metadata)
            
            
        seed_everything(opt.seed)
        t2i.prompt2image(prompt=opt.prompt,
            iterations=opt.n_iter,
            batch_size=opt.n_samples,
            steps=opt.ddim_steps,
            seed=opt.seed,
            cfg_scale=opt.scale,
            ddim_eta=opt.ddim_eta,
            image_callback=iter_callback,
            step_callback=substep_callback,
            width=opt.W,
            height=opt.H,
            sampler_name=opt.sampler,
            strength=opt.strength,
            init_img=g_init_img if opt.process == 'iti' else None
        )


    
def parse_args():
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
        "--debug_ui",
        action='store_true',
        help="only load UI, no backend",
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
        "--sampler",
        type=str,
        help="which sampler to use",
        default="k_lms"
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
        "--strength",
        type=float,
        default=0.7,
        help="img2img strength",
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
    parser.add_argument(
        '--device',
        '-d',
        type=str,
        default='cuda',
        help='Device to run Stable Diffusion on. Defaults to cuda `torch.cuda.current_device()` if avalible',
    )
    parser.add_argument(
        '--embedding_path',
        type=str,
        help='Path to a pre-trained embedding manager checkpoint - can only be set on command line',
    )
    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    return opt



if __name__ == "__main__":
    main()
