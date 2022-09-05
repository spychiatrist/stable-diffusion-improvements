import argparse, os, sys, glob
import queue
import json
#from ctypes import alignment
#from ast import arg
import hashlib as hl
#import numpy as np
from PIL import Image,ImageTk
from PIL.PngImagePlugin import PngInfo
import PySimpleGUI as sg
import tkinter as tk
import threading
#from torchvision.utils import make_grid
#import time
#from torch import autocast
#from contextlib import contextmanager, nullcontext
from itertools import product



class BackendInput:

    def __init__(
        self,
    ):    
        self._visualize_sub = False
        self.cancel = False  
        self.change_model = False  
        self.commandQueue = queue.Queue()

class BackendCommand:
    def __init__(
        self, 
    ):
        self.sampler = None
        self.prompt = None
        self.seed = None
        self.ddim_steps = None
        self.scale = None
        self.n_iter = None
        self.n_samples = None
        self.strength = None
        self.ddim_eta = None
        self.process = None
        self.width = None
        self.height = None
        self.img = None
        self.prevCmdDict = None
        self.prevCmdSamplenum = None
    

opt = None
g_backend_input = BackendInput()

window:sg.Window = None

g_backend_thread = None
g_settings = {'save_num':0, }

def SaveSettings():
    global g_settings, opt
    os.makedirs(opt.settings_path, exist_ok=True)
    with open( os.path.join(opt.settings_path, "settings.json"), "w") as f:
        f.write(json.dumps(g_settings))

def LoadSettings():
    global g_settings, opt
    sfp = os.path.join(opt.settings_path, "settings.json")
    if os.path.exists(sfp):
        with open(sfp, "r") as f:
            g_settings = json.load(f)
    else:
        SaveSettings()

def main():
    global opt, g_settings, g_backend_thread
    global window
    opt = parse_args()

    LoadSettings()
        
        

    window = make_ui()

    
    sem_generate.acquire()
    if not opt.debug_ui:
        g_backend_thread = threading.Thread(target=backend_loop).start()
    ui_thread(window)


def make_ui():
    global opt
    print("Starting UI...")

    programName = 'Stable Diffusion Interactive'

    sampler_choices = ['k_lms', 'k_euler_a', 'ddim', 'k_dpm_2_a', 'k_dpm_2', 'k_euler', 'k_heun', 'plms']
    def TextLabel(text, base_key, vis=True): return sg.Text(text+':', justification='r', size=(18,1), visible=vis, k='label_' + base_key)


    w_input=80
    def InputRow(text, base_key, default_text, tooltip, groupable=False, group_default=False, disabled=False, vis=True, input_size_v=1):
        ret = [
            TextLabel(text, base_key, vis=vis),
        ]
        if input_size_v==1:
            ret.append(sg.Input(key=base_key, default_text=default_text, tooltip=tooltip, visible=vis, disabled=disabled, disabled_readonly_background_color='#888888', size=(w_input,1)))
        else:
            dp = sg.DEFAULT_ELEMENT_PADDING
            ret.append(sg.Multiline(key=base_key, default_text=default_text, tooltip=tooltip, no_scrollbar=True, visible=vis, size=(w_input - 4,input_size_v), pad=((dp[0], 0),(0, 0))))
            ret.append(sg.Button('X', size=(2,1), k='x_'+base_key, pad=((0, 0),(dp[1], dp[1]))))

        ret.append(sg.Push())
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
        [TextLabel('Sampler', 'SamplerCombo'), sg.Combo(sampler_choices, default_value=opt.sampler if opt.sampler in sampler_choices else sampler_choices[0], enable_events=True, k='SamplerCombo' ), 
            sg.Radio('txt->img', 'SamplerProcess', k='tti', enable_events=True, default=True), sg.Radio('img->img', 'SamplerProcess', enable_events=True, k='iti'), sg.Radio('gfpgan', 'SamplerProcess', enable_events=True, k='gfp'), sg.Push(), GroupCheckbox('sampler', default=False)],

        InputRow('Prompt'               , 'prompt'      , opt.prompt      , 'Description of the image you would like to see.' , input_size_v=3                                                                                          , groupable=True, group_default=True),
        InputRow('Seed'                 , 'seed'        , opt.seed        , 'Seed for first image generation.'                                                                                                                          , groupable=True, group_default=True),
        InputRow('Sampler Substeps'     , 'ddim_steps'  , opt.ddim_steps  , 'Number of substeps per batch.'                                                                                                                             , groupable=True                    ),
        InputRow('Guidance Scale'       , 'scale'       , opt.scale       , 'Unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))'                                                                , groupable=True                    ),
        InputRow('Iterations'           , 'n_iter'      , opt.n_iter      , 'Number of batches per generation.'                                                                                                                                                             ),
        InputRow('Batch Size (Samples)' , 'n_samples'   , opt.n_samples   , 'Number of samples per batch.'                                                                                                                                                                  ),
        InputRow('Strength'             , 'strength'    , opt.strength    , 'Image-to-Image strength (0.0, 1.0)'                                                                                                                        , disabled=True                     ),
        InputRow('Sampler Eta'          , 'ddim_eta'    , opt.ddim_eta    , 'Deprecated: ddim eta'                                                                                                                                      , vis=False                         ),

        [
            sg.Checkbox('Skip Sample Save',key='skip_save', default=opt.skip_save, tooltip='If checked, program will not save each sample automatically.', visible=False),
            sg.Checkbox('Skip Grid Save',key='skip_grid', default=opt.skip_grid, tooltip='If checked, program will not save grid collages of each batch.', visible=False)
        ],
        [sg.Button('Generate', size=(20,2), disabled=True), sg.Button('Cancel', size=(12,2), disabled=False), sg.ProgressBar(k='GenerateProgress', max_value=100, s=(30, 20), orientation='h')],
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
        [sg.Text('Sample:', justification='l', expand_x=True, size=(60,4), key='SampleInfo', font='Consolas')],
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
        [sg.Push(), sg.FilesBrowse(button_text='Open', k='FileOpen', target='FileOpen', change_submits=True)],
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
            if _options is not None:
                window['SampleInfo'].update(
                    f"{_options['process']+'-'+_options['sampler']:14} | seed: {_options['seed']:010}:{_samplenum:03}\n"+
                    f"substeps: {_options['ddim_steps']:3}  | scale: {_options['scale']}\n"+
                    f"> \"{_options['prompt']}\"")
            else:
                window['SampleInfo'].update("Sample was not generated by this program.  No parameters to show.")
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
        global g_settings
        _img, _options, _i = datalist[index]

        if _options is None:
            return
        
        if '_saved' in _options and _options['_saved']:
            return
        _options['_saved'] = True
        # prompthash = hl.sha256(bytes(_options['prompt'], 'utf-8')).hexdigest()
        _path = os.path.join(opt.outdir, "interactive")
        os.makedirs(_path, exist_ok=True)
        _file = os.path.join(_path, f"{g_settings['save_num']:010}.png")

        g_settings['save_num'] += 1
        SaveSettings()

        metadata = PngInfo()

        if values['-SAVE-Embed-']:
            metadata.add_text("sdParams", json.dumps(_options))
            metadata.add_text("sdSubsample", str(_i))

        metadata.add_text('sd_interactive', 'Made with Stable Diffusion Interactive.  See https://github.com/spychiatrist/stable-diffusion-improvements')

        _img.save(_file, pnginfo=metadata)

    def LoadImage(path):
        _img = Image.open(path)
        _options = None
        _i = None
        if 'sdParams' in _img.text:
            _options = json.loads(_img.text['sdParams'])
            _i = int(_img.text['sdSubsample'])
        else:
            _img = _img.convert('RGB')#.resize((512,576), Image.Resampling.BOX)
            w = _img.size[0]
            h = _img.size[1]
            aspect = w/h
            s_dim = 0
            if aspect > 1: # height is smaller than width
                s_dim = h
            else: 
                s_dim = w

            min_dim = 512
            #ensure image is >512 in each dimension
            scale = 1
            if s_dim < min_dim:
                scale = min_dim/s_dim
                _img = _img.resize((int(round(w*scale)), int(round(h*scale))), Image.Resampling.LANCZOS)

            #pad the image into multiples of 64
            w_q = _img.size[0] - (_img.size[0] % 64) 
            h_q = _img.size[1] - (_img.size[1] % 64) 
            #_img = _img.crop((0,0,w_q, h_q))
            ni = Image.new('RGB', (w_q, h_q), (128,128,128))
            ni.paste(_img)
            _img = ni
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
        global g_backend_input
        
        

        l_scale =           ParseNumericParam( (values['scale']), float)
        l_strength =        ParseNumericParam( (values['strength']), float)
        l_ddim_eta =        ParseNumericParam( (values['ddim_eta']), float)
        l_ddim_steps =      ParseNumericParam( (values['ddim_steps']), int)
        l_seed =            ParseNumericParam( (values['seed']), int)
        
        perm_iterator = product(l_scale, l_strength, l_ddim_eta, l_ddim_steps, l_seed)

        for opt_it in perm_iterator:

            cmd = BackendCommand()

            if values['tti']: #text to image params:
                cmd.process = 'tti'
                cmd.width, cmd.height = (512, 512)
            elif values['gfp']:
                cmd.process = 'gfpgan'
                cmd.img = datalist[curr_sample_i][0] 
                cmd.prevCmdDict = datalist[curr_sample_i][1].copy()
                cmd.prevCmdSamplenum = datalist[curr_sample_i][2]
                cmd.prevCmdDict['process'] = cmd.process
                cmd.prevCmdDict['_saved'] = False

            elif values['iti']:
                cmd.process = 'iti'
                cmd.img = datalist[curr_sample_i][0]
                cmd.width, cmd.height = cmd.img.size

            g_backend_input._visualize_sub =  values['_visualize_sub']

            cmd.sampler =        values['SamplerCombo']
            cmd.prompt =        values['prompt']
            cmd.n_iter =        int(values['n_iter'])
            cmd.n_samples =     int(values['n_samples'])

            cmd.scale, cmd.strength, cmd.ddim_eta, cmd.ddim_steps, cmd.seed = opt_it

            g_backend_input.commandQueue.put(cmd)

        sem_generate.release()     

    def Greyout(base_keys):
        window['prompt'].update(disabled=False)
        window['seed'].update(disabled=False)
        window['ddim_steps'].update(disabled=False)
        window['scale'].update(disabled=False)
        window['n_iter'].update(disabled=False)
        window['n_samples'].update(disabled=False)
        window['strength'].update(disabled=False)
        window['ddim_eta'].update(disabled=False)
        for key in base_keys: window[key].update(disabled=True)


    def SetProcess(val:str):
        if val == 'tti':
            Greyout(['strength'])
        if val == 'iti':
            Greyout([])
        if val == 'gfp':
            Greyout(['prompt', 'seed', 'ddim_steps', 'scale', 'n_iter', 'n_samples'])

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
    window.bind('<Control-c>', '-COPY-')
    window.bind('<Control-s>', '-SAVE-')
    window.bind('<Control-S>', '-SAVEALL-')

    UpdateThumbnails()

    SetSampleAndInfo(0)

    while True:

        event, values = window.read()

        if event == 'Generate': # generate button event
            g_backend_input.cancel = False
            GenerateYield()
            SetThreadActionsDisabled(True)
            
        elif event == 'Cancel':
            g_backend_input.cancel = True


        elif event == '-READY-':
            SetThreadActionsDisabled(False)  

        # elif event == 'SamplerCombo':
        #     g_backend_input.sampler = values[event]
        #     g_backend_input.change_model = True

        #     SetThreadActionsDisabled(True)
        #     sem_generate.release()

        elif event in ['tti', 'iti', 'gfp']:
            SetProcess(event)


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
            imgs, _i, _max_it = values[event]
            if values['_visualize_sub'] and imgs != None: 
                #for img in imgs:
                window['Image'].update(data=ImageTk.PhotoImage(imgs[0]))

            window['GenerateProgress'].update(current_count=_i, max=_max_it-1)
            
        elif event == '-COPY-':
            window.TKroot.clipboard_clear()
            window.TKroot.clipboard_append("Hello there.")


        elif event == '-SAVE-':
            if len(datalist) > 0:
                SaveImage(curr_sample_i)
        
        elif event == 'FileOpen':
            s:str = values['FileOpen']
            l_files = s.split(';')
            for f in l_files:
                LoadImage(f)

        elif event == '-CLEARALL-':
            datalist.clear()
            UpdateThumbnails()
            SetSampleAndInfo(0)



        elif event == 'x_prompt': 
            window['prompt'].update('')
                


        elif event == '-RESET-PARAMS-':

            if len(datalist) > 0:
                _, _opt, _samplenum = datalist[curr_sample_i]

                if _opt is not None:

                    window['prompt'      ].update(value=str(_opt['prompt'      ]))
                    window['seed'        ].update(value=str(_opt['seed'        ]))
                    # window['seed_offset' ].update(value=str(_opt['seed_offset' ] + (_samplenum if values['-RESET-PARAMS-SingleShot-'] else 0)))
                    window['ddim_steps'  ].update(value=str(_opt['ddim_steps'  ]))
                    window['n_iter'      ].update(value=str(1 if values['-RESET-PARAMS-SingleShot-'] else _opt['n_iter'      ]))
                    window['n_samples'   ].update(value=str(1 if values['-RESET-PARAMS-SingleShot-'] else _opt['n_samples'   ]))
                    window['ddim_eta'    ].update(value=str(_opt['ddim_eta'    ]))
                    window['scale'       ].update(value=str(_opt['scale'       ]))
                    window['strength'    ].update(value=str(_opt['strength'    ]))

                    if values['SamplerCombo'] != _opt['sampler']:
                        window['SamplerCombo'].update( value=_opt['sampler'])
                        window.write_event_value('SamplerCombo', _opt['sampler'])
                else:
                    LogStatusBar("You can't reset parameters from an image that wasn't created with parameters.", 'w')

        elif event == '-SAVEALL-':
            for i in reversed(range(0, len(datalist))):
                SaveImage(i) #save in reverse order of history to get newest versions of overwrites last.

        # events that can happen only with global focus (i.e. focus is not a textbox)
        elif window.find_element_with_focus() == None:
            
            def opts_eq(d1, d2):
                if d1 is None or d2 is None:
                    return False
                _, _o1, _sn1 = d1
                _, _o2, _sn2 = d2
                def _cmp_key(key):
                    return _o1[key] == _o2[key]
                if values['g_seed'       ] and not (_cmp_key('seed')        and _sn1 == _sn2): return False
                # if values['g_seed_offset'] and not (_cmp_key('seed_offset') ): return False
                if values['g_ddim_steps' ] and not  _cmp_key('ddim_steps'                   ): return False
                if values['g_scale'      ] and not  _cmp_key('scale'                        ): return False
                if values['g_sampler'    ] and not  _cmp_key('sampler'                      ): return False
                if values['g_prompt'     ] and not  _cmp_key('prompt'                       ): return False
                return True
                    
            if event == '-CLEAR-':
                if curr_sample_i < len(datalist):
                    del datalist[curr_sample_i]
                    UpdateThumbnails()
                    i_attempt = min(len(datalist) - 1, curr_sample_i)
                    SetSampleAndInfo(i_attempt)

            elif event == '-D-ARROW-':
                #Search forward for next previous sample with matching settings
                if len(datalist) > 0:
                    curr_data = datalist[curr_sample_i]
                    for i, data in enumerate(datalist[curr_sample_i+1:]):
                        if opts_eq(curr_data, data):
                            SetSampleAndInfo(curr_sample_i + 1 + i)
                            break

            elif event == '-U-ARROW-':
                if curr_sample_i > 0:
                    curr_data = datalist[curr_sample_i]
                    for i, data in enumerate(datalist[curr_sample_i-1::-1]):
                        if opts_eq(curr_data, data):
                            SetSampleAndInfo(curr_sample_i - 1 - i)
                            break

            elif event == '-R-ARROW-':
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


    
    cmd:BackendCommand = None

    while True:
        #await mutex to proceed

        try:
            cmd = g_backend_input.commandQueue.get(False)

        except queue.Empty as e:
            
            window.write_event_value("-READY-", None)
            sem_generate.acquire()
            continue

        # if opt.change_model:
        #     t2i.sampler_name = opt.sampler
        #     t2i._set_sampler()
        #     opt.change_model = False
        #     continue

        

        def substep_callback( x0,  i ):
            imgs = None
            if g_backend_input._visualize_sub:
                imgs = t2i._samples_to_images(x0)
            window.write_event_value('-ITER-', (imgs, i, cmd.ddim_steps))
            if g_backend_input.cancel:
                raise KeyboardInterrupt

        def iter_callback(img:Image, seed, i):
            cmd.seed=seed
            _tmpimg = cmd.img
            cmd.img = None #don't store the image in the copy TODO fix this hack
            _metadata = (img, cmd.__dict__.copy(), i)
            cmd.img = _tmpimg
            window.write_event_value("-IMAGE-", _metadata)
            
            
        seed_everything(cmd.seed)
        if cmd.process == 'gfpgan':

            
            try:
                from ldm.gfpgan.gfpgan_tools import _run_gfpgan

                img = _run_gfpgan(cmd.img, cmd.strength, 1)
                window.write_event_value("-IMAGE-", (img, cmd.prevCmdDict, cmd.prevCmdSamplenum))

            except Exception as e:
                print(
                    f'Error running RealESRGAN - Your image was not upscaled.\n{e}'
                )

                # from ldm.gfpgan.gfpgan_tools import (
                #     real_esrgan_upscale,
                # )
                # if len(upscale) < 2:
                #     upscale.append(0.75)
                # image = real_esrgan_upscale(
                #     image,
                #     upscale[1],
                #     int(upscale[0])
                # )
        else:
            t2i.prompt2image(prompt=cmd.prompt,
                iterations=cmd.n_iter,
                batch_size=cmd.n_samples,
                steps=cmd.ddim_steps,
                seed=cmd.seed,
                cfg_scale=cmd.scale,
                ddim_eta=cmd.ddim_eta,
                image_callback=iter_callback,
                step_callback=substep_callback,
                width=cmd.width,
                height=cmd.height,
                sampler_name=cmd.sampler,
                strength=cmd.strength,
                init_img=cmd.img if cmd.process == 'iti' else None
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
        "--settings_path",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="settings"
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
    # parser.add_argument(
    #     "--seed_offset",
    #     type=int,
    #     default=0,
    #     help="how many samples forward should the seed state emulate",
    # )
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

    opt.process = 'tti'

    return opt



if __name__ == "__main__":
    main()
