import json
from pathlib import Path
import imageio
from sklearn.mixture import GaussianMixture
import config
import cv2
import einops
import numpy as np
import torch
import torch.nn.functional as F
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.lineart import LineartDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_bg_fg_sync import DDIMSampler


preprocessor = None

model_name = 'control_v11p_sd15_lineart'
model = create_model(f'./models/{model_name}.yaml').cpu()
model.load_state_dict(load_state_dict('./models/v1-5-pruned.ckpt', location='cuda'), strict=False)
model.load_state_dict(load_state_dict(f'./models/{model_name}.pth', location='cuda'), strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(det, input_image, input_depth, prompt_fg, prompt_bg, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution,
            ddim_steps, guess_mode, strength, scale, seed, eta, do_diffusion_sync):
    global preprocessor
    if 'Lineart' in det:
        if not isinstance(preprocessor, LineartDetector):
            preprocessor = LineartDetector()

    with torch.no_grad():
        input_image = HWC3(input_image)

        if det == 'None':
            detected_map = input_image.copy()
        else:
            detected_map = preprocessor(resize_image(input_image, detect_resolution), coarse='Coarse' in det)
            detected_map = HWC3(detected_map)

        img = resize_image(input_image, image_resolution)
        depth = resize_image(input_depth, image_resolution, is_depth=True)
        H, W, C = img.shape
        shape = (4, H // 8, W // 8)

        # predict FG/BG mask
        gm = GaussianMixture(n_components=2, random_state=0).fit(depth.flatten()[..., None])
        sep_sigmas_middleground = (gm.means_[0] + np.sqrt(gm.covariances_[0]) + gm.means_[1] - np.sqrt(gm.covariances_[1])) / 2.
        if sep_sigmas_middleground > gm.means_[0] and sep_sigmas_middleground < gm.means_[1]:
            fg_bg_sep = sep_sigmas_middleground
        else:
            fg_bg_sep = (gm.means_[0] + gm.means_[1]) / 2.
        mask_fg = (depth < fg_bg_sep).astype('uint8')
        mask_fg = torch.from_numpy(mask_fg).float().cuda()
        mask_fg = mask_fg.unsqueeze(0).unsqueeze(0)
        mask_fg = F.max_pool2d(mask_fg, kernel_size=3, stride=1, padding=[1, 1])
        mask_fg_latent = F.max_pool2d(mask_fg, kernel_size=8, stride=8)

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = 1.0 - torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        control = control * mask_fg

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {
            "c_concat_fg": [control],
            "c_concat_bg": None,
            "c_crossattn_fg": [model.get_learned_conditioning([prompt_fg + ', ' + a_prompt] * num_samples)],
            "c_crossattn_bg": [model.get_learned_conditioning([prompt_bg + ', ' + a_prompt] * num_samples)],
            "c_crossattn": [model.get_learned_conditioning([prompt_fg + ', ' + a_prompt] * num_samples)],
        }
        un_cond = {
            "c_concat_fg": None if guess_mode else [control],
            "c_concat_bg": None,
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     mask=mask_fg_latent if do_diffusion_sync else None,
                                                     do_diffusion_synchronization=do_diffusion_sync,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [detected_map] + results


import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="ControlNet Gradio to CLI")

    parser.add_argument("--input_image", type=str, required=True, help="Path to input image")
    parser.add_argument("--input_depth", type=str, required=True, help="Path to input depth image")
    parser.add_argument("--output_dir", type=str, default='./outputs', help="Path to outputs dir")
    parser.add_argument("--do_diffusion_sync", default=False, action='store_true', help="Do Diffusion Synchronization on FG/BG")
    parser.add_argument("--prompt_fg", type=str, required=True, help="Prompt for image foreground generation")
    parser.add_argument("--prompt_bg", type=str, default="", help="Prompt for image background generation")
    parser.add_argument("--num_samples", type=int, default=1, choices=range(1, 13),
                        help="Number of images to generate (1-12)")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed (-1 for random, max 2147483647)")
    parser.add_argument("--det", type=str, choices=["Lineart", "Lineart_Coarse", "None"], default="Lineart",
                        help="Preprocessor selection")

    # Advanced options
    parser.add_argument("--image_resolution", type=int, default=512, choices=range(256, 769, 64),
                        help="Image resolution (256-768, step 64)")
    parser.add_argument("--strength", type=float, default=1.0, help="Control strength (0.0-2.0)")
    parser.add_argument("--guess_mode", action='store_true', help="Enable guess mode")
    parser.add_argument("--detect_resolution", type=int, default=512, choices=range(128, 1025),
                        help="Preprocessor resolution (128-1024)")
    parser.add_argument("--ddim_steps", type=int, default=20, choices=range(1, 101), help="DDIM Steps (1-100)")
    parser.add_argument("--scale", type=float, default=9.0, help="Guidance scale (0.1-30.0)")
    parser.add_argument("--eta", type=float, default=1.0, help="DDIM ETA (0.0-1.0)")
    parser.add_argument("--a_prompt", type=str, default="best quality", help="Additional prompt")
    parser.add_argument("--n_prompt", type=str, default="lowres, bad anatomy, bad hands, cropped, worst quality",
                        help="Negative prompt")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    basename = Path(args.input_image).stem
    output_dir = args.output_dir
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(exist_ok=True, parents=True)

    # write args to file
    with open(Path(output_dir) / (basename + '_args.txt'), 'w') as f:
        f.write(json.dumps(vars(args), indent=4))

    # load input image + depth
    args.input_image = imageio.imread(args.input_image)
    payload = np.load(args.input_depth)
    args.input_depth = payload[payload.files[0]]

    # run sampling process
    process_args = vars(args)
    del process_args['output_dir']
    out = process(**process_args)
    image_trans = out[-1]

    # save inputs and outputs
    print("writing inputs and outputs to: ", output_dir)
    imageio.imwrite(Path(output_dir) / (basename + '.png'), args.input_image)
    imageio.imwrite(Path(output_dir) / (basename + '_trans.png'), image_trans)
    with open(Path(output_dir) / (basename + '_prompts.txt'), 'w') as f:
        f.write(f"prompt_fg: {args.prompt_fg}\n")
        f.write(f"prompt_bg: {args.prompt_bg}\n")
