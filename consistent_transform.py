import json
from pathlib import Path

import PIL
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
from cldm.ddim_hacked import DDIMSampler
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights


preprocessor = None

model_name = 'control_v11p_sd15_lineart'
model = create_model(f'./models/{model_name}.yaml').cpu()
model.load_state_dict(load_state_dict('./models/v1-5-pruned.ckpt', location='cuda'), strict=False)
model.load_state_dict(load_state_dict(f'./models/{model_name}.pth', location='cuda'), strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def calc_flow(input_frame1, input_frame2):
    # calc optical flow
    raft_model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).cuda()
    raft_model = raft_model.eval()
    input_frame1_th = torch.from_numpy(input_frame1).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.
    input_frame2_th = torch.from_numpy(input_frame2).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.

    flows = raft_model(input_frame1_th, input_frame2_th)
    flow_fw = flows[-1][0].permute((1, 2, 0)).cpu().numpy()

    flows = raft_model(input_frame2_th, input_frame1_th)
    flow_bw = flows[-1][0].permute((1, 2, 0)).cpu().numpy()

    th = 0.1  # pixel
    flow_fw_warped = warp_image(flow_fw, flow_bw)
    warp_mask = (np.abs(flow_bw + flow_fw_warped) < th).all(2)

    return flow_fw, warp_mask

def warp_image(image, flow):
    h, w, _ = flow.shape

    # Create mesh grid
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + flow[:, :, 0]).astype(np.float32)
    map_y = (y + flow[:, :, 1]).astype(np.float32)

    # Warp image using flow field
    warped = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped

def process(det, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution,
            mask, prev_warped, ddim_steps, guess_mode, strength, scale, seed, eta, do_diffusion_sync):
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
        H, W, C = img.shape
        shape = (4, H // 8, W // 8)

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = 1.0 - torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {
            "c_concat": [control],
            "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)],
        }
        un_cond = {
            "c_concat": None if guess_mode else [control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

        if mask is not None:
            mask = cv2.resize(mask.astype('float32'), (W // 8, H // 8), interpolation=cv2.INTER_AREA) > ((W // 8) / input_image.shape[1])
            mask = torch.from_numpy(mask).float().cuda().unsqueeze(0).unsqueeze(0)
        if prev_warped is not None:
            prev_warped = resize_image(prev_warped, image_resolution)
            prev_warped = torch.from_numpy(prev_warped).permute((2, 0, 1)).float().cuda().unsqueeze(0) / 127.0 - 1.0
            prev_warped = model.get_first_stage_encoding(model.encode_first_stage(prev_warped))

        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     mask=mask,
                                                     x0=prev_warped,
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

    parser.add_argument("--input_frame1", type=str, required=True, help="Path to input frame 1 in the video")
    parser.add_argument("--input_frame2", type=str, required=True, help="Path to input frame 2 in the video")
    parser.add_argument("--output_dir", type=str, default='./outputs', help="Path to outputs dir")
    parser.add_argument("--do_diffusion_sync", default=False, action='store_true', help="Do Diffusion Synchronization on FG/BG")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for image foreground generation")
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

    frame1_basename = Path(args.input_frame1).stem
    frame2_basename = Path(args.input_frame2).stem
    output_dir = args.output_dir
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(exist_ok=True, parents=True)

    # write args to file
    with open(Path(output_dir) / (frame1_basename + '_' + frame2_basename + '_args.txt'), 'w') as f:
        f.write(json.dumps(vars(args), indent=4))

    # load input image + depth
    input_frame1 = imageio.imread(args.input_frame1)
    input_frame2 = imageio.imread(args.input_frame2)

    # run sampling process
    process_args = vars(args)
    del process_args['output_dir']
    del process_args['input_frame1']
    del process_args['input_frame2']

    # transform frame 1
    process_args['input_image'] = input_frame1
    process_args['mask'] = None
    process_args['prev_warped'] = None
    frame1_trans = process(**process_args)[-1]

    # calc flow, nd project generated frame1 to frame2
    with torch.no_grad():
        input_frame1 = resize_image(input_frame1, args.image_resolution)
        input_frame2 = resize_image(input_frame2, args.image_resolution)
        flow_bw, warp_mask = calc_flow(input_frame2, input_frame1)
    prev_image_warped = warp_image(frame1_trans, flow_bw)

    # transform frame 2
    process_args['input_image'] = input_frame2
    process_args['mask'] = warp_mask
    process_args['prev_warped'] = prev_image_warped
    frame2_trans = process(**process_args)[-1]

    # save inputs and outputs
    print("writing inputs and outputs to: ", output_dir)
    imageio.imwrite(Path(output_dir) / (frame1_basename + '.png'), input_frame1)
    imageio.imwrite(Path(output_dir) / (frame2_basename + '.png'), input_frame2)
    imageio.imwrite(Path(output_dir) / (frame1_basename + '_trans.png'), frame1_trans)
    imageio.imwrite(Path(output_dir) / (frame2_basename + '_trans.png'), frame2_trans)
    with open(Path(output_dir) / (frame1_basename + '_' + frame2_basename + '_prompt.txt'), 'w') as f:
        f.write(f"{args.prompt}\n")
