from pathlib import Path
import sys
from PIL import Image
from utils_ootd import get_mask_location
import torch

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from ootd.inference_ootd_hd import OOTDiffusionHD
from ootd.inference_ootd_dc import OOTDiffusionDC


import argparse
parser = argparse.ArgumentParser(description='run ootd')
parser.add_argument('--gpu_id', '-g', type=int, default=0, required=False)
parser.add_argument('--model_path', type=str, default="", required=True)
parser.add_argument('--cloth_path', type=str, default="", required=True)
parser.add_argument('--model_type', type=str, default="hd", required=False)
parser.add_argument('--category', '-c', type=int, default=0, required=False)
parser.add_argument('--scale', type=float, default=2.0, required=False)
parser.add_argument('--step', type=int, default=20, required=False)
parser.add_argument('--sample', type=int, default=4, required=False)
parser.add_argument('--seed', type=int, default=-1, required=False)
args = parser.parse_args()


openpose_model = OpenPose(args.gpu_id)
parsing_model = Parsing(args.gpu_id)


category_dict = ['upperbody', 'lowerbody', 'dress']
category_dict_utils = ['upper_body', 'lower_body', 'dresses']

model_type = args.model_type # "hd" or "dc"
category = args.category # 0:upperbody; 1:lowerbody; 2:dress
cloth_path = args.cloth_path
model_path = args.model_path

image_scale = args.scale
n_steps = args.step
n_samples = args.sample
seed = args.seed

if model_type == "hd":
    model = OOTDiffusionHD(args.gpu_id)
elif model_type == "dc":
    model = OOTDiffusionDC(args.gpu_id)
else:
    raise ValueError("model_type must be \'hd\' or \'dc\'!")

if "AMD" in torch.cuda.get_device_name() or "Radeon" in torch.cuda.get_device_name():
    try:
        from flash_attn import flash_attn_func

        sdpa = torch.nn.functional.scaled_dot_product_attention

        def sdpa_hijack(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
            if query.shape[3] <= 128 and attn_mask is None:
                hidden_states = flash_attn_func(
                    q=query.transpose(1, 2),
                    k=key.transpose(1, 2),
                    v=value.transpose(1, 2),
                    dropout_p=dropout_p,
                    causal=is_causal,
                    softmax_scale=scale,
                ).transpose(1, 2)
            else:
                hidden_states = sdpa(
                    query=query,
                    key=key,
                    value=value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                )
            return hidden_states

        torch.nn.functional.scaled_dot_product_attention = sdpa_hijack
        print("# # #\nHijacked SDPA with ROCm Flash Attention\n# # #")
    except ImportError as e:
        print(f"# # #\nCould not load Flash Attention for hijack:\n{e}\n# # #")
else:
    print(f"# # #\nCould not detect AMD GPU from:\n{torch.cuda.get_device_name()}\n# # #")

if __name__ == '__main__':

    if model_type == 'hd' and category != 0:
        raise ValueError("model_type \'hd\' requires category == 0 (upperbody)!")

    temp_height = 768
    temp_width = int(temp_height * 4 / 3)
    print(temp_height, temp_width)
    # cloth_img = Image.open(cloth_path).resize((768, 1024))
    # model_img = Image.open(model_path).resize((768, 1024))
    cloth_img = Image.open(cloth_path).resize((temp_height, temp_width))
    model_img = Image.open(model_path).resize((temp_height, temp_width))
    # for i in range(0, 1000):
    keypoints = openpose_model(model_img.resize((384, 512)))
    model_parse, _ = parsing_model(model_img.resize((384, 512)))
    print(keypoints, model_parse)

    mask, mask_gray = get_mask_location(model_type, category_dict_utils[category], model_parse, keypoints)
    # mask = mask.resize((768, 1024), Image.NEAREST)
    # mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
    mask = mask.resize((temp_height, temp_width), Image.NEAREST)
    mask_gray = mask_gray.resize((temp_height, temp_width), Image.NEAREST)
    
    masked_vton_img = Image.composite(mask_gray, model_img, mask)
    masked_vton_img.save('./images_output/mask.jpg')

    # exit(0)
    images = model(
        model_type=model_type,
        category=category_dict[category],
        image_garm=cloth_img,
        image_vton=masked_vton_img,
        mask=mask,
        image_ori=model_img,
        num_samples=n_samples,
        num_steps=n_steps,
        image_scale=image_scale,
        seed=seed,
    )

    image_idx = 0
    for image in images:
        image.save('./images_output/out_' + model_type + '_' + str(image_idx) + '.png')
        image_idx += 1
