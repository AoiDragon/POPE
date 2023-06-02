import os
import json
from PIL import Image

import torch
import tqdm
import random

from seem.xdecoder.BaseModel import BaseModel
from seem.xdecoder import build_model
from seem.utils.distributed import init_distributed
from seem.utils.arguments import load_opt_from_config_files
from seem.utils.constants import COCO_PANOPTIC_CLASSES

from seem.tasks import *

conf_files = 'seem/configs/seem/seem_focall_lang.yaml'
opt = load_opt_from_config_files(conf_files)
opt = init_distributed(opt)

# META DATA
cur_model = 'None'
if 'focalt' in conf_files:
    pretrained_pth = os.path.join("seem/seem_focalt_v2.pt")
    if not os.path.exists(pretrained_pth):
        os.system("wget {}".format("https://projects4jw.blob.core.windows.net/x-decoder/release/seem_focalt_v2.pt"))
    cur_model = 'Focal-T'
elif 'focal' in conf_files:
    pretrained_pth = os.path.join("seem/seem_focall_v1.pt")
    if not os.path.exists(pretrained_pth):
        os.system("wget {}".format("https://projects4jw.blob.core.windows.net/x-decoder/release/seem_focall_v1.pt"))
    cur_model = 'Focal-L'

'''
build model
'''
model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)
audio = None
@torch.no_grad()
def inference(image, task, *args, **kwargs):
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        return interactive_infer_image(model, audio, image, task, *args, **kwargs)

def segment(img_path_file, seg_num):
    result_list = []
    img_list = json.loads(open(img_path_file, 'r').read())
    img_list = random.choices(img_list, k=seg_num)
    
    segment_result_file = open(f'segmentation/{img_path_file.split(".json")[0].split("/")[-1]}_segmentation_result.json', 'w')
    for data in tqdm.tqdm(img_list):
        image_path = os.path.join('/mnt/duyifan/data/coco/', data['image'])
        image = {'image': Image.open(image_path).convert('RGB')}
        result_image, categories = inference(image = image, task = ['Panoptic'])
        if len(set(categories)) <= 3 or data['image'] in img_list:
            continue
        result = {'image': data['image'], 'objects': list(set(categories))}
        result_list.append(result)
        segment_result_file.write(json.dumps(result) + '\n')
    print(f"segmentation result saved in segmentation/{img_path_file}_segmentation_result.json...")
    return result_list