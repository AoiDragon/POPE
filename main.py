import os
import argparse
import json
import random
from typing import List, Any

from utils import get_image, generate_ground_truth_objects, pope


# from SEEM.app import run_seem


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--auto_seg", type=bool, default=False, help="Whether to use automatic segmentation")
    parser.add_argument("--img_path", type=str, default="./image/", help="The path to images to be segmented")
    parser.add_argument("--seg_path", type=str, default="./segmentation/coco_ground_truth_segmentation.json", help="The segmentation file")
    parser.add_argument("--seg_num", type=int, default=1000, help="The number of images to be segmented")
    parser.add_argument("--sample_num", type=int, default=3, help="The number of sampled negative objects")
    parser.add_argument("--img_num", type=int, default=500, help="The number of images used for building POPE")
    parser.add_argument("--template", type=str, default="Is there a {} in the image?", help="The prompt template of POPE")
    parser.add_argument("--dataset", type=str, default="coco", help="The name of dataset, which is used as filename")
    parser.add_argument("--save_path", type=str, default="./output/", help="The save path of generated POPE")

    args = parser.parse_args()

    return args


def main():
    config = parse_args()
    
    if config.auto_seg:
        print("Start automatic segmentation...")
        from seem.seg import segment
        segment_results = segment(config.img_path, config.seg_num)
    else:
        segment_results = [json.loads(q) for q in open(config.seg_path, 'r')]
    processed_segment_results = []

    # Sample images which contain more than sample_num objects
    for image in segment_results:
        if len(image["objects"]) >= config.sample_num:
            processed_segment_results.append(image)

    assert len(processed_segment_results) >= config.img_num, "The number of images that contain more than {} objects is less than {}.".format(config.sample_num, config.img_num)
    processed_segment_results = random.sample(processed_segment_results, config.img_num)

    # Organize the ground truth objects and their co-occurring frequency
    save_path = os.path.join(config.save_path, config.dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ground_truth_objects = generate_ground_truth_objects(processed_segment_results, save_path, config.dataset)

    # Generate three kinds of POPE
    pope(ground_truth_objects, processed_segment_results, config.sample_num, config.template, "random", save_path, config.dataset)
    pope(ground_truth_objects, processed_segment_results, config.sample_num, config.template, "popular", save_path, config.dataset)
    pope(ground_truth_objects, processed_segment_results, config.sample_num, config.template, "adversarial", save_path, config.dataset)


if __name__ == '__main__':
    main()
