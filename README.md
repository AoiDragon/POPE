# POPE: Polling-based Object Probing Evaluation for Object Hallucination

This repo provides the source code & data of our paper: [Evaluating Object Hallucination in Large Vision-Language Models (Arxiv 2023)](https://arxiv.org/abs/2305.10355).

```
@InProceedings{Li-hallucination-2023,
  author =  {Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao and Ji-Rong Wen},
  title =   {Evaluating Object Hallucination in Large Vision-Language Models},
  year =    {2023},  
  journal={arXiv preprint arXiv:2305.10355},
  url={https://arxiv.org/pdf/2305.10355}
}
```

<img src="./assets/POPE.png" alt="image-20230517233229650" style="zoom:80%;" />

## Update
- [10/8] 🎉🎉🎉 Our work is accepted to EMNLP 2023！
- [6/25] We upload the caption result used in Table 2. You can find them under `./caption_data`.
- [6/3] We incorporate the code for using POPE with SEEM to handle unannotated datasets.
- [6/1] We release the code of POPE.

## Collect ground-truth objects

POPE can be easily built based on datasets with annotations about objects in the image. With the help of automatic segmentation tools like SEEM, you can also build POPE on any dataset you want to test.

### From annotations

If you want to buile POPE on datasets with object annotations (*e.g.* COCO) , you should first organize the annotations in a json file with the following format:

```json
{"image": "COCO_val2014_000000131089.jpg", "objects": ["person", "baseball bat"]}
{"image": "COCO_val2014_000000393225.jpg", "objects": ["bowl", "spoon", "carrot"]}
```

Here the `image` is the filename of each image and the `objects` is the objects in the image extracted from annotations. You can also add other keys (*e.g.* `image_id` in COCO) , but the above two must be included.

We provide the annotation results of the validation set of COCO 2014 under [./segmentation/](./segmentation/coco_ground_truth_segmentation.json). We refer to [LisaAnne/Hallucination](https://github.com/LisaAnne/Hallucination) to collect objects from segmentations and captions.

### From automatic segmentation results

Besides building POPE with object annotations, our method also supports building POPE on raw images. Leveraing the automatic segmentation tools (e.g. [SEEM](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once)), we can first extract the objects in the image and then build the POPE as the above method. If you want to buile POPE with SEEM , you should first organize the annotations in a json file with the following format:

```json
[{"image": "COCO_val2014_000000131089.jpg"}, {"image": "COCO_val2014_000000393225.jpg"}]
```

## Build POPE

Once you have the ground-truth objects prepared, you can build your own POPE by running:

```python
python main.py
```

You can customize your POPE by specifying these configs:

- `--auto_seg`: Whether to use automatic segmentation results. Default = False.
- `--img_path`: The path to the json file that contains the path of images to be segmented.
- `--seg_path`: The path to the segmentation path that containing ground-truth objects in the image.
- `--seg_num`: The number of images to be segmented. Default = 1000.
- `--sample_num`: The number of  negative objects to be sample for each image. Default = 3.
- `--img_num`: The number of images for building POPE. Default = 500.
- `--template`: The prompt template. Default = "Is there a {} in the image?".
- `--dataset`: The dataset name used for the filename of the built POPE. Default = "coco".
- `--save_path`: The save path of the built POPE. Default = "./output/"

If you want to employ SEEM to segment images, you can build POPE by running:

```python
python main.py --auto_seg True --img_path ./segmentation/coco_val_images.json --seg_num 1000
```

After the execution, you will find 5 json files under "./output/{dataset}/":

- `{dataset}_pope_random.json`: The POPE built with random negative sampling strategy. Each question is a `dict` in the following format:

  ```json
  {"question_id": 1, "image": "COCO_val2014_000000016631.jpg", "text": "Is there a person in the image?", "label": "yes"}
  {"question_id": 2, "image": "COCO_val2014_000000016631.jpg", "text": "Is there a refrigerator in the image?", "label": "no"}
  ```

- `{dataset}_pope_popular.json`: The POPE built with popular negative sampling strategy.

- `{dataset}_pope_adversarial.json`: The POPE built with adversarial negative sampling strategy.

- `{dataset}_ground_truth_objects.json`: The appearance frequencies of all objects in the selected images.

- `{dataset}_co_occur.json`: The co-occurrence frequencies of all objects in the selected images.


## Evaluation

Now you can use built POPE to evaluate LVLMs and evaluate their object hallucination. The answer of LVLMs should be organized in a json file in the following format:

```json
{"question": "is there a bird in the image?", "answer": "yes"}
{"question": "is there a tree in the image?", "answer": "no"}
```

Notice that the order of POPE questions and the answer should be the same.

Then you should specify the `ans_file` (*i.e.* the answer file of LVLMs) and `label_file` (*i.e.* the POPE file) in "evaluate.py" and evaluate the results by running:

```python
python evaluate.py
```

The program will report the Accuracy, Precision, Recall, F1 Score and Yes ratio as metrics.
