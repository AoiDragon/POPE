import os
import random
import json


def get_image(img_path, seg_num):
    img_list = [os.path.join(img_path, img) for img in os.listdir(img_path)]
    sampled_images = random.sample(img_list, seg_num)
    return sampled_images


def create_question(question_id, image, Object, label, template):
    question = dict()
    question["question_id"] = question_id
    question["image"] = image
    template1 = template
    template2 = template.replace("a", "an")
    if Object[0] not in ["a", "e", "i", "o", "u"]:
        question["text"] = template1.format(Object)
    elif Object[0] in ["a", "e", "i", "o", "u"]:
        question["text"] = template2.format(Object)
    question["label"] = label
    return question


def pope(ground_truth_objects, segment_results, sample_num, template, neg_strategy, save_path, dataset):
    question_list = []
    question_id = 1
    output_file = os.path.join(save_path, dataset + "_pope_" + neg_strategy + ".json")

    gt_objects_list = list(ground_truth_objects.keys())
    sorted_objects = sorted(ground_truth_objects.items(), key=lambda x: x[1], reverse=True)
    sorted_co_occur = compute_co_occurrence(segment_results, save_path, dataset)

    for image in segment_results:
        history_object_list = []

        # Positive sampling
        for i in range(sample_num):
            pos_object = image["objects"][i]
            history_object_list.append(pos_object)
            question = create_question(question_id, image['image'], pos_object, 'yes', template)
            question_list.append(question)
            question_id += 1

            # Negative sampling (random)
            if neg_strategy == "random":
                selected_object = random.choice(gt_objects_list)
                while selected_object in history_object_list or selected_object in image["objects"]:
                    selected_object = random.choice(gt_objects_list)
                history_object_list.append(selected_object)
                question = create_question(question_id, image["image"], selected_object, 'no', template)
                question_list.append(question)
                question_id += 1

            # Negative sampling (popular)
            elif neg_strategy == "popular":
                flag = 0
                for j in range(len(sorted_objects)):
                    selected_object = sorted_objects[j][0]
                    if selected_object not in history_object_list and selected_object not in image["objects"]:
                        history_object_list.append(selected_object)
                        question = create_question(question_id, image["image"], selected_object, 'no', template)
                        question_list.append(question)
                        question_id += 1
                        flag = 1
                        break

                # In case no object is selected
                if not flag:
                    while True:
                        selected_object = random.choice(gt_objects_list)
                        if selected_object not in history_object_list and selected_object not in image["objects"]:
                            history_object_list.append(selected_object)
                            question = create_question(question_id, image["image"], selected_object, 'no', template)
                            question_list.append(question)
                            question_id += 1
                            break

            # Negative sampling (Adversarial)
            elif neg_strategy == "adversarial":
                flag = 0
                for j in range(len(sorted_co_occur[pos_object])):
                    selected_object = sorted_co_occur[pos_object][j]
                    if selected_object not in history_object_list and selected_object not in image["objects"]:
                        history_object_list.append(selected_object)
                        question = create_question(question_id, image["image"], selected_object, 'no', template)
                        question_list.append(question)
                        question_id += 1
                        flag = 1
                        break

                if not flag:
                    while True:
                        selected_object = random.choice(gt_objects_list)
                        if selected_object not in history_object_list and selected_object not in image["objects"]:
                            history_object_list.append(selected_object)
                            question = create_question(question_id, image["image"], selected_object, 'no', template)
                            question_list.append(question)
                            question_id += 1
                            break

    with open(output_file, 'w') as f:
        for question in question_list:
            json_str = json.dumps(question)
            f.write(json_str + "\n")


def generate_ground_truth_objects(segment_results, save_path, dataset):
    gt_objects = dict()
    output_file = os.path.join(save_path, dataset + "_ground_truth_objects.json")

    for image in segment_results:
        seg = image['objects']
        for o in seg:
            if o not in gt_objects:
                gt_objects[o] = 1
            else:
                gt_objects[o] += 1

    with open(output_file, 'w') as f:
        json_str = json.dumps(gt_objects)
        f.write(json_str)

    return gt_objects


def compute_co_occurrence(segment_results, save_path, dataset):
    output_file = os.path.join(save_path, dataset + "_co_occur.json")
    co_occur = dict()

    for image in segment_results:
        objects = image["objects"]
        for o in objects:
            if o not in co_occur:
                co_occur[o] = dict()
            for other_o in objects:
                if o == other_o:
                    continue
                if other_o not in co_occur[o]:
                    co_occur[o][other_o] = 1
                else:
                    co_occur[o][other_o] += 1

    sorted_co_occur = dict()
    for o in co_occur:
        objects = co_occur[o]
        sorted_co_occur_objects = sorted(objects.items(), key=lambda x: x[1], reverse=True)
        sorted_co_occur[o] = [item[0] for item in sorted_co_occur_objects]

    with open(output_file, 'w') as f:
        json_str = json.dumps(sorted_co_occur)
        f.write(json_str)

    return sorted_co_occur
