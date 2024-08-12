import os
from typing import Literal

import yaml


def get_data_paths_from_yaml(yaml_path, dataset_type: Literal["type1", "type2"]):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    splits = ["train", "val", "test"]
    dataset_root_dir = os.path.dirname(data["train"])

    def _get_split_paths(split):
        image_dir = data[split]
        # Derive the label directory by navigating up one level and then into the 'labels' folder
        label_dir = os.path.join(os.path.dirname(image_dir), 'labels', os.path.basename(image_dir))
        image_label_pairs = []

        for image_name in os.listdir(image_dir):
            if image_name.endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(image_dir, image_name)
                label_name = os.path.splitext(image_name)[0] + '.txt'
                label_path = os.path.join(label_dir, label_name)

                if not os.path.exists(label_path):
                    # image_label_pairs.append((image_path, label_path, split))
                    label_path = None
                # else:
                #     image_label_pairs.append((image_path, None, split))

                image_label_pairs.append({"image": image_path, "label": label_path, "split": split})

        return image_label_pairs

    paths = []
    if dataset_type == "type1":
        for split in splits:
            if split in data:
                paths.append(_get_split_paths(split))

    if dataset_type == "type2":

    return paths


pairs = get_data_paths_from_yaml(
    r'C:\Users\Bulaya\PycharmProjects\DentalDiseasesDetection\model\dental_seg_augmented_2\data.yaml')

[print(k) for k in pairs]
