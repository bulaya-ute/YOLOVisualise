# import os
# from typing import Literal
#
# import yaml
#
#
# # def _get_data_paths_from_yaml(yaml_path, dataset_type: Literal["type1", "type2"]):
# #     with open(yaml_path, 'r') as file:
# #         data = yaml.safe_load(file)
# #     dataset_type = dataset_type.lower()
# #     splits = ["train", "val", "test"]
# #     image_folders = {}
# #     for split in splits:
# #         if split in data:
# #             image_folders[split] = data[split]
# #     if not image_folders:
# #         # Display descriptive error if no splits found in the file
# #         string = ""
# #         if len(splits) > 1:
# #             for split in splits[:-1]:
# #                 string += f"{split}, "
# #             string += string[:-2] + " or " + string[-1]
# #         else:
# #             string = splits[0]
# #         raise ValueError(f"Dataset does not contain {string} data.")
# #
# #     paths = []
# #
# #     if dataset_type == "type1":
# #         labels_dir = os.path.abspath(os.path.join(image_folders["train"], "..", "..", "labels"))
# #         split_folder_name = os.path.basename(image_folders["train"])
# #         for split, image_dir in image_folders.items():
# #             for image_name in os.listdir(image_dir):
# #                 if not image_name.endswith((".jpg", ".jpeg", ".png")):
# #                     continue
# #                 image_path = os.path.join(image_dir, image_name)
# #                 label_path = os.path.join(labels_dir, split_folder_name, os.path.splitext(image_name)[0] + '.txt')
# #                 if not os.path.exists(label_path):
# #                     label_path = None
# #                 paths.append({"image": image_path, "label": label_path, "split": split})
# #
# #     elif dataset_type == "type2":
# #         for split, image_dir in image_folders.items():
# #             label_dir = os.path.abspath(os.path.join(image_folders[split], "..", "labels"))
# #             for image_name in os.listdir(image_dir):
# #                 image_path = os.path.join(image_dir, image_name)
# #                 label_path = os.path.join(label_dir, os.path.splitext(image_name)[0] + '.txt')
# #                 if not os.path.exists(label_path):
# #                     label_path = None
# #                 paths.append({"image": image_path, "label": label_path, "split": split})
# #
# #     else:
# #         raise ValueError(f"Dataset type '{dataset_type}' not supported.")
# #     return paths
#
#
# pairs = _get_data_paths_from_yaml(
#     r'C:\Users\Bulaya\PycharmProjects\DentalDiseasesDetection\model\dental_seg_augmented_2\data.yaml', dataset_type="type1")
#
# [print(k) for k in pairs]
# # print(pairs)
