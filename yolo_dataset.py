from copy import copy
import os
from math import ceil
from random import uniform, choice, choices, shuffle
from typing import Literal

import cv2
import numpy as np
from tqdm import tqdm

from utils import change_extension, warp_image_and_points, points_to_coords, fill_translucent_polygon, \
    relative_to_absolute_coords, coords_to_points, absolute_to_relative_coords, generate_unique_filename, \
    create_directory, select_from_weighted_sublists, darken_color, get_yolo_bbox_from_polygon, clip_polygon_to_bbox, \
    clip_bbox_to_bbox


def _get_data_paths_from_dir(dataset_dir):
    splits = ("train", "val", "test")
    data_paths = []  # In [(image_path, label_path, split), ...] format

    if os.path.exists(f"{dataset_dir}/images"):
        for split in splits:
            images_dir = os.path.join(dataset_dir, "images", split)
            if os.path.exists(images_dir):
                for img_path in [os.path.join(images_dir, f) for f in os.listdir(images_dir)]:
                    # Check that the image exists
                    if img_path.split(".")[-1].lower() in ("jpg", "jpeg", "png"):
                        label_path = os.path.join(dataset_dir, "labels", split,
                                                  change_extension(os.path.basename(img_path), "txt"))
                        data_paths.append((img_path, label_path, split))
        return data_paths

    for split in splits:
        if os.path.exists(os.path.join(dataset_dir, split)):
            for img_path in [os.path.join(dataset_dir, split, "images", f) for
                             f in os.listdir(os.path.join(dataset_dir, split, "images"))]:
                label_path = os.path.join(dataset_dir, split, "labels",
                                          change_extension(os.path.basename(img_path), "txt"))
                data_paths.append((img_path, label_path, split))

    return data_paths


class Dataset:
    def __init__(self, dataset_dir=None, data=None, task: str = "segment"):
        self.contents: list["DatasetEntry"] = []
        self.task = task
        if dataset_dir is not None:
            data_paths = _get_data_paths_from_dir(dataset_dir)
            self._load_data(data_paths, task=task)
        elif data is not None:
            raise NotImplementedError("Handling yaml files will be implemented in future.")

        self.class_names = {class_index: f"class_{class_index}" for class_index in self.classes_present}

    def _load_data(self, data: list[tuple[str, str, str]], task, clip=False):
        for img_path, label_path, split in tqdm(data, desc="Loading dataset"):

            # Load the image in RGB format
            loaded_image = cv2.imread(img_path)
            loaded_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)

            if os.path.exists(label_path):
                with open(label_path, "r") as _file_obj:
                    _rows = _file_obj.readlines()
                    annotations = []
                    for _row in _rows:
                        _row_data = [eval(d) for d in _row.split()]
                        _class_index = _row_data.pop(0)
                        if clip:
                            if task == "segment":
                                points_as_coords = points_to_coords(_row_data[1:])
                                _row_data = clip_polygon_to_bbox((0, 0, 1, 1), points_as_coords)
                                if not _row_data:
                                    continue
                                _row_data = coords_to_points(_row_data)
                                # print("before clip", _row_data[1:])
                                # print("after clip", _row_data)
                            elif task == "detect":
                                _row_data = clip_bbox_to_bbox((0, 0, 1, 1), _row_data[1:])
                            else:
                                raise NotImplementedError("Other tasks not yet implemented!")
                            if not _row_data:
                                continue

                        annotations.append([_class_index] + _row_data)
            else:
                annotations = None

            self.contents.append(DatasetEntry(loaded_image, annotations, split, task=task))

    def name_classes(self, class_names: dict):
        self.class_names.update(class_names)

    def show_random_sample(self):
        weights = [int(bool(e.annotations)) for e in self.contents]
        if not any(weights):
            sample = choice(self.contents)
            sample.show(show_instances=False)
        else:
            sample = choices(self.contents, weights=weights, k=1)[0]
            sample.show(show_instances=True)

    def normalise_class_indexes(self):
        unique_classes = set()
        index_mapping = {}
        for entry in self.contents:
            unique_classes = unique_classes.union(set(entry.classes_present))

        sorted_classes = sorted(list(unique_classes))
        for i, class_index in enumerate(sorted_classes):
            index_mapping[class_index] = i

        # Start updating the entries
        for entry in self.contents:
            entry.remap_class_indexes(index_mapping)

        new_names = {}
        for old_index, class_name in self.class_names.items():
            new_index = index_mapping[old_index]
            new_names[new_index] = class_name
        self.class_names = new_names

    @property
    def classes_present(self):
        classes = set()
        for entry in self.contents:
            classes = classes.union(entry.classes_present)
        return classes

    def save(self, output_dir):
        self.normalise_class_indexes()

        split_counts = {}
        for entry in tqdm(self.contents, desc="Saving"):
            if entry.split not in split_counts:
                split_counts[entry.split] = 1
                create_directory(os.path.join(output_dir, "images", entry.split))
                create_directory(os.path.join(output_dir, "labels", entry.split))
            else:
                split_counts[entry.split] += 1

            img_basename = f"{split_counts[entry.split]}.jpg"
            label_basename = change_extension(img_basename, "txt")
            entry.save_image(os.path.join(output_dir, "images", entry.split, img_basename))
            entry.save_label(os.path.join(output_dir, "labels", entry.split, label_basename))

        with open(os.path.join(output_dir, "data.yaml"), "w") as file_obj:
            data = f"train: {os.path.join(output_dir, 'train', 'images')}\n"
            data += f"val: {os.path.join(output_dir, 'val', 'images')}\n"
            data += f"nc: {len(self.classes_present)}\n"
            data += f"classes: {[self.class_names[index] for index in sorted(self.class_names.keys())]}\n"
            file_obj.write(data)

    def __add__(self, other):
        if isinstance(other, Dataset):
            # new_dataset = copy.copy(self)
            self.contents += other.contents
            return self
        raise ValueError("Incompatible datatypes for this operation")

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.contents[item]

    def convert_task(self, new_task: str):
        if new_task not in ("detect", "segment"):
            raise NotImplementedError("Tasks other than 'detect' and 'segment' will be implemented in future")

        entries_to_convert = [e for e in self.contents if e.task != new_task]

        for entry in tqdm(entries_to_convert, desc=f"Converting dataset to {new_task}"):
            entry.convert_task(new_task)

    def add_augmentations(self, ratio: int = 0.3, ):
        for _ in tqdm(range(ceil(ratio * len(self.contents))), desc="Augmenting"):
            class_frequencies = [c.classes_with_duplicates for c in self.contents]
            selected_index = select_from_weighted_sublists(class_frequencies)

            # Randomly picked dataset entry to be duplicated, then augmented
            selected_entry = self.contents[selected_index]

            new_entry = copy(selected_entry)
            new_entry.augment()
            self.contents.append(new_entry)

    def shuffle_data(self):
        shuffle(self.contents)

    def show_numbers(self):
        split_data = {"train": {cls: 0 for cls in self.classes_present},
                      "val": {cls: 0 for cls in self.classes_present},
                      "test": {cls: 0 for cls in self.classes_present}}

        for entry in self.contents:
            for instance_row in entry.annotations:
                class_index = instance_row[0]
                split_data[entry.split][class_index] += 1

        print("=======================================================")
        for split, data in split_data.items():
            print(f"{split.upper()}:")
            for cls, freq in data.items():
                print(f"\t{cls}: {freq}")
        print(len)
        print("=======================================================")


class DatasetEntry:
    # Colors are in BGR format
    colors = [
        [96, 43, 186],  # Blue
        [255, 255, 255],  # White
        [147, 112, 219],  # Purple
        [67, 160, 71],  # Green
        [0, 255, 0],  # Lime
        [189, 83, 107],  # Red
        [255, 235, 59],  # Yellow
        [0, 0, 0],  # Black
        [0, 140, 255],  # Orange
        [0, 128, 128],  # Teal
        [230, 126, 179],  # Pink
        [128, 128, 128]  # Gray
    ]

    def __init__(self, _image: np.ndarray, _annotations: list[list[float]],
                 split: Literal["train", "val", "test"], task="segment"):
        self.split = split
        self._image = _image
        if not _annotations:
            self.annotations = []
        else:
            self.annotations = _annotations
        self.other_attrs = {}
        self.task = task

    @property
    def classes_present(self):
        if not self.annotations:
            return []
        return sorted(list({label[0] for label in self.annotations}))

    @property
    def classes_with_duplicates(self):
        return sorted([label[0] for label in self.annotations])

    def get_image(self):
        return self._image

    def get_points(self):
        return [p[1:] for p in self.annotations]

    def get_classes_present(self):
        classes = set()
        for row in self.annotations:
            classes.add(int(row[0]))
        return sorted(list(classes))

    def set_points(self, new_points):
        for old, new in zip(new_points, self.get_points()):
            if len(old) != len(new):
                raise ValueError("Mismatching lengths of lists")

        for _i, (new_row, old_row) in enumerate(zip(new_points, self.get_points())):
            for j, new_element in enumerate(new_row):
                self.annotations[_i][j + 1] = new_element
            pass

    def show(self, show_instances=True):
        if show_instances:
            h, w = self._image.shape[:2]
            for instance_details in self.annotations:
                instance_details = instance_details[::]
                _class_index = int(instance_details.pop(0))
                color = self.colors[_class_index % len(self.colors)]
                if self.task == "segment":
                    abs_coords = relative_to_absolute_coords(points_to_coords(instance_details), w, h)
                elif self.task == "detect":
                    x_rel, y_rel, w_rel, h_rel = instance_details
                    xc, yc, ww, hh = x_rel * w, y_rel * h, w_rel * w * 0.5, h_rel * h * 0.5
                    abs_coords = [(xc - ww, yc - hh), (xc + ww, yc - hh), (xc + ww, yc + hh), (xc - ww, yc + hh)]
                else:
                    raise ValueError(f"Unknown task '{self.task}'")
                for i, start_point in enumerate(abs_coords):
                    end_point = (abs_coords + [abs_coords[0]])[i + 1]
                    start_point = [int(p) for p in start_point]
                    end_point = [int(p) for p in end_point]
                    cv2.line(self._image, start_point, end_point, darken_color(color), 2)
                fill_translucent_polygon(self._image, abs_coords, color)

        cv2.imshow(self.split, cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
        try:
            cv2.destroyWindow("Image")
        except cv2.error:
            pass

    def augment(self, h_shear=None, v_shear=None, rotate=None, flip=None):
        if h_shear is None:
            h_shear = uniform(-0.5, 0.5)
        if v_shear is None:
            v_shear = uniform(-0.5, 0.5)
        if rotate is None:
            rotate = uniform(-10.0, 10.0)
        if flip is None:
            flip = choice(["h", None])

        img_height, img_width = self._image.shape[:2]
        coords = points_to_coords(self.get_points())
        for _i, line in enumerate(coords):
            for _j, (_x, _y) in enumerate(line):
                coords[_i][_j] = [_x * img_width, _y * img_height]

        self._image, new_coords = warp_image_and_points(self._image, h_shear, v_shear, rotate, flip,
                                                        image_points=coords)
        new_coords = absolute_to_relative_coords(new_coords, img_width, img_height)
        new_points = coords_to_points(new_coords)
        self.set_points(new_points)

    def export_annotations(self):
        if self.annotations:
            return "\n".join([" ".join([str(num) for num in line]) for line in self.annotations])
        return ""

    def save_image(self, file_path):
        # output_image_dir = os.path.join(output_dir, "images", self.split)
        # output_label_dir = os.path.join(output_dir, "labels", self.split)
        #
        # if create_dir:
        #     create_directory(output_image_dir)
        #     create_directory(output_label_dir)

        # image_basename = generate_unique_filename("1.jpg", output_image_dir)
        # label_basename = change_extension(image_basename, "txt")

        cv2.imwrite(file_path, cv2.cvtColor(self._image, cv2.COLOR_RGB2BGR))
        # if self.annotations:
        #     with open(os.path.join(output_label_dir, label_basename), "w") as label_file:
        #         label_file.write(self.export_annotations())

    def save_label(self, file_path):
        if self.annotations:
            with open(file_path, "w") as label_file:
                label_file.write(self.export_annotations())

    def remap_class_indexes(self, mapping: dict):
        """Mapping is a dict in format {old_index: new_index}"""
        if not self.annotations:
            return
        for row in self.annotations:
            current = row[0]
            row[0] = mapping[current]

    def convert_task(self, new_task: str):
        if self.task == "segment":
            if new_task == "detect":
                for i, row in enumerate(self.annotations):
                    class_index = row.pop(0)
                    instance_poly = points_to_coords(row)
                    instance_bbox = get_yolo_bbox_from_polygon(instance_poly)
                    self.annotations[i] = [class_index] + instance_bbox
                    if len(instance_bbox) != 4:
                        print("Before", row)
                        print("After", instance_bbox, "\n")
                self.task = new_task
            else:
                NotImplementedError(f"Unimplemented conversion from {self.task} to {new_task}")


if __name__ == "__main__":
    locusts_dataset = Dataset(r"C:\Users\Bulaya\PycharmProjects\BirdsDetection\datasets\locust_dataset",
                              task="detect")
    locusts_dataset.show_numbers()

    birds_dataset = Dataset(r"C:\Users\Bulaya\PycharmProjects\BirdsDetection\datasets\coco_birds")
    birds_dataset.show_numbers()

    for _ in range(10):
        birds_dataset.show_random_sample()
    for _ in range(10):
        locusts_dataset.show_random_sample()

    merged_dataset = locusts_dataset + birds_dataset
    merged_dataset.show_numbers()

    merged_dataset.convert_task("detect")
    # print("before normalisation", merged_dataset.classes_present)
    merged_dataset.normalise_class_indexes()
    # print("after normalisation", merged_dataset.classes_present)

    # input()
    for _ in range(20):
        merged_dataset.show_random_sample()
        # locusts_dataset.show_random_sample()
        # birds_dataset.show_random_sample()
    merged_dataset.save("./birds_and_locust_detection")
    # dental_dataset.show_random_sample()
