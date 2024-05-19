import copy
import os
from math import ceil
from random import uniform, choice, choices
from typing import Literal

import cv2
import numpy as np
from tqdm import tqdm

from utils import change_extension, warp_image_and_points, points_to_coords, fill_translucent_polygon, \
    relative_to_absolute_coords, coords_to_points, absolute_to_relative_coords, generate_unique_filename, \
    create_directory, select_from_weighted_sublists, darken_color, get_yolo_bounding_box


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
    def __init__(self, dataset_dir=None, data=None, task: Literal["detect", "segment"] = "segment"):
        self.content: list["DatasetEntry"] = []

        self.task = task
        if dataset_dir is not None:
            data_paths = _get_data_paths_from_dir(dataset_dir)
            self.load_data(data_paths, task=task)
        elif data is not None:
            raise NotImplementedError("Handling yaml files will be implemented in future.")

    def load(self, train_images_dir=None, val_images_dir=None, test_images_dir=None):
        directories_with_splits = []

        if train_images_dir:
            for img_path in [os.path.join(train_images_dir, f) for f in os.listdir(train_images_dir)]:
                directories_with_splits.append((img_path, "train"))
        if val_images_dir:
            for img_path in [os.path.join(val_images_dir, f) for f in os.listdir(val_images_dir)]:
                directories_with_splits.append((img_path, "val"))
        if test_images_dir:
            for img_path in [os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir)]:
                directories_with_splits.append((img_path, "test"))

        # Begin loading the data
        for img_path, split in tqdm(directories_with_splits, desc="Loading dataset"):
            # Load the image in RGB format
            loaded_image = cv2.imread(img_path)
            loaded_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)

            # Get label path and load labels
            label_path = os.path.join(img_path, "..", "..", "labels", change_extension(img_path, "txt"))
            if os.path.exists(label_path):
                with open(label_path, "r") as _file_obj:
                    _rows = _file_obj.readlines()
                    annotations = []
                    for _row in _rows:
                        _row_data = [eval(d) for d in _row.split()]
                        _class_index = _row_data[0]
                        annotations.append(_row_data)
            else:
                annotations = None

            self.content.append(DatasetEntry(loaded_image, annotations, split))

    def load_data(self, data: list[tuple[str, str, str]], task):
        for img_path, label_path, split in tqdm(data, desc="Loading data"):
            # Load the image in RGB format
            loaded_image = cv2.imread(img_path)
            loaded_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)

            if os.path.exists(label_path):
                with open(label_path, "r") as _file_obj:
                    _rows = _file_obj.readlines()
                    annotations = []
                    for _row in _rows:
                        _row_data = [eval(d) for d in _row.split()]
                        _class_index = _row_data[0]
                        annotations.append(_row_data)
            else:
                annotations = None

            self.content.append(DatasetEntry(loaded_image, annotations, split, task=task))

    def show_random_sample(self):
        weights = [int(bool(e.annotations)) for e in self.content]
        if not any(weights):
            sample = choice(self.content)
            sample.show(show_instances=False)
        else:
            sample = choices(self.content, weights=weights, k=1)[0]
            sample.show(show_instances=True)

    def normalise_class_indexes(self):
        unique_classes = set()
        index_mapping = {}
        for entry in self.content:
            unique_classes.union(set(entry.get_classes_present()))
        sorted_classes = sorted(list(unique_classes))
        for i, class_index in enumerate(sorted_classes):
            index_mapping[class_index] = i

        # Start updating the entries
        for entry in self.content:
            entry.remap_class_indexes(index_mapping)

    def save(self, output_dir):
        splits_saved = []
        print(f"Output directory: {output_dir}")
        for entry in tqdm(self.content, desc="Saving"):
            if entry.split not in splits_saved:
                splits_saved.append(entry.split)
                create_directory(os.path.join(output_dir, "images", entry.split))
                create_directory(os.path.join(output_dir, "labels", entry.split))
            entry.save(output_dir)

    def __add__(self, other):
        if isinstance(other, Dataset):
            new_dataset = copy.copy(self)
            new_dataset.content += other.content
            return new_dataset
        raise ValueError("Incompatible datatypes for this operation")

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.content[item]

    def convert_task(self, new_task: str):
        if new_task not in ("detect", "segment"):
            raise NotImplementedError("Tasks other than 'detect' and 'segment' will be implemented in future")

        entries_to_convert = [e for e in self.content if e.task != new_task]

        for entry in tqdm(entries_to_convert, desc=f"Converting dataset to {new_task}"):
            entry.convert_task(new_task)


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
        self.annotations = _annotations
        self.other_attrs = {}
        self.task = task

    @property
    def classes(self):
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

        cv2.imshow("Image", cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB))
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
        return "\n".join([" ".join([str(num) for num in line]) for line in self.annotations])

    def save(self, output_dir, create_dir=False):
        output_image_dir = os.path.join(output_dir, "images", self.split)
        output_label_dir = os.path.join(output_dir, "labels", self.split)

        if create_dir:
            create_directory(output_image_dir)
            create_directory(output_label_dir)

        image_basename = generate_unique_filename("1.jpg", output_image_dir)
        label_basename = change_extension(image_basename, "txt")

        cv2.imwrite(os.path.join(output_image_dir, image_basename), cv2.cvtColor(self._image, cv2.COLOR_RGB2BGR))
        with open(os.path.join(output_label_dir, label_basename), "w") as label_file:
            label_file.write(self.export_annotations())

    def remap_class_indexes(self, mapping: dict):
        """Mapping is a dict in format {old_index: new_index}"""
        for row in self.annotations:
            current = row[0]
            row[0] = mapping[current]

    def convert_task(self, new_task: str):
        if self.task == "segment":
            if new_task == "detect":
                for i, row in enumerate(self.annotations):
                    class_index = row.pop(0)
                    rel_points = row
                    rel_coords = points_to_coords(rel_points)
                    rel_bbox = get_yolo_bounding_box(rel_coords)
                    self.annotations[i] = [i]

                self.task = new_task
            else:
                NotImplementedError(f"Incompatible conversion from {self.task} to {new_task}")


if __name__ == "__main__":
    birds_dataset = Dataset(r"C:\Users\Bulaya\PycharmProjects\BirdsDetection\datasets\coco_birds")
    locusts_dataset = Dataset(r"C:\Users\Bulaya\PycharmProjects\BirdsDetection\datasets\locust_dataset",
                              task="detect")
    merged_dataset = locusts_dataset + birds_dataset
    for _ in range(20):
        merged_dataset.show_random_sample()
        # locusts_dataset.show_random_sample()
        # birds_dataset.show_random_sample()
    # dental_dataset.show_random_sample()
