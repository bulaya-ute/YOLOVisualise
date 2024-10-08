from copy import copy
import os
from math import ceil
from random import uniform, choice, choices, shuffle
from typing import Literal, Optional
import cv2
import numpy as np
import yaml
from tqdm import tqdm

from utils import change_extension, warp_image_and_points, points_to_coords, fill_translucent_polygon, \
    relative_to_absolute_coords, coords_to_points, absolute_to_relative_coords, \
    create_directory, select_from_weighted_sublists, darken_color, get_yolo_bbox_from_polygon, clip_polygon_to_bbox, \
    clip_bbox_to_bbox, resize_image_to_fit

"""
Type 1 Dataset Structure:
    dataset_dir
        images
            train
            val
            test
        labels
            train
            val
            test

Type 2 Dataset Structure:
    dataset_dir
        train:
            images
            labels
        val:
            images
            labels
        test:
            images
            labels
"""


def _get_dataset_structure_type(_dataset: str) -> Literal["type1"] | Literal["type2"]:
    """
    Determines the dataset structure type of the dataset presented.
    :param _dataset: Must be either path to dataset directory or yaml file
    :return: "type1" or "type2"
    """
    splits = ["train", "val", "test"]

    # First check if a directory has been provided
    if os.path.isdir(_dataset):
        dataset_dir = _dataset
        # Check if it is a type 1 dataset
        if os.path.exists(f"{dataset_dir}/images"):
            return "type1"
        # Check if it is a type 2 dataset
        elif any([os.path.exists(f"{dataset_dir}/{split}") for split in splits]):
            return "type2"
        else:
            raise ValueError("Dataset directory is invalid.")

    elif os.path.isfile(_dataset):
        with open(_dataset, 'r') as file:
            data = yaml.safe_load(file)
        image_folders = {}
        for split in splits:
            if split in data:
                image_folder = data[split]
                parent_dir = str(os.path.basename(os.path.dirname(image_folder)))
                if parent_dir.lower() == "images":
                    return "type1"
                else:
                    return "type2"
        raise ValueError("This file doesn't contain any dataset split data. Unable to get dataset structure type.")

    else:
        raise ValueError("The file/directory provided does not exist or is invalid.")


def _get_data_paths_from_dir(dataset_dir, dataset_type: Literal["type1"] | Literal["type2"]) \
        -> list[dict[str, str | None]]:
    """
    Returns a list of dictionaries containing the data paths of all the dataset
    images, as well as their associated labels (if present) and the split they belong to.

    :param dataset_dir: The path to the dataset directory.
    :return: A list of dictionaries, each in the format:
        {"image": image_path, "label": label_path | None, "split": split}
    """

    dataset_type = dataset_type.lower()

    def find_similar_file(path: str) -> Optional[str]:
        dir_name = os.path.dirname(path)
        filename = str(os.path.basename(path).lower())

        dir_contents = os.listdir(dir_name)
        if filename in dir_contents:
            return os.path.join(dir_name, filename)

        for dir_file in [f.lower() for f in dir_contents]:
            if filename.lower() in dir_file:
                return os.path.join(dir_name, dir_file)
        return None

    splits = ("train", "val", "test")
    data_paths = []  # Format: [{"image": image_path, "labels": label_path, "split": split}, ...]

    # Check if it is a type 1 dataset
    if dataset_type == "type1":
        for split in splits:
            images_dir = os.path.join(dataset_dir, "images", split)
            split_dir_name = find_similar_file(images_dir)
            if split_dir_name:
                split_basename = os.path.basename(split_dir_name)
                images_dir = split_dir_name
                for img_path in [os.path.join(images_dir, f) for f in os.listdir(images_dir)]:
                    # Check that the image exists
                    if img_path.split(".")[-1].lower() in ("jpg", "jpeg", "png"):
                        label_path = os.path.join(dataset_dir, "labels", split_basename,
                                                  change_extension(os.path.basename(img_path), "txt"))
                        data_paths.append({"image": str(img_path), "label": str(label_path), "split": str(split)})

    # Assume it is a type 2 dataset
    elif dataset_type == "type2":
        for split in splits:
            split_dir = os.path.join(dataset_dir, split)
            if find_similar_file(split_dir):
                split_dir = find_similar_file(split_dir)
                for img_path in [os.path.join(split_dir, "images", f) for
                                 f in os.listdir(os.path.join(dataset_dir, split, "images"))]:
                    label_path = os.path.join(dataset_dir, split, "labels",
                                              change_extension(os.path.basename(img_path), "txt"))
                    data_paths.append({"image": str(img_path), "label": str(label_path), "split": str(split)})

    # In case something illegal is provided
    else:
        raise ValueError(f"Dataset type '{dataset_type}' is not supported or is invalid.")

    return data_paths


def _get_data_paths_from_yaml(yaml_data: dict, dataset_type: Literal["type1", "type2"]):
    dataset_type = dataset_type.lower()
    splits = ["train", "val", "test"]
    image_folders = {}
    for split in splits:
        if split in yaml_data:
            image_folders[split] = yaml_data[split]
    if not image_folders:
        # Display descriptive error if no splits found in the file
        string = ""
        if len(splits) > 1:
            for split in splits[:-1]:
                string += f"{split}, "
            string += string[:-2] + " or " + string[-1]
        else:
            string = splits[0]
        raise ValueError(f"Dataset does not contain {string} data.")

    paths = []

    if dataset_type == "type1":
        for split, image_dir in image_folders.items():
            labels_dir = os.path.abspath(os.path.join(image_folders[split], "..", "..", "labels"))
            split_folder_name = os.path.basename(image_folders[split])
            for image_name in os.listdir(image_dir):
                if not image_name.endswith((".jpg", ".jpeg", ".png")):
                    continue
                image_path = os.path.join(image_dir, image_name)
                label_path = os.path.join(labels_dir, split_folder_name, os.path.splitext(image_name)[0] + '.txt')
                if not os.path.exists(label_path):
                    label_path = None
                paths.append({"image": image_path, "label": label_path, "split": split})

    elif dataset_type == "type2":
        for split, image_dir in image_folders.items():
            label_dir = os.path.abspath(os.path.join(image_folders[split], "..", "labels"))
            for image_name in os.listdir(image_dir):
                image_path = os.path.join(image_dir, image_name)
                label_path = os.path.join(label_dir, os.path.splitext(image_name)[0] + '.txt')
                if not os.path.exists(label_path):
                    label_path = None
                paths.append({"image": image_path, "label": label_path, "split": split})

    else:
        raise ValueError(f"Dataset type '{dataset_type}' not supported.")
    return paths#


def _get_class_names_from_yaml(yaml_data: dict) -> dict[int, str]:
    classes = yaml_data["classes"]
    return {i: classes[i] for i in range(len(classes))}


class Dataset:
    def __init__(self, _dataset, task: str = "segment"):
        """
        Load a YOLO dataset. The entire dataset will be in memory, including the annotations.

        :param _dataset: Path to .yaml file containing dataset information or path to directory containing dataset images
            and labels in YOLO format.
        :param task: The task the dataset is for e.g. "segmentation", "detection".
        """
        self.contents: list["DatasetEntry"] = []
        self.task = task
        self.other_attrs = {}

        dataset_type = _get_dataset_structure_type(_dataset)
        if os.path.isfile(_dataset):
            with open(_dataset, 'r') as file:
                yaml_data = yaml.safe_load(file)

            data_paths = _get_data_paths_from_yaml(yaml_data, dataset_type)
            self._load_data(data_paths, task=task)
            self.class_names = _get_class_names_from_yaml(yaml_data)
        elif os.path.isdir(_dataset):
            data_paths = _get_data_paths_from_dir(_dataset, dataset_type)
            self._load_data(data_paths, task=task)
            self.class_names = {class_index: f"class_{class_index}" for class_index in self.classes_present}

        else:
            raise ValueError("The .yaml file or directory provided does not exist.")

    def _load_data(self, data: list[dict[str, str | None]], task, clip=False):
        for entry in tqdm(data, desc="Loading dataset"):
            img_path, label_path, split = entry["image"], entry["label"], entry["split"]

            # Load the image in RGB format
            loaded_image = cv2.imread(img_path)
            loaded_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)

            if label_path and os.path.exists(label_path):
                with open(label_path, "r") as _file_obj:
                    _rows = _file_obj.readlines()
                    annotations = []
                    for _row in _rows:
                        _row_data = [eval(d) for d in _row.split()]
                        _class_index = _row_data.pop(0)
                        if clip:
                            if task == "segment":
                                # points_as_coords = points_to_coords(_row_data[1:])
                                # _row_data = clip_polygon_to_bbox((0.0, 0.0, 1.0, 1.0), points_as_coords)
                                # if not _row_data:
                                #     continue
                                # _row_data = coords_to_points(_row_data)
                                pass
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

    def remove_unannotated(self):
        removed = 0
        i = 0
        while i < len(self.contents):
            entry = self.contents[i]
            if entry.annotations:
                i += 1
            else:
                self.contents.pop(i)
                removed += 1
        print(f"Removed {removed} background images.")

    def name_classes(self, class_names: dict):
        self.class_names.update(class_names)

    def show_random_sample(self):
        weights = [int(bool(e.annotations)) + 1 for e in self.contents]
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

    def export(self, output_dir, _format="YOLO", variation="type1"):
        if _format == "YOLO":
            if variation == "type1":
                # self.normalise_class_indexes()

                for s in {e.split for e in self.contents}:
                    create_directory(os.path.join(output_dir, "images", s))
                    create_directory(os.path.join(output_dir, "labels", s))

                split_counts = {}
                for entry in tqdm(self.contents, desc="Saving"):
                    if entry.split not in split_counts:
                        split_counts[entry.split] = 1
                    else:
                        split_counts[entry.split] += 1

                    img_basename = f"{split_counts[entry.split]}.jpg"
                    label_basename = change_extension(img_basename, "txt")
                    entry.save_image(os.path.join(output_dir, "images", entry.split, img_basename))
                    entry.save_label(os.path.join(output_dir, "labels", entry.split, label_basename))
            else:
                raise NotImplementedError("Other variations of YOLO not yet implemented!")

            with open(os.path.join(output_dir, "data.yaml"), "w") as file_obj:
                data = f"train: {os.path.join(output_dir, 'images', 'train')}\n"
                data += f"val: {os.path.join(output_dir, 'images', 'val')}\n"
                data += f"nc: {len(self.classes_present)}\n"
                data += f"classes: {[self.class_names[index] for index in sorted(self.class_names.keys())]}\n"
                file_obj.write(data)
        else:
            raise NotImplementedError("Only YOLO format is supported currently.")

    def __add__(self, other):
        if isinstance(other, Dataset):
            # new_dataset = copy.copy(self)
            self.contents += other.contents
            return self
        raise ValueError("Incompatible datatypes for this operation")

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

            new_entry = selected_entry.copy()
            new_entry.augment()
            self.contents.append(new_entry)

    def shuffle_data(self):
        shuffle(self.contents)

    def show_instance_counts(self):
        split_data = {"train": {cls: 0 for cls in self.classes_present},
                      "val": {cls: 0 for cls in self.classes_present},
                      "test": {cls: 0 for cls in self.classes_present}}

        for entry in self.contents:
            for instance_row in entry.annotations:
                class_index = instance_row[0]
                split_data[entry.split][class_index] += 1

        print("===================== INSTANCE COUNTS =====================")

        for split, data in split_data.items():
            print(f"{split.upper()}:")
            for cls, freq in data.items():
                print(f"\t{cls}: {freq}")
        print(len(self.contents))
        print("===========================================================")

    def copy(self):
        contents_copy = [entry.copy() for entry in self.contents]
        class_names_copy = {copy(k): copy(v) for k, v in self.class_names.items()}
        other_attrs_copy = {copy(k): copy(v) for k, v in self.other_attrs.items()}
        task_copy = self.task
        obj_copy = copy(self)
        obj_copy.contents = contents_copy
        obj_copy.class_names = class_names_copy
        obj_copy.task = task_copy
        return obj_copy

    def clip_vertices_to_image(self):
        """ Modifies the vertices to be within the image """

        for entry in tqdm(self.contents, desc="Clipping points"):
            entry.clip_vertices_to_image()

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.contents[item]
        raise ValueError(f"Unsupported data type. Expected int, got {type(item)}.")

    def set_class_names(self, class_names: dict):
        self.class_names.update(class_names)

    def redistribute_splits(self, train_split=0.8, val_split=0.2, test_split=0.0) -> None:
        """
        Randomly distributes the entries into new splits, but the probability of an entry being assigned
        to a specific split is weighted on the arguments provided above.
        """
        splits = ["train", "val", "test"]
        weights = [train_split, val_split, test_split]
        for entry in tqdm(self.contents, desc="Splitting"):
            new_split = choices(splits, weights=weights, k=1)[0]
            entry.set_split(new_split)


class DatasetEntry:
    """
    Base class for dataset entry. It contains the image, annotations and more information
    such as the task (e.g. detect or segment).
    """
    # Colors are in RGB format
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
        image = self._image.copy()
        if show_instances:
            h, w = image.shape[:2]
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
                    cv2.line(image, start_point, end_point, darken_color(color), 2)
                fill_translucent_polygon(image, abs_coords, color)

        cv2.imshow("Sample Preview", resize_image_to_fit(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        cv2.waitKey(0)
        try:
            cv2.destroyWindow("Random Sample Preview")
        except cv2.error:
            pass

    def augment(self, h_shear=None, v_shear=None, rotate=None, flip=False, aug_value=0.2, aug_strength=0.2):
        """
        Augment the current entry.
        :param h_shear: Horizontal shear. Can be negative or positive.
        :param v_shear: Vertical shear. Can be negative or positive.
        :param rotate: Angle of rotation.
        :param flip: Set True to flip image
        :return:
        """
        destination_task = self.task
        if self.task != "segment":
            self.convert_task("segment")

        if h_shear is None:
            h_shear = choice([-1, 1]) * uniform(aug_value, aug_value + aug_strength)
        if v_shear is None:
            v_shear = choice([-1, 1]) * uniform(aug_value, aug_value + aug_strength)
        if rotate is None:
            rotate = choice([-1, 1]) * uniform(aug_value, aug_value + aug_strength)
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

        self.convert_task(destination_task)
        self.clip_vertices_to_image()

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
                        raise RuntimeError
                self.task = new_task
            else:
                NotImplementedError(f"Unimplemented conversion from {self.task} to {new_task}")

        elif self.task == "detect":
            if new_task == "segment":
                for i, row in enumerate(self.annotations):
                    class_index = row.pop(0)
                    x, y, w, h = row
                    half_w, half_h = w / 2, h / 2
                    self.annotations[i] = [class_index] + [x - half_w, y - half_h,
                                                           x - half_w, y + half_h,
                                                           x + half_w, y + half_h,
                                                           x + half_w, y - half_h]
                self.task = new_task
            else:
                NotImplementedError(f"Unimplemented conversion from {self.task} to {new_task}")

        else:
            NotImplementedError(f"Unimplemented conversion from {self.task} to {new_task}")

    def copy(self):
        image_duplicate = self._image.copy()
        annotations_duplicate = [[n for n in row] for row in self.annotations]
        split_duplicate = self.split
        task_duplicate = self.task
        obj_duplicate = DatasetEntry(image_duplicate, annotations_duplicate, split_duplicate, task_duplicate)
        return obj_duplicate

    def clip_vertices_to_image(self):
        destination_task = self.task
        self.convert_task("segment")

        i = 0
        while i < len(self.annotations):
            row = self.annotations[i]
            class_index = row[0]
            old_points = row[1:]
            old_points_coords = points_to_coords(old_points)

            new_points_coords = clip_polygon_to_bbox([0.0, 0.0, 1.0, 1.0], old_points_coords)
            # for x, y in new_points_coords:
            #     if x < 0.0 or x > 1.0 or y < 0.0
            if new_points_coords:
                new_points = coords_to_points(new_points_coords)
                self.annotations[i] = [class_index] + new_points
                i += 1
            else:
                self.annotations.pop(i)

        # Return to correct task
        self.convert_task(destination_task)

    def set_split(self, split: Literal["train", "val", "test"]):
        self.split = split


if __name__ == "__main__":
    # dataset = Dataset(_dataset=r"C:\Users\Bulaya\PycharmProjects\DentalDiseasesDetection\model\dental_seg_augmented_2",
    #                   task="segment")
    dataset = Dataset(
        _dataset=r"C:\Users\Bulaya\PycharmProjects\YOLOVisualise\dental_dataset\data.yaml",
        task="segment")
    # dataset = Dataset(
    #     _dataset=r"C:\Users\Bulaya\PycharmProjects\DentalDiseasesDetection\model\dental_seg_augmented_2\data.yaml",
    #     task="segment")

    dataset.shuffle_data()
    dataset.redistribute_splits(train_split=0.8, val_split=0.2)
    # dataset.show_instance_counts()
    for _ in range(100):

        r = choice(dataset)
        while r.annotations:
            r = choice(dataset)
        print(r.split, r.annotations)
        r.show()

    dataset.export("dental_dataset")
