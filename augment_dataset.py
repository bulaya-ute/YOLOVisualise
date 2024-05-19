import os
from math import ceil
from random import uniform, choice

import cv2
from tqdm import tqdm

from utils import change_extension, warp_image_and_points, points_to_coords, fill_translucent_polygon, \
    relative_to_absolute_coords, coords_to_points, absolute_to_relative_coords, generate_unique_filename, \
    create_directory, select_from_weighted_sublists

dataset_dir = r"C:\Users\Bulaya\PycharmProjects\DentalDiseasesDetection\datasets\dental_segmentation_yolo2"
augmentation_split = "train"
augmentation_ratio = 2

images_path = os.path.join(dataset_dir, "images", augmentation_split)
labels_path = os.path.join(dataset_dir, "labels", augmentation_split)

dataset_contents = []
instance_numbers = {}


class ImageData:
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

    def __init__(self, _image_path: str, _label_path: str, split="train"):
        self.split = split
        self._image_path = _image_path
        self._label_path = _label_path
        self.priority = 1
        self.other_attrs = {}
        _img = cv2.imread(_image_path)
        self._image = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        with open(_label_path, "r") as _file_obj:
            _rows = _file_obj.readlines()
            self.annotations = []
            for _row in _rows:
                _row_data = [eval(d) for d in _row.split()]
                _class_index = _row_data[0]
                if _class_index not in instance_numbers:
                    instance_numbers[_class_index] = 1
                else:
                    instance_numbers[_class_index] += 1
                self.annotations.append(_row_data)

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
            for instance_details in self.annotations:
                h, w = self._image.shape[:2]
                _class_index = instance_details.pop(0)
                color = self.colors[_class_index % len(self.colors)]
                abs_coords = relative_to_absolute_coords(points_to_coords(instance_details), w, h)
                fill_translucent_polygon(self._image, abs_coords, color)

        cv2.imshow(self.image_path, cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
        try:
            cv2.destroyWindow(self.image_path)
        except cv2.error:
            pass

    @property
    def image_path(self):
        return self._image_path

    @property
    def label_path(self):
        return self._label_path

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


for image_path in tqdm([os.path.join(images_path, f) for f in os.listdir(images_path)], desc="Scanning dataset"):
    # Check if this image has a label
    label_path = f"{labels_path}/" + (os.path.basename(change_extension(image_path, "txt")))
    if os.path.exists(label_path):
        with open(label_path, "r") as file_obj:
            rows = file_obj.readlines()
            label_data = []
            for row in rows:
                data = [eval(d) for d in row.split()]
                class_index = data[0]
                if class_index not in instance_numbers:
                    instance_numbers[class_index] = 1
                else:
                    instance_numbers[class_index] += 1
                label_data.append(data)
        if rows:
            dataset_contents.append(ImageData(image_path, label_path))
    else:
        pass


for i in tqdm(range(ceil(augmentation_ratio * len(dataset_contents))), desc="Augmenting"):
    class_frequencies = [c.classes_with_duplicates for c in dataset_contents]
    selected_index = select_from_weighted_sublists(class_frequencies)
    selected_image_data = dataset_contents[selected_index]

    original_image = selected_image_data.get_image()
    original_points = selected_image_data.get_points()

    augmented_image_data = ImageData(selected_image_data.image_path, selected_image_data.label_path)
    augmented_image_data.augment()
    dataset_contents.append(augmented_image_data)


for entry in tqdm(dataset_contents, desc="Saving"):
    entry.save("../datasets/dental_seg_yolo_augmented", create_dir=True)
