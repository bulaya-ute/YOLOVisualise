import os
import re
import shutil
from typing import Union, Literal
from PIL import Image
import numpy as np
import cv2
from random import random, choices
import collections


def create_directory(directory_path, verbose=False):
    """
    Creates a directory if it doesn't exist.

    Args:
        directory_path (str): The path to the directory to create.
        :param directory_path:
        :param verbose:
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        if verbose:
            print(f"Directory created: {os.path.abspath(directory_path)}")
    else:
        if verbose:
            print(f"Directory already exists: {os.path.abspath(directory_path)}")


def copy_file(source_path, destination_folder):
    source_path = os.path.abspath(source_path)
    destination_folder = os.path.abspath(destination_folder)
    base_name = os.path.basename(source_path)
    new_filename = base_name
    count = 0
    while os.path.exists(os.path.join(destination_folder, new_filename)):
        fn, ext = os.path.splitext(base_name)
        new_filename = f"{fn}({count}){ext}"
        count += 1
    destination_path = os.path.join(destination_folder, new_filename)
    shutil.copy2(source_path, destination_path)
    return new_filename


def add_padding(image, up=0, down=0, left=0, right=0):
    """
    Add transparent padding to the input image.

    Args:
    image (PIL.Image.Image): Input image.
    up (int): Padding pixels to add to the top (upper). Default is 0.
    down (int): Padding pixels to add to the bottom (lower). Default is 0.
    left (int): Padding pixels to add to the left. Default is 0.
    right (int): Padding pixels to add to the right. Default is 0.

    Returns:
    PIL.Image.Image: Image with added transparent padding.
    """
    # Get input image dimensions
    width, height = image.size

    # Calculate new dimensions with padding
    new_width = width + left + right
    new_height = height + up + down

    # Create a new blank image with the new dimensions and fill it with transparent pixels
    padded_image = Image.new('RGBA', (new_width, new_height), color=(0, 0, 0, 0))

    # Paste the original image onto the padded image with the specified offsets
    padded_image.paste(image, (left, up))

    return padded_image


def overlay_images(_background_image: Image, _overlay_image: Image, x=0.0, y=0.0, seg_points=None):
    """
    Places an image on top of the other. Raise error of overlay image is larger
    than background either by width or height.
    :param _background_image: Background image.
    :param _overlay_image: Image to overlay.
    :param x: Horizontal offset ratio. Must be in [0, 1].
    :param y: Vertical offset ratio. Must be in [0, 1].
    :param seg_points: Segmentation points of the overlay image. Now ones will be returned if provided.
    :return: New image, and new segmentation points.
    """
    unedited_overlay = _overlay_image.copy()

    # If background image doesn't have an alpha channel, convert it to RGBA
    if _background_image.mode != 'RGBA':
        _background_image = _background_image.convert('RGBA')

    # If overlay image doesn't have an alpha channel, convert it to RGBA
    if _overlay_image.mode != 'RGBA':
        _overlay_image = _overlay_image.convert('RGBA')

    # Composite the images
    if _overlay_image.width > _background_image.width or _overlay_image.height > _background_image.height:
        raise RuntimeError("Overlay is larger than background")

    x_offset = int(x * (_background_image.width - _overlay_image.width))
    y_offset = int(y * (_background_image.height - _overlay_image.height))
    _overlay_image = add_padding(_overlay_image, left=x_offset, up=y_offset)

    width_pad = max([0, _background_image.width - _overlay_image.width])
    height_pad = max([0, _background_image.height - _overlay_image.height])
    _overlay_image = add_padding(_overlay_image, down=height_pad, right=width_pad)
    result = Image.alpha_composite(_background_image, _overlay_image)

    new_seg_points = []
    if seg_points is not None:
        for x, y in seg_points:
            new_x = (x * unedited_overlay.width + x_offset) / _background_image.width
            new_y = (y * unedited_overlay.height + y_offset) / _background_image.height
            new_seg_points.append((new_x, new_y))

    return result, new_seg_points


def generate_filename(source_path, destination_folder):
    # source_path = os.path.abspath(source_path)
    destination_folder = os.path.abspath(destination_folder)
    base_name = os.path.basename(source_path)
    new_filename = base_name
    count = 0
    while os.path.exists(os.path.join(destination_folder, new_filename)):
        fn, ext = os.path.splitext(base_name)
        new_filename = f"{fn}({count}){ext}"
        count += 1
    # destination_path = os.path.join(destination_folder, new_filename)
    # shutil.copy2(source_path, destination_path)
    return new_filename


def extract_using_polygon(image, points, save_as=None):
    # Create a mask from the polygon
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    points = np.array(points)
    cv2.fillPoly(mask, [points], 255)
    alpha_channel = mask.copy()
    cv2.fillPoly(alpha_channel, [points], 255)

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # rgba_image = cv2.merge(masked_image, alpha_channel)
    rgba_image = np.dstack((masked_image, alpha_channel))

    # Extract the bounding box coordinates
    x, y, w, h = cv2.boundingRect(points)

    # Extract the cropped part using the bounding box coordinates
    cropped_rgba = rgba_image[y:y + h, x:x + w]

    if save_as:
        cv2.imwrite(save_as, cropped_rgba)
    return cropped_rgba


def shear_and_warp(image_arr, shear_horizontal_angle, shear_vertical_angle, vertical_warp, horizontal_warp):
    """
    Distort an image using the specified parameters

    :param image_arr: Image array
    :param shear_horizontal_angle: Angle of horizontal shear. Make it negative for opposite effect.
    :param shear_vertical_angle: Angle of vertical shear. Make it negative for opposite effect.
    :param vertical_warp: Ratio of vertical warp. Must be <= 1. Make it negative for opposite effect.
    :param horizontal_warp: Ratio of horizontal warp. Must be <= 1. Make it negative for opposite effect.
    :return: Modified image.
    """

    # image_arr = np.array(Image.open(image_path).convert("RGBA"))
    image_arr = cv2.cvtColor(image_arr, cv2.COLOR_RGB2RGBA)

    height, width = image_arr.shape[:2]

    # Define destination points based on warping
    mid_y = height // 2
    mid_x = width // 2

    dst_top_left = np.float32([0, 0])
    dst_top_right = np.float32([width, 0])
    dst_bottom_right = np.float32([width, height])
    dst_bottom_left = np.float32([0, height])

    if vertical_warp > 0:
        dst_top_right[1] = mid_y - mid_y * vertical_warp
        dst_bottom_right[1] = mid_y + mid_y * vertical_warp
    else:
        dst_top_left[1] = mid_y - abs(mid_y * vertical_warp)
        dst_bottom_left[1] = mid_y + abs(mid_y * vertical_warp)

    if horizontal_warp > 0:
        dst_top_right[0] = mid_x + mid_x * horizontal_warp
        dst_top_left[0] = mid_x - mid_x * horizontal_warp
    else:
        dst_bottom_right[0] = mid_x + mid_x * abs(horizontal_warp)
        dst_bottom_left[0] = mid_x - mid_x * abs(horizontal_warp)

    # Convert angles to radians
    shear_horizontal_radians = np.radians(shear_horizontal_angle)
    shear_vertical_radians = np.radians(shear_vertical_angle)

    up_pad, down_pad, left_pad, right_pad = 0, 0, 0, 0

    h_pad = abs(int(height * np.tan(shear_horizontal_radians)))
    v_pad = abs(int(width * np.tan(shear_vertical_radians)))

    if shear_vertical_angle > 0:
        up_pad = v_pad
        dst_top_left[1] += v_pad
        dst_bottom_left[1] += v_pad
    else:
        down_pad = v_pad
        dst_top_right[1] += v_pad
        dst_bottom_right[1] += v_pad

    if shear_horizontal_angle > 0:
        left_pad = h_pad
        dst_top_right[0] += h_pad
        dst_top_left[0] += h_pad
    else:
        right_pad = h_pad
        dst_bottom_right[0] += h_pad
        dst_bottom_left[0] += h_pad

    image_pil = add_padding(Image.fromarray(image_arr), up=up_pad, down=down_pad, left=left_pad, right=right_pad)

    # Define source points (corners of the input image)
    src_top_left = [left_pad, up_pad]
    src_top_right = [width + left_pad, up_pad]
    src_bottom_right = [width + left_pad, height + up_pad]
    src_bottom_left = [left_pad, height + up_pad]

    src_pts = np.float32([src_top_left, src_top_right, src_bottom_right, src_bottom_left])

    new_width, new_height = width + h_pad, height + v_pad

    # Define destination points (adjusted for desired warping)
    points = [dst_top_left, dst_top_right, dst_bottom_right, dst_bottom_left]

    # Calculate perspective transform matrix
    transform_matrix = cv2.getPerspectiveTransform(src_pts, np.array([np.float32([x, y]) for x, y in points]))

    # Warp the image
    warped_image_arr = cv2.warpPerspective(np.array(image_pil), transform_matrix, (new_width, new_height))
    warped_image_pil = Image.fromarray(warped_image_arr)
    segmentation_points = [[x / new_width, y / new_height] for x, y in points]
    return np.asarray(warped_image_pil), segmentation_points


def scale_to_fit_window(image, window_width, window_height):
    window_width = int(window_width)
    window_height = int(window_height)

    # Get the dimensions of the input image
    image_height, image_width = image.shape[:2]

    # Calculate the aspect ratio of the input image
    image_aspect_ratio = image_width / image_height

    # Calculate the aspect ratio of the window
    window_aspect_ratio = window_width / window_height

    # If the image aspect ratio is greater than the window aspect ratio,
    # resize the image based on the window width
    if image_aspect_ratio > window_aspect_ratio:
        new_width = window_width
        new_height = int(new_width / image_aspect_ratio)
    # Otherwise, resize the image based on the window height
    else:
        new_height = window_height
        new_width = int(new_height * image_aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image


def scale_image(image, scale_factor):
    # Get the new dimensions based on the scale factor
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)

    # Resize the image using the new dimensions
    scaled_image = cv2.resize(image, (new_width, new_height))

    return scaled_image


def overlay_images(_background_image, _overlay_image, x=0.0, y=0.0, seg_points=None):
    """
    Places an image on top of the other. Raise error of overlay image is larger
    than background either by width or height.
    :param _background_image: Background image.
    :param _overlay_image: Image to overlay.
    :param x: Horizontal offset ratio. Must be in [0, 1].
    :param y: Vertical offset ratio. Must be in [0, 1].
    :param seg_points: Segmentation points of the overlay image. Now ones will be returned if provided.
    :return: New image, and new segmentation points.
    """
    unedited_overlay = _overlay_image.copy()
    _background_image = Image.fromarray(_background_image)
    _overlay_image = Image.fromarray(_overlay_image)

    # If background image doesn't have an alpha channel, convert it to RGBA
    if _background_image.mode != 'RGBA':
        _background_image = _background_image.convert('RGBA')

    # If overlay image doesn't have an alpha channel, convert it to RGBA
    if _overlay_image.mode != 'RGBA':
        _overlay_image = _overlay_image.convert('RGBA')

    # Composite the images
    if _overlay_image.width > _background_image.width or _overlay_image.height > _background_image.height:
        raise RuntimeError("Overlay is larger than background")

    x_offset = int(x * (_background_image.width - _overlay_image.width))
    y_offset = int(y * (_background_image.height - _overlay_image.height))
    _overlay_image = add_padding(_overlay_image, left=x_offset, up=y_offset)

    width_pad = max([0, _background_image.width - _overlay_image.width])
    height_pad = max([0, _background_image.height - _overlay_image.height])
    _overlay_image = add_padding(_overlay_image, down=height_pad, right=width_pad)
    result = Image.alpha_composite(_background_image, _overlay_image)
    result = np.asarray(result)

    new_seg_points = []
    if seg_points is not None:
        for x, y in seg_points:
            new_x = (x * unedited_overlay.width + x_offset) / _background_image.width
            new_y = (y * unedited_overlay.height + y_offset) / _background_image.height
            new_seg_points.append((new_x, new_y))
        return result, new_seg_points

    return result


def generate_unique_filename(current_filename: str, directory: str):
    current_filename = os.path.basename(current_filename)
    new_filename = current_filename
    number = 0
    while new_filename in os.listdir(directory):
        filename, ext = ".".join(new_filename.split(".")[:-1]), "." + new_filename.split(".")[-1]
        if filename.isnumeric():
            new_filename = filename = str(int(filename) + 1) + ext
            continue

        new_filename = f"{filename}_{number}{ext}"
        number += 1
        # raise NotImplementedError
    return new_filename


def change_extension(filename: str, new_ext):
    if new_ext[0] == ".":
        new_ext = new_ext[1:]

    split_stuff = filename.split(".")
    return ".".join(split_stuff[:-1] + [new_ext])


def warp_image_and_points(image: np.ndarray, h_deform: float, v_deform: float, rotate: float,
                          flip: Union[Literal["h", "v", "hv"], None] = None, image_points=None):
    """

    :param image: Image to warp in numpy array format
    :param image_points: Coordinates to be warped with respect to the image
    :param h_deform: Ratio in which to expand either the top or bottom. Positive
        value will expand the bottom, and negative the top.
    :param v_deform: Ratio in which to expand either the left or right. Positive
        value will expand the right, and negative the left.
    :param rotate: Angle of rotation.
    :param flip:
    :return:
    """

    height, width = image.shape[:2]
    src_points = [(0, 0), (width, 0), (0, height), (width, height)]  # tl tr bl br
    dst_points = [list(p) for p in src_points]  # tl tr bl br

    if h_deform > 0:
        dst_points[2][0] -= h_deform * width * 0.5
        dst_points[3][0] += h_deform * width * 0.5
    else:
        dst_points[0][0] -= -h_deform * width * 0.5
        dst_points[1][0] += -h_deform * width * 0.5

    if v_deform < 0:
        dst_points[0][1] -= -v_deform * height * 0.5
        dst_points[2][1] += -v_deform * height * 0.5
    else:
        dst_points[1][1] -= v_deform * height * 0.5
        dst_points[3][1] += v_deform * height * 0.5

    dst_points = rotate_points(dst_points, rotate, (width // 2, height // 2))

    if flip:
        if "h" in flip.lower():
            center_x = width // 2
            for i, (x, y) in enumerate(dst_points):
                distance = abs(center_x - x)
                if x < center_x:
                    dst_points[i][0] += 2 * distance
                else:
                    dst_points[i][0] -= 2 * distance

        if "v" in flip.lower():
            center_y = height // 2
            for i, (x, y) in enumerate(dst_points):
                distance = abs(center_y - y)
                if y < center_y:
                    dst_points[i][1] += 2 * distance
                else:
                    dst_points[i][1] -= 2 * distance

    # Compute the perspective transformation matrix
    dst_points = np.float32([np.float32(p) for p in dst_points])
    matrix = cv2.getPerspectiveTransform(np.float32(src_points), dst_points)

    # Warp the image
    warped_image = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))

    def transform_coords(coords: list):
        # print("in function", coords)
        if type(coords[0][0]) in (int, float):
            _transformed_coords = []
            for point in coords:
                # Convert to homogeneous coordinates
                homogeneous_point = np.array([point[0], point[1], 1]).reshape((3, 1))
                # Apply the transformation matrix
                transformed_point = np.dot(matrix, homogeneous_point)
                # Convert back to Cartesian coordinates
                _transformed_coords.append(
                    (transformed_point[0] / transformed_point[2], transformed_point[1] / transformed_point[2]))
            return [(int(_x[0]), int(_y[0])) for _x, _y in _transformed_coords]

        else:
            return [transform_coords(c) for c in coords]

    # Transform the coordinates
    if image_points:
        # print(image_points, "----")
        transformed_coords = transform_coords(image_points)
        # print(transformed_coords)
        return warped_image, transformed_coords

    return warped_image


def rotate_points(points, angle, pivot):
    """
    Rotate multiple points around a given pivot point by a certain angle.

    Args:
    points (list of tuples): A list of (x, y) coordinates to rotate.
    angle (float): The rotation angle in degrees.
    pivot (tuple): The (x, y) coordinates of the pivot point.

    Returns:
    list of tuples: A list of (x, y) coordinates of the rotated points.
    """

    def rotate_point(point, angle, pivot):
        """
        Rotate a point around a given pivot point by a certain angle.

        Args:
        point (tuple): The (x, y) coordinates of the point to rotate.
        angle (float): The rotation angle in degrees.
        pivot (tuple): The (x, y) coordinates of the pivot point.

        Returns:
        tuple: The (x, y) coordinates of the rotated point.
        """
        angle_rad = np.radians(angle)  # Convert angle to radians

        # Translate point to origin
        translated_x = point[0] - pivot[0]
        translated_y = point[1] - pivot[1]

        # Apply rotation
        rotated_x = translated_x * np.cos(angle_rad) - translated_y * np.sin(angle_rad)
        rotated_y = translated_x * np.sin(angle_rad) + translated_y * np.cos(angle_rad)

        # Translate point back
        final_x = rotated_x + pivot[0]
        final_y = rotated_y + pivot[1]

        return [final_x, final_y]

    return [rotate_point(point, angle, pivot) for point in points]


def probability(prob: float) -> bool:
    """
    Return True with the given probability, else False.

    :param prob: A float between 0 and 1 representing the probability of returning True.
    :return: True with the given probability, else False.
    """
    if not 0 <= prob <= 1:
        raise ValueError("Probability must be a float between 0 and 1")

    return random() < prob


def weighted_random_selection(values: list) -> int:
    """
    Randomly select a number from the list, where the numbers represent the weights/probabilities of being selected.

    :param values: List of non-negative numbers representing weights.
    :return: Selected number based on the weights.
    """
    total_weight = sum(values)
    if total_weight == 0:
        raise ValueError("At least one weight must be greater than 0")

    selected_value = choices(values, weights=values, k=1)[0]
    return selected_value


def points_to_coords(points: list):
    """
    Convert a list of [x1, y1, x2, y2, ...] points to [(x1, y1), (x2, y2), ...] coordinates
    :param points:
    :return:
    """
    if isinstance(points[0], (int, float)):
        if len(points) % 2 != 0:
            raise ValueError(f"An even number of points must be supplied. Got {points}")

        return list(zip(points[0::2], points[1::2]))
    return [points_to_coords(pl) for pl in points]


def coords_to_points(coords: list):
    if type(coords[0][0]) in (int, float):
        points_list = []
        for _x, _y in coords:
            points_list += [_x, _y]
        return points_list

    return [coords_to_points(line) for line in coords]


def multiply_list(input_list, number):
    for i in range(len(input_list)):
        if type(input_list[i]) in (int, float):
            input_list[i] *= number
        else:
            multiply_list(input_list[i], number)
    return input_list


def fill_translucent_polygon(image, points, color=(128, 128, 128), alpha=0.5):
    """
    Fill a translucent polygon on the given image.

    :param image: The original image.
    :param points: The points of the polygon (list of tuples).
    :param color: The color of the polygon (BGR tuple).
    :param alpha: The transparency factor (0.0 to 1.0).
    """
    # Convert points to a numpy array
    points = np.array(points, dtype=np.int32)

    # Create an overlay image
    overlay = image.copy()

    # Draw the filled polygon on the overlay image
    cv2.fillPoly(overlay, [points], color)

    # Blend the overlay with the original image
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)


def relative_to_absolute_coords(rel_coords, width, height):
    return [[x * width, y * height] for x, y in rel_coords]


def absolute_to_relative_coords(abs_coords: list, width, height):
    if type(abs_coords[0][0]) in (int, float):
        return [[x / width, y / height] for x, y in abs_coords]
    return [absolute_to_relative_coords(line, width, height) for line in abs_coords]


def get_yolo_bbox_from_polygon(polygon_vertices):
    """
    Calculates the bounding box coordinates for a polygon.

    Args:
        polygon_vertices: A list of lists, where each inner list represents the
            (x, y) coordinates of a vertex in the polygon.

    Returns:
        A list containing the (xmin, ymin, xmax, ymax) coordinates of the bounding box.
    """

    xmin = float('inf')
    ymin = float('inf')
    xmax = float('-inf')
    ymax = float('-inf')

    for vertex in polygon_vertices:
        x, y = vertex
        xmin = min(xmin, x)
        ymin = min(ymin, y)
        xmax = max(xmax, x)
        ymax = max(ymax, y)

    # return [xmin, ymin, xmax, ymax]
    return [(xmin + xmax) / 2, (ymin + ymax) / 2, (xmax - xmin), (ymax - ymin)]


def increment_filename(filename):
    """
    Increment the filename by 1. If the filename already contains a number, increment that number.
    Otherwise, append a number to the filename.

    :param filename: The original filename.
    :return: The incremented filename.
    """
    base, ext = os.path.splitext(filename)

    # Match if the filename ends with a number
    match = re.search(r"_(\d+)$", base)

    if match:
        # Increment the number found
        number = int(match.group(1)) + 1
        new_base = re.sub(r"_(\d+)$", f"_{number}", base)
    else:
        # If no number found, append (2) to the base name
        new_base = f"{base}_2"

    # Combine the new base name with the original extension
    incremented_filename = f"{new_base}{ext}"
    return incremented_filename


def get_poly_from_yolo_bbox(bbox):
    x, y, w, h = bbox
    ww, hh = w * 0.5, h * 0.5
    return [(x - ww, y - hh), (x + ww, y - hh), (x - ww, y + hh), (x + ww, y + hh)]


def darken_color(bgr_color, darkening_factor=0.1):
    """
    Darkens a BGR color by a certain factor.

    Args:
        bgr_color: A tuple representing a BGR color (Blue, Green, Red).
        darkening_factor: A float between 0.0 and 1.0 representing the amount to darken (default=0.1).

    Returns:
        A tuple representing the darkened BGR color.
    """

    new_color = []
    for channel in bgr_color:
        # Reduce each color channel value by the darkening factor
        darkened_channel = int(channel * (1 - darkening_factor))
        # Clamp the value to be within 0-255 (BGR color range)
        new_color.append(max(0, min(darkened_channel, 255)))

    return tuple(new_color)


def calculate_rarity(sublists):
    """
    Calculate the rarity of each integer in the sublists.

    :param sublists: List of sublists containing integers.
    :return: Dictionary with integers as keys and their rarity as values.
    """
    # Count the frequency of each integer
    frequency = collections.Counter(x for sublist in sublists for x in sublist)

    # Calculate the rarity (inverse of frequency)
    total_count = sum(frequency.values())
    rarity = {k: total_count / v for k, v in frequency.items()}

    return rarity


def assign_weights(sublists, rarity):
    """
    Assign weights to each sublist based on the rarity of the integers it contains.

    :param sublists: List of sublists containing integers.
    :param rarity: Dictionary with integers as keys and their rarity as values.
    :return: List of weights corresponding to each sublist.
    """
    weights = []
    for sublist in sublists:
        # Calculate weight as the sum of rarities divided by the number of integers in the sublist
        # This gives priority to sublists with rare integers and fewer common integers
        if sublist:  # Ensure sublist is not empty
            weight = sum(rarity[x] for x in sublist) / len(sublist)
        else:
            weight = 0
        weights.append(weight)
    return weights


def select_from_weighted_sublists(sublists):
    """
    Append a fixed number of integer-containing sublists to the super list.
    Each appended item is a copy of a randomly selected existing item, selected with weighted probability.

    :param sublists: List of sublists containing integers.
    :param num_append: Number of sublists to append.
    :return: The modified list with appended sublists.
    """
    # Append sublists based on weighted random selection
    # for _ in range(num_append):

    # Calculate rarity of integers
    rarity = calculate_rarity(sublists)

    # Assign weights to sublists
    weights = assign_weights(sublists, rarity)

    # Normalize weights to ensure they sum up to 1
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    selected_sublist_index = choices([i for i in range(len(sublists))], weights=normalized_weights, k=1)[0]
    # sublists.append(list(selected_sublist))  # append a copy of the selected sublist

    return selected_sublist_index


def clip_polygon_to_bbox(bbox, polygon):
    """
    Clip a polygon to the given bounding box.

    :param bbox: Bounding box coordinates (xmin, ymin, xmax, ymax).
    :param polygon: List of polygon vertices [(x1, y1), (x2, y2), ...].
    :return: New polygon vertices after clipping, or None if less than 3 vertices remain.
    """

    def inside(p, _edge):
        if _edge == 'left':
            return p[0] >= xmin
        elif _edge == 'right':
            return p[0] <= xmax
        elif _edge == 'bottom':
            return p[1] >= ymin
        elif _edge == 'top':
            return p[1] <= ymax

    def intersect(p1, p2, _edge):
        if _edge == 'left':
            x, y = xmin, p1[1] + (xmin - p1[0]) * (p2[1] - p1[1]) / (p2[0] - p1[0])
        elif _edge == 'right':
            x, y = xmax, p1[1] + (xmax - p1[0]) * (p2[1] - p1[1]) / (p2[0] - p1[0])
        elif _edge == 'bottom':
            x, y = p1[0] + (ymin - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]), ymin
        elif _edge == 'top':
            x, y = p1[0] + (ymax - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]), ymax
        else:
            raise ValueError("Unknown edge")
        return [x, y]

    def clip_polygon(_polygon, _edge):
        _clipped_polygon = []
        for i in range(len(_polygon)):
            p1 = _polygon[i - 1]
            p2 = _polygon[i]
            if inside(p2, _edge):
                if not inside(p1, _edge):
                    _clipped_polygon.append(intersect(p1, p2, _edge))
                _clipped_polygon.append(p2)
            elif inside(p1, _edge):
                _clipped_polygon.append(intersect(p1, p2, _edge))
        return _clipped_polygon

    xmin, ymin, xmax, ymax = bbox
    edges = ['left', 'right', 'bottom', 'top']

    clipped_polygon = polygon
    for edge in edges:
        clipped_polygon = clip_polygon(clipped_polygon, edge)
        if len(clipped_polygon) < 3:
            return None

    return clipped_polygon


def clip_polygon_to_bbox(bbox, polygon):
    """
    Clip a polygon to the given bounding box.

    :param bbox: Bounding box coordinates (xmin, ymin, xmax, ymax).
    :param polygon: List of polygon vertices [(x1, y1), (x2, y2), ...].
    :return: New polygon vertices after clipping, or None if less than 3 vertices remain.
    """

    # print(bbox)
    # print(polygon)
    def inside(p, _edge):
        if _edge == 'left':
            return p[0] >= x_min
        elif _edge == 'right':
            return p[0] <= x_max
        elif _edge == 'bottom':
            return p[1] >= y_min
        elif _edge == 'top':
            return p[1] <= y_max

    def intersect(p1, p2, _edge):
        if _edge == 'left':
            x, y = x_min, p1[1] + (x_min - p1[0]) * (p2[1] - p1[1]) / (p2[0] - p1[0])
        elif _edge == 'right':
            x, y = x_max, p1[1] + (x_max - p1[0]) * (p2[1] - p1[1]) / (p2[0] - p1[0])
        elif _edge == 'bottom':
            x, y = p1[0] + (y_min - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]), y_min
        elif _edge == 'top':
            x, y = p1[0] + (y_max - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]), y_max
        else:
            raise RuntimeError(f"Unknown argument: '{_edge}'")
        return [x, y]

    def clip_polygon(_poly, _edge):
        _clipped_polygon = []
        for i in range(len(_poly)):
            p1 = _poly[i - 1]
            p2 = _poly[i]
            if inside(p2, _edge):
                if not inside(p1, _edge):
                    _clipped_polygon.append(intersect(p1, p2, _edge))
                _clipped_polygon.append(p2)
            elif inside(p1, _edge):
                _clipped_polygon.append(intersect(p1, p2, _edge))
        return _clipped_polygon

    # Handles when a superlist of polygons is supplied instead
    if type(polygon[0][0]) in (list, tuple):
        return [clip_polygon_to_bbox(bbox, p) for p in polygon]

    x_min, y_min, x_max, y_max = bbox
    edges = ['left', 'right', 'bottom', 'top']

    if bbox[0] != 0.0 or bbox[1] != 0.0 or bbox[2] != 1.0 or bbox[3] != 1.0:
        raise Exception("Something here")

    # Check if it sticks out of the bbox
    for x, y in polygon:
        if x < x_min or x > x_max or y < y_min or y > y_max:
            break  # Polygon sticks out
    else:
        return polygon  # Polygon is already within the bbox


    clipped_polygon = polygon
    for edge in edges:
        clipped_polygon = clip_polygon(clipped_polygon, edge)
        if len(clipped_polygon) < 3:
            return None
    # print()
    for i, (x, y) in enumerate(clipped_polygon):
        if x < x_min or x > x_max or y < y_min or y > y_max:
            print("clipping", x, y)
            x, y = clipped_polygon[i]
            clipped_polygon[i] = [min([0.0, max(1.0, x)]), min([0.0, max(1.0, y)])]
            raise RuntimeError("Fix this bug!!!!")
    return clipped_polygon


def clip_bbox_to_bbox(_bbox, bbox_to_clip):
    bbox_poly = get_poly_from_yolo_bbox(bbox_to_clip)
    clipped_poly = clip_polygon_to_bbox(_bbox, bbox_poly)
    if clipped_poly:
        return get_yolo_bbox_from_polygon(clipped_poly)
    return None


def resize_image_to_fit(image: np.ndarray, target_width=640, target_height=480) -> np.ndarray:
    """
    Resizes an image to fit within the specified width and height while maintaining its aspect ratio.

    :param image: A NumPy array representing the image.
    :param target_width: The target width to fit the image within.
    :param target_height: The target height to fit the image within.
    :return: A new NumPy array representing the resized image.
    """
    # Get the original dimensions of the image
    original_height, original_width = image.shape[:2]

    # Calculate the aspect ratios
    aspect_ratio = original_width / original_height
    target_aspect_ratio = target_width / target_height

    # Determine the new dimensions
    if aspect_ratio > target_aspect_ratio:
        # Image is wider than target aspect ratio, fit to width
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        # Image is taller than target aspect ratio, fit to height
        new_width = int(target_height * aspect_ratio)
        new_height = target_height

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image


if __name__ == "__main__":
    bbox = [0, 0, 1, 1]
    polygon = [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]
    print(clip_polygon_to_bbox(bbox, polygon))
