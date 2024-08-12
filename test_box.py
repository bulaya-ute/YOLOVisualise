import os
import yaml


def load_image_label_pairs(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)

    def get_image_label_pairs(split):
        image_dir = data[split]
        # Derive the label directory by navigating up one level and then into the 'labels' folder
        label_dir = os.path.join(os.path.dirname(image_dir), 'labels', os.path.basename(image_dir))
        image_label_pairs = []

        for image_name in os.listdir(image_dir):
            if image_name.endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(image_dir, image_name)
                label_name = os.path.splitext(image_name)[0] + '.txt'
                label_path = os.path.join(label_dir, label_name)

                if os.path.exists(label_path):
                    image_label_pairs.append((image_path, label_path, split))
                else:
                    image_label_pairs.append((image_path, None, split))

        return image_label_pairs

    # Combine pairs from both train and val splits
    all_pairs = get_image_label_pairs('train') + get_image_label_pairs('val')
    return all_pairs


pairs = load_image_label_pairs(
    r'C:\Users\Bulaya\PycharmProjects\DentalDiseasesDetection\model\dental_seg_augmented_2\data.yaml')

[print(k) for k in pairs]
