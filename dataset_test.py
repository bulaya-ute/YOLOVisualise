from yolo_dataset import Dataset

dataset = Dataset(r"C:\Users\Bulaya\PycharmProjects\DentalDiseasesDetection\datasets\dental_seg_augmented_2")
for e in dataset:
    for row in e.annotations:
        for n in row[1:]:
            if n < 0 or n > 1:
                print(row.index(n), n, row)
                break
        else:
            continue
        break
    else:
        continue
    e.show()
