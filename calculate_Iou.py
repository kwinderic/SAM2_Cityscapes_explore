import json
from collections import defaultdict

def calculate_average_iou(input_path, output_path):
    with open(input_path, 'r') as file:
        data = json.load(file)
        print(len(data))

    iou_dict = defaultdict(list)
    for obj in data:
        object_id = obj.get('object_id', {})
        label = object_id.get('label')
        iou = obj.get('iou')
        if label is not None and iou is not None:
            iou_dict[label].append(iou)

    with open(output_path, 'w') as file:
        for label, ious in iou_dict.items():
            average_iou = sum(ious) / len(ious) if ious else 0
            file.write(f'Label {label}: Average IoU = {average_iou:.4f}\n')

if __name__ == "__main__":
    # input_file = './cityscapes_dataset/results_b+.json'
    # output_file = './cityscapes_dataset/average_iou_b+.txt'
    input_file = './cityscapes_dataset/results_l.json'
    output_file = './cityscapes_dataset/average_iou_l.txt'
    calculate_average_iou(input_file, output_file)
    input_file = './cityscapes_dataset/results_s.json'
    output_file = './cityscapes_dataset/average_iou_s.txt'
    calculate_average_iou(input_file, output_file)
    input_file = './cityscapes_dataset/results_t.json'
    output_file = './cityscapes_dataset/average_iou_t.txt'
    calculate_average_iou(input_file, output_file)
