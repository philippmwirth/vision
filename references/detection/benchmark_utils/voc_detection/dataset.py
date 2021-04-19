import torch
import torchvision

import transforms as T

def index_to_filename(dataset, index: int):
    return f'image_{index}.png'


def filename_to_index(filename: str):
    suffix = filename[6:]
    return int(suffix[:-4])


class PrepareInstanceVOC(object):
    """

    """

    CLASSES = (
        "__background__ ",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )

    def __init__(self, train: bool):
        self.train = train

        trans = [T.ToTensor()]
        if train:
            trans.append(T.RandomHorizontalFlip(0.5))
        
        self.transforms = T.Compose(trans)

    def __call__(self, image, target):
        anno = target['annotation']
        h, w = anno['size']['height'], anno['size']['width']
        boxes = []
        classes = []
        area = []
        iscrowd = []
        objects = anno['object']
        if not isinstance(objects, list):
            objects = [objects]
        for obj in objects:
            bbox = obj['bndbox']
            bbox = [int(bbox[n]) - 1 for n in ['xmin', 'ymin', 'xmax', 'ymax']]
            boxes.append(bbox)
            classes.append(self.CLASSES.index(obj['name']))
            iscrowd.append(int(obj['difficult']))
            area.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        classes = torch.as_tensor(classes)
        area = torch.as_tensor(area)
        iscrowd = torch.as_tensor(iscrowd)

        image_id = anno['filename'][5:-4]
        image_id = torch.as_tensor([int(image_id)])

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        target["area"] = area
        target["iscrowd"] = iscrowd

        image, target = self.transforms(image, target)

        target["boxes"].clamp_(min=0)

        return image, target


def get_VOCDetection_dataset(image_set, transform, data_path):
    """

    """
    dataset = torchvision.datasets.VOCDetection(
        data_path,
        download=False,
        image_set=image_set,
        transforms=transform
    )
    return dataset, 21


def get_transform(train):
    """

    """
    return PrepareInstanceVOC(train)