import torch
import torchvision

import transforms as T

def index_to_filename(dataset, index: int):
    return f'image_{index}.png'


def filename_to_index(filename: str):
    suffix = filename[6:]
    return int(suffix[:-4])


class PrepareInstanceKITTI(object):
    """

    """

    CLASSES = (
        'Car',
        'Van',
        'Truck',
        'Pedestrian',
        'Person_sitting',
        'Cyclist',
        'Tram',
        'Misc',
        'DontCare'
    )

    def __init__(self, train: bool):
        self.train = train

        trans = [T.ToTensor()]
        if train:
            trans.append(T.RandomHorizontalFlip(0.5))
        
        self.transforms = T.Compose(trans)

    def __call__(self, image, target):
        boxes = []
        classes = []
        area = []
        iscrowd = []

        objects = target
        if not isinstance(objects, list):
            objects = [objects]
        for obj in objects:
            bbox = obj['bbox']
            boxes.append(bbox)
            classes.append(self.CLASSES.index(obj['type']))
            iscrowd.append(0)
            area.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        classes = torch.as_tensor(classes)
        area = torch.as_tensor(area)
        iscrowd = torch.as_tensor(iscrowd)

        image_id = 0
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


def get_dataset(image_set, transform, data_path):
    """

    """
    dataset = torchvision.datasets.Kitti(
        data_path,
        train=True,
        download=True,
        transforms=transform
    )

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train, test = torch.utils.data.random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(train, test)
    if image_set == 'train':
        return train, 9
    else:
        return test, 9


def get_transform(train):
    """

    """
    return PrepareInstanceKITTI(train)
