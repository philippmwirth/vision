
import torch
import torchvision
import lightly

from lightly.data.lighty_subset import LightlySubset
from lightly.active_learning.utils.bounding_box import BoundingBox
from lightly.active_learning.utils.object_detection_output import ObjectDetectionOutput

import utils
from benchmark_utils.voc_detection.dataset import get_transform


def predict_on_filenames(model, agent, lightly_dataset, device):
    """

    """

    # make predictions on the unlabeled set
    dataset_unlabeled = LightlySubset(
        lightly_dataset,
        agent.unlabeled_set
    )
    # hack: set transforms for evaluation and then unset them again
    dataset_unlabeled.base_dataset.dataset.transforms = get_transform(train=False)
    data_loader_unlabeled = torch.utils.data.DataLoader(
        dataset_unlabeled,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=utils.collate_fn
    )

    model.eval()
    model_output = []
    with torch.no_grad():
        for i, (image, target, filename) in enumerate(data_loader_unlabeled):
            
            # make sure the predictions are in the correct order
            assert filename[0] == agent.unlabeled_set[i]

            # move image to device
            image = [image[0].to(device)]

            # get prediction from model
            prediction = model(image)[0]

            # nms (necessary?)
            keep = torchvision.ops.nms(
                prediction['boxes'],
                prediction['scores'],
                iou_threshold=0.5,
            )
            boxes = prediction['boxes'][keep]
            labels = prediction['labels'][keep]
            scores = prediction['scores'][keep]

            # TODO: comment
            # TODO: get actual bounding box shape
            _, height, width = image[0].shape
            box_objects = []
            for box in boxes:
                x0, x1 = box[0], box[2]
                y0, y1 = box[1], box[3]
                box_objects.append(BoundingBox(0., 0., 0., 0.))

            # put everything into object detection output
            output = ObjectDetectionOutput.from_scores(
                box_objects,
                [float(s) for s in scores.cpu().numpy()],
                [int(l) for l in labels.cpu().numpy()],
            )

            # extend list
            model_output.append(output)
    
    dataset_unlabeled.base_dataset.dataset.transforms = None
    return model_output