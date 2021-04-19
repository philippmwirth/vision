


def get_metrics(evaluator, n_samples):
    """

    """

    metrics = {}
    metrics['no_labelled_samples'] = n_samples

    # average precision
    metrics['AP IoU=0.50:0.95'] = evaluator.coco_eval['bbox'].stats[0]
    metrics['AP IoU=0.50'] = evaluator.coco_eval['bbox'].stats[1]
    metrics['AP IoU=0.75'] = evaluator.coco_eval['bbox'].stats[2]
    metrics['AP IoU=0.50:0.95 small'] = evaluator.coco_eval['bbox'].stats[3]
    metrics['AP IoU=0.50:0.95 medium'] = evaluator.coco_eval['bbox'].stats[4]
    metrics['AP IoU=0.50:0.95 large'] = evaluator.coco_eval['bbox'].stats[5]

    # average recall
    metrics['AR IoU=0.50:0.95'] = evaluator.coco_eval['bbox'].stats[6]
    metrics['AR IoU=0.50:0.95'] = evaluator.coco_eval['bbox'].stats[7]
    metrics['AR IoU=0.50:0.95'] = evaluator.coco_eval['bbox'].stats[8]
    metrics['AR IoU=0.50:0.95 small'] = evaluator.coco_eval['bbox'].stats[9]
    metrics['AR IoU=0.50:0.95 medium'] = evaluator.coco_eval['bbox'].stats[10]
    metrics['AR IoU=0.50:0.95 large'] = evaluator.coco_eval['bbox'].stats[11]

    return metrics