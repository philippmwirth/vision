r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.

On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3

Also, if you train Keypoint R-CNN, the default hyperparameters are
    --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3
Because the number of images is smaller in the person keypoint subset of COCO,
the number of epochs should be adapted so that we have the same number of iterations.
"""
import datetime
import os
import time
import json

import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn


from coco_utils import get_coco, get_coco_kp

from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from engine import train_one_epoch, evaluate

import utils

import lightly
from lightly.api.api_workflow_client import ApiWorkflowClient
from lightly.active_learning.agents.agent import ActiveLearningAgent
from lightly.active_learning.config import SamplerConfig
from lightly.openapi_generated.swagger_client import SamplingMethod
from lightly.active_learning.scorers import ScorerObjectDetection


from benchmark_utils.plogger import ALEpisodeLog, ALBenchmarkPlogger
from benchmark_utils.voc_detection.dataset import get_VOCDetection_dataset, get_transform, index_to_filename, filename_to_index
from benchmark_utils.voc_detection.metrics import get_metrics
from benchmark_utils.boxes.predict import predict_on_filenames


def main(args, sampling_method: SamplingMethod):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")

    # get datasets (1 for training, 1 for testing, 1 for api)
    dataset, num_classes = get_VOCDetection_dataset("train", get_transform(train=True), args.data_path)
    dataset_test, _ = get_VOCDetection_dataset("val", get_transform(train=False), args.data_path)

    # get a lightly dataset (for api)
    lightly_dataset, _ = get_VOCDetection_dataset('train', None, args.data_path)
    lightly_dataset = lightly.data.LightlyDataset.from_torch_dataset(lightly_dataset, index_to_filename=index_to_filename)

    # upload training dataset (without transforms)
    client = ApiWorkflowClient(token=args.token)
    try:
        client.set_dataset_id_by_name(args.dataset_name)
        print('Dataset already exists, skipping upload.')
    except ValueError:
        client.create_new_dataset_with_unique_name(
            dataset_basename=args.dataset_name
        )
        client.upload_dataset(lightly_dataset, mode='meta')

    # upload embeddings
    try:
        client.set_embedding_id_by_name(args.embedding_name)
        print('Embedding already exists, skipping upload.')
    except ValueError:
        client.upload_embeddings(args.embedding_path, args.embedding_name)

    # initialize active learning agent and scorer
    agent = ActiveLearningAgent(client)
    scorer = None # will be initialized lazily if necessary

    # initialize episode log
    task_config = {'model': args.model, 'epochs': args.epochs}
    sampler_config = {'name': sampling_method}
    log = ALEpisodeLog(task_config=task_config, sampler_config=sampler_config)

    # iterate over how many samples are in the training datasets
    for n_samples in args.steps:
        
        # sample new training set
        if scorer is None:
            # first iteration (use random for all but coreset)
            method = sampling_method if sampling_method == SamplingMethod.CORESET else SamplingMethod.RANDOM
            config = SamplerConfig(n_samples=n_samples, method=method)
            labeled_set = agent.query(config)
        else:
            # following iterations
            config = SamplerConfig(n_samples=n_samples, method=sampling_method)
            labeled_set = agent.query(config, scorer)

        # create subset of training set based on selected samples
        labeled_indices = [filename_to_index(i) for i in labeled_set]
        dataset_train = torch.utils.data.Subset(dataset, labeled_indices)

        print("Creating data loaders")
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset_train)
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)

        if args.aspect_ratio_group_factor >= 0:
            group_ids = create_aspect_ratio_groups(dataset_train, k=args.aspect_ratio_group_factor)
            train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
        else:
            train_batch_sampler = torch.utils.data.BatchSampler(
                train_sampler, args.batch_size, drop_last=True)

        data_loader = torch.utils.data.DataLoader(
            dataset_train, batch_sampler=train_batch_sampler, num_workers=args.workers,
            collate_fn=utils.collate_fn)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1,
            sampler=test_sampler, num_workers=args.workers,
            collate_fn=utils.collate_fn)

        print("Creating model")
        kwargs = {
            "trainable_backbone_layers": args.trainable_backbone_layers
        }
        if "rcnn" in args.model:
            if args.rpn_score_thresh is not None:
                kwargs["rpn_score_thresh"] = args.rpn_score_thresh
        model = torchvision.models.detection.__dict__[args.model](num_classes=num_classes, pretrained=args.pretrained,
                                                                **kwargs)
        model.to(device)

        model_without_ddp = model
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

        if args.test_only:
            evaluate(model, data_loader_test, device=device)
            return

        print(f"Start training with dataloader of length {len(data_loader)}")
        start_time = time.time()
        model.train()
        for epoch in range(args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq)
            lr_scheduler.step()
            if args.output_dir:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'args': args,
                    'epoch': epoch},
                    os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

        # evaluate after every iteration
        evaluator = evaluate(model, data_loader_test, device=device)
        metrics = get_metrics(evaluator, n_samples)
        log.save_metrics(metrics=metrics)

        if sampling_method in [SamplingMethod.CORAL, SamplingMethod.ACTIVE_LEARNING]:

            # make prediction on unlabeled images
            model_output = predict_on_filenames(model, agent, lightly_dataset, device)
            
            # create scorer
            scorer = ScorerObjectDetection(model_output)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    return log


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)


    # benchmarmking parameters
    parser.add_argument('--steps', required=True, metavar='N', type=int, nargs='+')
    parser.add_argument('--token', required=True)
    parser.add_argument('--dataset_name', type=str, default='VOCDetection')
    parser.add_argument('--embedding_name', type=str, default='VOCEmbedding')
    parser.add_argument('--embedding_path', type=str, default='embeddings/VOCDetection/embeddings.csv')
    parser.add_argument('--log_json', type=str, default='outputs/VOCDetection.json')

    # torchvision reference args
    parser.add_argument('--data-path', default='datasets/', help='dataset')
    parser.add_argument('--dataset', default='coco', help='dataset')
    #parser.add_argument('--model', default='maskrcnn_resnet50_fpn', help='model')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model') # use this so we don't have to worry about segmentation
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--epochs', default=26, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--lr', default=0.02, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                        'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument('--rpn-score-thresh', default=None, type=float, help='rpn score threshold for faster-rcnn')
    parser.add_argument('--trainable-backbone-layers', default=None, type=int,
                        help='number of trainable layers of backbone')
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    
    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    # benchmark
    benchmark_plogger = ALBenchmarkPlogger(filename=args.log_json)
    sampling_methods = [
        SamplingMethod.CORESET,
        SamplingMethod.CORAL,
        SamplingMethod.ACTIVE_LEARNING,
        SamplingMethod.RANDOM,
    ]
    for sampling_method in sampling_methods:
        for _ in range(3):
            log = main(args, sampling_method)
            benchmark_plogger.append_al_episode_logs_to_file([log])
