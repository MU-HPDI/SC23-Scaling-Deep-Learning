# the mmdetection repo has been copied to /workspace/mmdetection so we can use
# their configs as a base
_base_ = "/workspace/mmdetection/configs/ssd/ssd512_coco.py"

# pretrained weights
load_from = "https://download.openmmlab.com/mmdetection/v2.0/ssd/ssd512_coco/ssd512_coco_20210803_022849-0a47a1ca.pth"

# turn off benchmark
cudnn_benchmark = False

data_root = "/data/RarePlanes-Real/"
# info on the data
data = dict(
    # how many images on each GPU.
    # Batch size will be samples per GPU * Number of GPUs
    samples_per_gpu=16,
    # how many workers per GPU
    workers_per_gpu=0,
    # info about the train data
    train=dict(
        _delete_=True,
        type="DarkNetDataset",
        ann_file=None,  # required for Custom Datasets
        data_root="/data/DOTA/train_chipped512",
        classes="/data/DOTA/dota_classes.txt",
        filter_empty_gt=True,
        index_type=0,
        coordinate_type="top_left",
        # the pipeline that each train example will go through
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(type="RandomFlip", flip_ratio=0.5),
            dict(
                type="Normalize",
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=True,
            ),
            dict(type="DefaultFormatBundle"),  # this just transforms to a tensor
            dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
        ],
    ),
)

# how often should checkpoints be saved
checkpoint_config = dict(interval=10)

# which GPUs should be used
gpu_ids = [0]

# the random seed to use for weight init
seed = 0

# the output directory of models and logs
work_dir = "./train_output"


# what port should our NCCL use over TCP
dist_port = 8880

# what kind of runner: Epoch or Iteration
# and the max number
runner = dict(type="EpochBasedRunner", max_epochs=300)

# the optimizer's learning rate
optimizer = dict(lr=0.001)

# how to decay the learning rate
lr_config = dict(
    _delete_=True,
    policy="step",
    step=[260, 280],
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=0.001,
)
