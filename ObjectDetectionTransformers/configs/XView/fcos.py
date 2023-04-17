# the mmdetection repo has been copied to /workspace/mmdetection so we can use
# their configs as a base
_base_ = "/workspace/mmdetection/configs/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco.py"
# pretrained weights
load_from = "https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco-ede514a8.pth"

# info on the data
data = dict(
    # how many images on each GPU.
    # Batch size will be samples per GPU * Number of GPUs
    samples_per_gpu=6,
    # how many workers per GPU
    workers_per_gpu=1,
    # info about the train data
    train=dict(
        _delete_=True,
        type="BlackbirdDataset",
        ann_file="/data/XView/chipped512/train_labels.json",
        img_prefix="/data/XView/chipped512/images",
        filter_empty_gt=True,
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
gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

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
optimizer = dict(lr=0.02)

# how to decay the learning rate
lr_config = dict(
    # remove all keys in this section
    # from the base config
    # keeping only what we
    # define below
    _delete_=True,
    # how should we decay LR
    policy="step",
    # at what steps
    step=[260, 280],
    # how do we warmup the LR
    warmup="linear",
    # how many iterations
    warmup_iters=500,
    # what do we mult by
    warmup_ratio=0.001,
)
