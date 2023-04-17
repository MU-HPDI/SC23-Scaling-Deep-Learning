_base_ = "/workspace/mmdetection/configs/retinanet/retinanet_x101_64x4d_fpn_mstrain_640-800_3x_coco.py"

load_from = "https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_mstrain_3x_coco/retinanet_x101_64x4d_fpn_mstrain_3x_coco_20210719_051838-022c2187.pth"

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

# In mstrain 3x config, img_scale=[(1333, 640), (1333, 800)],
# multiscale_mode='range'
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="Resize",
        img_scale=[(1333, 640), (1333, 800)],
        multiscale_mode="range",
        keep_ratio=True,
    ),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
model = dict(pretrained=None)

#########################################################
# info on the data
# info on the data
data_root = "/data/RarePlanes-Real/"
# info on the data
data = dict(
    # how many images on each GPU.
    # Batch size will be samples per GPU * Number of GPUs
    samples_per_gpu=6,
    # how many workers per GPU
    workers_per_gpu=1,
    # info about the train data
    train=dict(
        # remove all the keys from the base in this section
        _delete_=True,
        type="DarkNetDataset",
        ann_file=None,  # required for Custom Datasets
        data_root="/data/DOTA/train_chipped512",
        classes="/data/DOTA/dota_classes.txt",
        filter_empty_gt=True,
        index_type=0,
        coordinate_type="top_left",
        pipeline=train_pipeline,
    ),
    train_dataloader=dict(dist=True, shuffle=True, persistent_workers=True),
)

# how often should checkpoints be saved
checkpoint_config = dict(interval=5)

# which GPUs should be used
gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

# the random seed to use for weight init
seed = 0

# the output directory of models and logs
work_dir = "./train_output"


# what port should our NCCL use over TCP
dist_port = 9990

optimizer = dict(type="SGD", lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[260, 280],
)
runner = dict(type="EpochBasedRunner", max_epochs=300)
