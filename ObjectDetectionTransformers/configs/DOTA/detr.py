_base_ = "/workspace/mmdetection/configs/detr/detr_r50_8x2_150e_coco.py"
load_from = "https://download.openmmlab.com/mmdetection/v2.0/detr/detr_r50_8x2_150e_coco/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth"


#########################################################
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(
        type="AutoAugment",
        policies=[
            [
                dict(
                    type="Resize",
                    img_scale=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    multiscale_mode="value",
                    keep_ratio=True,
                )
            ],
            [
                dict(
                    type="Resize",
                    img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                    multiscale_mode="value",
                    keep_ratio=True,
                ),
                dict(
                    type="RandomCrop",
                    crop_type="absolute_range",
                    crop_size=(384, 600),
                    allow_negative_crop=True,
                ),
                dict(
                    type="Resize",
                    img_scale=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    multiscale_mode="value",
                    override=True,
                    keep_ratio=True,
                ),
            ],
        ],
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=1),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
# info on the data
data = dict(
    # how many images on each GPU.
    # Batch size will be samples per GPU * Number of GPUs
    samples_per_gpu=8,
    # how many workers per GPU
    workers_per_gpu=1,
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
        # train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
        # from the default setting in mmdet.
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

# what kind of runner: Epoch or Iteration
# and the max number
runner = dict(type="EpochBasedRunner", max_epochs=300)
