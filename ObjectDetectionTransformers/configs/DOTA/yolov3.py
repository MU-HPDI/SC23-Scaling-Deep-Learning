_base_ = "/workspace/mmdetection/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py"
load_from = "https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-608_273e_coco/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth"

img_norm_cfg = dict(mean=[0, 0, 0], std=[255.0, 255.0, 255.0], to_rgb=True)
train_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="Expand",
        mean=img_norm_cfg["mean"],
        to_rgb=img_norm_cfg["to_rgb"],
        ratio_range=(1, 2),
    ),
    dict(
        type="MinIoURandomCrop",
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3,
    ),
    dict(type="Resize", img_scale=[(512, 512)], keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

#########################################################
# info on the data
data_root = "/data/RarePlanes-Real/"
# info on the data
data = dict(
    # how many images on each GPU.
    # Batch size will be samples per GPU * Number of GPUs
    samples_per_gpu=1,
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
dist_port = 8880

runner = dict(type="EpochBasedRunner", max_epochs=300)
