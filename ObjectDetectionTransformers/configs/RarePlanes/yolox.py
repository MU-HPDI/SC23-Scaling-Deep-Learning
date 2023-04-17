# the mmdetection repo has been copied to /workspace/mmdetection so we can use
# their configs as a base
_base_ = "/workspace/mmdetection/configs/yolox/yolox_x_8x8_300e_coco.py"

# pretrained weights
load_from = "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth"


img_scale = (512, 512)

model = dict(
    input_size=img_scale,
)

train_pipeline = [
    dict(type="Mosaic", img_scale=img_scale, pad_val=114.0),
    dict(
        type="RandomAffine",
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
    ),
    dict(type="MixUp", img_scale=img_scale, ratio_range=(0.8, 1.6), pad_val=114.0),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip", flip_ratio=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(type="Resize", img_scale=img_scale, keep_ratio=True),
    dict(
        type="Pad",
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0)),
    ),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

data_root = "/data/RarePlanes-Real/"
train_dataset = dict(
    type="MultiImageMixDataset",
    dataset=dict(
        type="CocoDataset",
        # the prefix in front of the image filenames
        ann_file=data_root + "coco_rareplanes/role_real_coco_train.json",
        img_prefix=data_root + "train/PS-RGB_tiled/",
        classes=data_root + "coco_rareplanes/classes.txt",
        filter_empty_gt=True,
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations", with_bbox=True),
        ],
    ),
    pipeline=train_pipeline,
    cache=False,
)
# info on the data
data = dict(
    # how many images on each GPU.
    # Batch size will be samples per GPU * Number of GPUs
    samples_per_gpu=6,
    # how many workers per GPU
    workers_per_gpu=1,
    # info about the train data
    train=train_dataset,
)

# how often should checkpoints be saved
checkpoint_config = dict(interval=5)

# which GPUs should be used
gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]

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
