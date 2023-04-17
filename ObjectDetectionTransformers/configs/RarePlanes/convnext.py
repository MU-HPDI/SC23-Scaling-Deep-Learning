_base_ = "/workspace/cgimmd/configs/cgi_base.py"

custom_model = dict(type='convnext_faster_rcnn')

load_from = "/data/weights/faster_rcnn_convnext_coco.pth"

checkpoint_config = dict(interval=5)

gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]

device = 'cuda'

dist_port = 8880
seed = 0
work_dir = './train_output'

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[260, 280])

runner = dict(type='EpochBasedRunner', max_epochs=300)

data_root = "/data/RarePlanes-Real/"
# info on the data
data = dict(
    # how many images on each GPU.
    # Batch size will be samples per GPU * Number of GPUs
    samples_per_gpu=12,
    # how many workers per GPU
    workers_per_gpu=1,

    # info about the train data
    train=dict(
        _delete_=True,
        # what type of dataset
        type="CocoDataset",
        # the prefix in front of the image filenames
        ann_file=data_root + "coco_rareplanes/role_real_coco_train.json",
        img_prefix=data_root + 'train/PS-RGB_tiled/',
        classes=data_root + "coco_rareplanes/train_classes.txt",
        filter_empty_gt=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels'],
                meta_keys=[
                    'filename', 'img_shape', 'pad_shape', 'scale_factor'
                ])
        ]),
    train_dataloader=dict(dist=True, shuffle=True, persistent_workers=True)
)
