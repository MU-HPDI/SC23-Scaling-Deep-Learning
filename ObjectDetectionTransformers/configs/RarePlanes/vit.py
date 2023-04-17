_base_ = "/workspace/cgimmd/configs/cgi_base.py"

custom_model = dict(type='vit_retinanet')

load_from="/data/weights/upernet_vit-b16_mln_512x512_80k_ade20k_20210624_130547-0403cee1_BACKBONE.pth"

checkpoint_config = dict(interval=5)

gpu_ids = [0]

device = 'cuda'

dist_port = 8880
seed = 0
work_dir = './train_output'

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

runner = dict(type='EpochBasedRunner', max_epochs=300)

data_root = "/data/RarePlanes-Real/"
# info on the data
data = dict(
    # how many images on each GPU.
    # Batch size will be samples per GPU * Number of GPUs
    samples_per_gpu=8,
    # how many workers per GPU
    workers_per_gpu=4,

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
            dict(type="RandomFlip", flip_ratio=0.5, direction=["horizontal", "vertical"]),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True
            ),
            dict(type="Pad", size=(512,512), pad_val=0),
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
