_base_ = './_base_/default_runtime.py'

dataset_type = 'PoseDataset'
ann_file = 'data/dataset.pkl'

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn',
        graph_cfg=dict(layout='coco', mode='spatial')),
    cls_head=dict(type='GCNHead', num_classes=20, in_channels=256))

train_pipeline = [
    # dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['jm']),
    dict(type='UniformSampleFrames', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]
val_pipeline = [
    # dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['jm']),
    dict(
        type='UniformSampleFrames', clip_len=100, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]
test_pipeline = [
    # dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['jm']),
    dict(
        type='UniformSampleFrames', clip_len=10, num_clips=10,
        test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=128,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file,
            pipeline=train_pipeline,
            split='xsub_train')))
val_dataloader = dict(
    batch_size=128,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=val_pipeline,
        split='xsub_val',
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline,
        split='xsub_val',
        test_mode=True))

val_evaluator = [dict(type='AccMetric')]
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=50, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=0.15, momentum=0.9, weight_decay=0.0005, nesterov=True))

# optim_wrapper = dict(
#     optimizer=dict(
#         type='Adam',
#         lr=0.001,
#         betas=(0.9, 0.999),
#         eps=1e-08,
#         weight_decay=0,
#         amsgrad=False)
# )

param_scheduler = dict(
    type='MultiStepLR',
    by_epoch=True,
    milestones=[100, 150],
    gamma=0.1)

default_hooks = dict(checkpoint=dict(interval=1), logger=dict(interval=100))

load_from = 'checkpoints/best_checkpoint.pth'