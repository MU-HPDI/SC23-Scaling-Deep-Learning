MODELS="convnext deformable-detr detr fcos retinanet ssd swin vit yolov3 yolox"
DATASETS="dota xview rareplanes"
OUTPUTDIR="/output"
BATCHSIZE=4

for MODEL in $MODELS; do
    export MODEL=$MODEL
    for DATASET in $DATASETS; do
        export WORKDIR=$OUTPUTDIR/$DATASET/$MODEL
        export DATASET=$DATASET
        envsubst < train_job.yml | kubectl apply -f -
    done
done
