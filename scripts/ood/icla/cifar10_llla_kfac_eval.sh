EXP_NAME='c10-llla-kfac'

export WANDB_BASE_URL="https://api.wandb.ai"
if [ -f .env ]; then
  export $(cat .env | xargs)
fi
unset WANDB_ENTITY

export CUDA_VISIBLE_DEVICES=0

for SEED in 0 1 2 
do
    PYTHONPATH='.':$PYTHONPATH \
    python3 main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/datasets/cifar10/cifar10_ood.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/llla.yml \
    --num_workers 8 \
    --exp_name "eval" \
    --output_dir "output_inference/$SEED" \
    --postprocessor.postprocessor_args.llla_type 'k-fac' \
    --postprocessor.postprocessor_args.checkpoint_path "pretrained_weights/icla_c10/$(expr $SEED + 1)/best.ckpt" \
    --postprocessor.postprocessor_args.optimize_precision true \
    --dataset.test.batch_size 64 \
    --dataset.val.batch_size 64 \
    --ood_dataset.batch_size 64 \
    --recorder.project icla \
    --recorder.experiment "$EXP_NAME-eval-seed-$SEED" \
    --recorder.group "$EXP_NAME-eval"

    PYTHONPATH='.':$PYTHONPATH \
    python3 main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/datasets/cifar10/cifar10_ood.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/test/test_ece_la.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/llla.yml \
    --merge_option "merge" \
    --num_workers 8 \
    --exp_name "eval" \
    --output_dir "output_inference/$SEED" \
    --postprocessor.postprocessor_args.llla_type 'k-fac' \
    --postprocessor.postprocessor_args.checkpoint_path "pretrained_weights/icla_c10/$(expr $SEED + 1)/best.ckpt" \
    --postprocessor.postprocessor_args.optimize_precision true \
    --dataset.test.batch_size 64 \
    --dataset.val.batch_size 64 \
    --ood_dataset.batch_size 64 \
    --recorder.project icla \
    --recorder.experiment "$EXP_NAME-eval-ece-seed-$SEED" \
    --recorder.group "$EXP_NAME-eval"
done