#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem=20g
#SBATCH --job-name="ast_as"
#SBATCH --output=./log_%j.txt
#SBATCH --time=12:00:00

set -x
# comment this line if not running on sls cluster
# . /data/sls/scratch/share-201907/slstoolchainrc
# source ../../../venvssast/bin/activate
source /cluster/tufts/hpc/apps/rhel8/external/apps/anaconda/202406/bin/activate
export TORCH_HOME=../../pretrained_models
mkdir -p ./exp

if [ -e SSAST-Base-Patch-400.pth ]
then
    echo "pretrained model already downloaded."
else
    wget https://www.dropbox.com/s/ewrzpco95n9jdz6/SSAST-Base-Patch-400.pth?dl=1 -O SSAST-Base-Patch-400.pth
fi

pretrain_exp=
pretrain_model=SSAST-Base-Patch-400
pretrain_path=./${pretrain_exp}/${pretrain_model}.pth

dataset=birdclef
set=balanced
dataset_mean=-4.2677393
dataset_std=4.5689974
target_length=1024
noise=False

task=ft_avgtok
model_size=base
head_lr=1
warmup=True

last_layer_finetuning=True
lr_decay=0.5

if [ $set == balanced ]
then
  bal=none
  lr=5e-5
  epoch=50
  # tr_data=/data/sls/scratch/yuangong/aed-pc/src/enhance_label/datafiles_local/balanced_train_data_type1_2_mean.json
  tr_data=/cluster/tufts/cs152l3dclass/ashen05/ssast/src/finetune/audioset/Data/train_audio.json
elif [ $set == full ]
then
  bal=bal
  lr=1e-5
  epoch=25
  # tr_data=/data/sls/scratch/yuangong/aed-pc/src/enhance_label/datafiles_local/whole_train_data.json
  tr_data=/cluster/tufts/cs152l3dclass/ashen05/ssast/src/finetune/audioset/Data/train_audio.json
fi

# te_data=/data/sls/scratch/yuangong/audioset/datafiles/eval_data.json
te_data=/cluster/tufts/cs152l3dclass/ashen05/ssast/src/finetune/audioset/Data/test_audio.json
va_data=/cluster/tufts/cs152l3dclass/ashen05/ssast/src/finetune/audioset/Data/val_audio.json
freqm=48
timem=192
mixup=0.5
fstride=10
tstride=10
fshape=16
tshape=16
batch_size=12
exp_dir=./exp/SSAST-50epochs-lr${lr}-lastlayerft${last_layer_finetuning}-decay${lr_decay}
class_indices=/cluster/tufts/cs152l3dclass/ashen05/ssast/src/finetune/audioset/data/birdclef_class_labels.csv

CUDA_CACHE_DISABLE=1 python -W ignore ../../run.py --dataset ${dataset} \
--data-train ${tr_data} --data-val ${va_data} --data-eval "${te_data}" --exp-dir $exp_dir \
--label-csv ${class_indices} --n_class 12 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} --warmup False --task ${task} \
--model_size ${model_size} --adaptschedule False \
--pretrained_mdl_path ${pretrain_path} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} \
--num_mel_bins 128 --head_lr ${head_lr} --noise ${noise} \
--lrscheduler_start 10 --lrscheduler_step 5 --lrscheduler_decay ${lr_decay} --wa True --wa_start 6 --wa_end 25 \
--loss BCE --metrics mAP --last_layer_ft "${last_layer_finetuning}"