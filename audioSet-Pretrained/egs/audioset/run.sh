#!/bin/bash
#SBATCH -p sm
#SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-3,sls-sm-[5,6]
##SBATCH -p gpu
##SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=48000
#SBATCH --job-name="ast_as"
#SBATCH --output=./log_%j.txt

set -x
# comment this line if not running on sls cluster
# . /data/sls/scratch/share-201907/slstoolchainrc
# source ../../venvast/bin/activate
# export TORCH_HOME=../../pretrained_models

# pathToCode="/Users/avtar/Library/CloudStorage/OneDrive-Tufts/Tufts CS/CS152 L3D/Project/Code/"
pathToCode=/cluster/tufts/cs152l3dclass/nfalic01/Bird-Call-Identifier---Limited-Labelled-Data/
export TORCH_HOME="${pathToCode}audioSet-Pretrained/pretrained_models"

model=ast
dataset=audioset
# full or balanced for audioset
set=full
imagenetpretrain=True
if [ $set == balanced ]
then
  bal=none
  lr=5e-5
  epoch=25
  # tr_data=/data/sls/scratch/yuangong/aed-pc/src/enhance_label/datafiles_local/balanced_train_data_type1_2_mean.json
  # tr_data=/cluster/tufts/cs152l3dclass/nfalic01/Bird-Call-Identifier---Limited-Labelled-Data/Data/train_audio.json
  # tr_data="/Users/avtar/Library/CloudStorage/OneDrive-Tufts/Tufts CS/CS152 L3D/Project/Code/Data/train_audio.json"
  # tr_data="${pathToCode}Data/train_audio.json"
  tr_data=/cluster/tufts/cs152l3dclass/nfalic01/Bird-Call-Identifier---Limited-Labelled-Data/audioSet-Pretrained/Data/train_audio.json
  lrscheduler_start=10
  lrscheduler_step=5
  lrscheduler_decay=0.5
  wa_start=6
  wa_end=25
else
  bal=bal
  lr=1e-5
  epoch=5
  # tr_data=/data/sls/scratch/yuangong/aed-pc/src/enhance_label/datafiles_local/whole_train_data.json
  # tr_data=/cluster/tufts/cs152l3dclass/nfalic01/Bird-Call-Identifier---Limited-Labelled-Data/Data/train_audio.json
  # tr_data="/Users/avtar/Library/CloudStorage/OneDrive-Tufts/Tufts CS/CS152 L3D/Project/Code/Data/train_audio.json"
  tr_data=/cluster/tufts/cs152l3dclass/nfalic01/Bird-Call-Identifier---Limited-Labelled-Data/audioSet-Pretrained/Data/train_audio.json
  
  lrscheduler_start=2
  lrscheduler_step=1
  lrscheduler_decay=0.5
  wa_start=1
  wa_end=5
fi
# te_data=/data/sls/scratch/yuangong/audioset/datafiles/eval_data.json
# te_data=/cluster/tufts/cs152l3dclass/nfalic01/Bird-Call-Identifier---Limited-Labelled-Data/Data/test_audio.json
# va_data=/cluster/tufts/cs152l3dclass/nfalic01/Bird-Call-Identifier---Limited-Labelled-Data/Data/val_audio.json
# te_data="/Users/avtar/Library/CloudStorage/OneDrive-Tufts/Tufts CS/CS152 L3D/Project/Code/Data/test_audio.json"
# va_data="/Users/avtar/Library/CloudStorage/OneDrive-Tufts/Tufts CS/CS152 L3D/Project/Code/Data/val_audio.json"

te_data=/cluster/tufts/cs152l3dclass/nfalic01/Bird-Call-Identifier---Limited-Labelled-Data/audioSet-Pretrained/Data/test_audio.json
va_data=/cluster/tufts/cs152l3dclass/nfalic01/Bird-Call-Identifier---Limited-Labelled-Data/audioSet-Pretrained/Data/val_audio.json
class_indices=/cluster/tufts/cs152l3dclass/nfalic01/Bird-Call-Identifier---Limited-Labelled-Data/audioSet-Pretrained/egs/audioset/data/bird_class_labels_indices.csv
freqm=48
timem=192
mixup=0.5
# corresponding to overlap of 6 for 16*16 patches
fstride=10
tstride=10
batch_size=12
num_workers=16

dataset_mean=-4.2677393
dataset_std=4.5689974
audio_length=1024
noise=False

metrics=mAP
loss=BCE
warmup=True
wa=True
runPath="${pathToCode}audioSet-Pretrained/src/run.py"

# exp_dir=./exp/test-${set}-f$fstride-t$tstride-p$imagenetpretrain-b$batch_size-lr${lr}-decoupe
exp_dir=./exp/birdclef_audio1
if [ -d $exp_dir ]; then
  echo 'exp exist'
  exit
fi
mkdir -p $exp_dir

# Added debug mode - disable by replacing "python -m debugpy --listen 5678 --wait-for-client" with "python -W ignore"
CUDA_CACHE_DISABLE=1 python -W ignore "${runPath}" --model "${model}" --dataset "${dataset}" -w "${num_workers}" \
--data-train "${tr_data}" --data-val "${va_data}" --data-eval "${te_data}" --exp-dir "${exp_dir}" \
--label-csv "${class_indices}" --n_class 12 \
--lr "${lr}" --n-epochs "${epoch}" --batch-size "${batch_size}" --save_model True \
--freqm "${freqm}" --timem "${timem}" --mixup "${mixup}" --bal "${bal}" \
--tstride "${tstride}" --fstride "${fstride}" --imagenet_pretrain "${imagenetpretrain}" \
--dataset_mean "${dataset_mean}" --dataset_std "${dataset_std}" --audio_length "${audio_length}" --noise "${noise}" \
--metrics "${metrics}" --loss "${loss}" --warmup "${warmup}" --lrscheduler_start "${lrscheduler_start}" --lrscheduler_step "${lrscheduler_step}" --lrscheduler_decay "${lrscheduler_decay}" \
--wa "${wa}" --wa_start "${wa_start}" --wa_end "${wa_end}"