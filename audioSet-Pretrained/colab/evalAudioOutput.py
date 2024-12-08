import sys
import os
import datetime
# sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
# from utilities import *
import time
import torch
from torch import nn
import torchaudio
import numpy as np
import pickle
from torch.cuda.amp import autocast,GradScaler
import matplotlib.pyplot as plt
import soundfile

# Add the src directory to the Python path
src_dir = os.path.abspath("/cluster/tufts/cs152l3dclass/arekhi01/Bird-Call-Identifier---Limited-Labelled-Data/audioSet-Pretrained/src")
if src_dir not in sys.path:
    sys.path.append(src_dir)
sys.path.insert(0, "src/utilities")
sys.path.insert(0, "src/models")

import dataloader
import models
from utilities import *
from traintest import train, validate
import numpy as np
from scipy import stats
import torch
from src.models import ASTModel
from stats import calculate_stats
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

exp_dir = "exp/birdclef_audio25"

epoch = "test_audio"
target = np.loadtxt(exp_dir+'/predictions/test_target.csv', delimiter=',')
audio_output = np.loadtxt(exp_dir+'/predictions/predictions_' + str(epoch) + '.csv', delimiter=',')


statsOut = calculate_stats(audio_output, target)

# Initialize lists to store true and predicted classes
true_classes = []
pred_classes = []

correct_labels = 0
incorrect_labels = 0

for idx in range(0, len(target)):
    trueClass = np.argmax(target[idx])
    predClass = np.argmax(audio_output[idx])

    # Append true and predicted classes to the lists
    true_classes.append(trueClass)
    pred_classes.append(predClass)
    if trueClass != predClass:
        # print(f"True class: {trueClass}, Predicted class: {predClass}")
        incorrect_labels += 1
    else:
        # print(f"Correctly predicted class: {trueClass}")
        correct_labels += 1

print(f"Correctly predicted labels: {correct_labels}")
print(f"Incorrectly predicted labels: {incorrect_labels}")

# After the loop, calculate the confusion matrix
conf_matrix = confusion_matrix(true_classes, pred_classes, normalize='true')

class_names = [
    "Greenish Warbler",
    "Black-crowned Night-Heron",
    "Blyth's Reed Warbler",
    "Little Egret",
    "Great Egret",
    "Red-whiskered Bulbul",
    "Eurasian Coot",
    "Rose-ringed Parakeet",
    "Greater Racket-tailed Drongo",
    "Gray-headed Canary-Flycatcher",
    "Gray Heron",
    "White-breasted Waterhen"
]

# Convert statsOut to a DataFrame for better readability
df = pd.DataFrame(statsOut, index=class_names)

# Print the DataFrame as a table
print(df)

# Save the DataFrame to a CSV file
df.to_csv(exp_dir + '/predictions/stats_output_' + str(epoch) + '.csv', index=True)

# Plot the confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix')

# Adjust the x-axis labels
plt.xticks(ticks=np.arange(len(class_names)) + 0.5, labels=class_names, rotation=45, ha='right')  # Add ha='right' for proper alignment
plt.tight_layout()  # Adjust the padding to make sure everything fits

# Save the figure
plt.savefig(exp_dir + '/predictions/val_confusion_matrix' + str(epoch) + '.png')

# Show the plot
plt.show()

# =====================

class ASTModelVis(ASTModel):
    def get_att_map(self, block, x):
        qkv = block.attn.qkv
        num_heads = block.attn.num_heads
        scale = block.attn.scale
        B, N, C = x.shape
        qkv = qkv(x).reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        return attn

    def forward_visualization(self, x):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        # save the attention map of each of 12 Transformer layer
        att_list = []
        for blk in self.v.blocks:
            cur_att = self.get_att_map(blk, x)
            att_list.append(cur_att)
            x = blk(x)
        return att_list

def make_features(wav_name, mel_bins, target_length=1024):
    waveform, sr = torchaudio.load(wav_name)
    # assert sr == 16000, 'input audio sampling rate must be 16kHz'

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0, frame_shift=10)

    n_frames = fbank.shape[0]

    # print(f'[*INFO] {wav_name} has {n_frames} frames')
    # print(f'[*INFO] {wav_name} has dimensions {fbank.shape} and type {fbank.dtype}')
    
    # import matplotlib.pyplot as plt

    # plt.imshow(fbank.T, aspect='auto', origin='lower')
    # plt.title('Mel Spectrogram')
    # plt.xlabel('Time')
    # plt.ylabel('Mel Frequency')
    # plt.colorbar(format='%+2.0f dB')
    # plt.show()
    
    # print(f'fbank min: {fbank.min()}, fbank max: {fbank.max()}')
    
    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank


def load_label(label_csv):
    with open(label_csv, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        lines = list(reader)
    labels = []
    ids = []  # Each label has a unique id such as "/m/068hy"
    for i1 in range(1, len(lines)):
        id = lines[i1][1]
        label = lines[i1][2]
        ids.append(id)
        labels.append(label)
    return labels


# eval_data_path = ""

# eval_loader = torch.utils.data.DataLoader(
#     dataloader.AudiosetDataset(eval_data_path, label_csv=args.label_csv, audio_conf=val_audio_conf),
#     batch_size=100, shuffle=False, num_workers=16, pin_memory=True)


# feats = make_features('../sample_audios/sample_audio.flac', mel_bins=128)   


# stats, _ = validate(audio_model, eval_loader, args, model_idx)
# mAP = np.mean([stat['AP'] for stat in stats])
# mAUC = np.mean([stat['auc'] for stat in stats])
# dprime = d_prime(mAUC)
# ensemble_res[model_idx, :] = [mAP, mAUC, dprime]
# print("Model {:d} {:s} mAP: {:.6f}, AUC: {:.6f}, d-prime: {:.6f}".format(model_idx, mdl, mAP, mAUC, dprime))

# input_tdim = 1024
# checkpoint_path = '../pretrained_models/audio_mdl.pth'
# # now load the visualization model
# ast_mdl = ASTModelVis(label_dim=12, input_tdim=input_tdim, imagenet_pretrain=True, audioset_pretrain=False)
# print(f'[*INFO] load checkpoint: {checkpoint_path}')
# checkpoint = torch.load(checkpoint_path, map_location=torch.device('mps'))
# audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
# audio_model.load_state_dict(checkpoint)
# audio_model = audio_model.to(torch.device("mps"))
# audio_model.eval()     

# with torch.no_grad():
#   with autocast():
#     output = audio_model.forward(feats_data)
#     output = torch.sigmoid(output)
# result_output = output.data.cpu().numpy()[0]
# sorted_indexes = np.argsort(result_output)[::-1]

# # Print audio tagging top probabilities
# print('Predice results:')
# for k in range(10):
#     print('- {}: {:.4f}'.format(np.array(labels)[sorted_indexes[k]], result_output[sorted_indexes[k]]))
# print('Listen to this sample: ')
# IPython.display.Audio('../sample_audios/sample_audio.flac')