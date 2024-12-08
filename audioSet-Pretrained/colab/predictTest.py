import sys
import os, csv, argparse, wget
import torch, torchaudio, timm
import numpy as np
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
import time
from torch import nn
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

# ===================== Change this =====================
home_dir = '/cluster/tufts/cs152l3dclass/arekhi01/Bird-Call-Identifier---Limited-Labelled-Data/audioSet-Pretrained/'

modelDirSubPath = 'birdclef_audio25/'
modelName = 'best_audio_model.pth'
 
te_data = f"{home_dir}Data/test_audio_nfalic01.json"

run_name = "test"
# =======================================================

model_path = f"{home_dir}exp/{modelDirSubPath}models/{modelName}"

sys.path.insert(0, f"{home_dir}src/")
sys.path.append('../')

import models
from utilities import *
from utilities.stats import calculate_stats
from traintest import train, validate
import json
import importlib.util
spec = importlib.util.spec_from_file_location("dataloader", f"{home_dir}src/dataloader.py")
dataloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataloader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def make_features(wav_name, mel_bins, target_length=1024):
#     waveform, sr = torchaudio.load(wav_name)
#     # assert sr == 16000, 'input audio sampling rate must be 16kHz'

#     fbank = torchaudio.compliance.kaldi.fbank(
#         waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
#         window_type='hanning', num_mel_bins=mel_bins, dither=0.0, frame_shift=10)

#     n_frames = fbank.shape[0]

#     # print(f'[*INFO] {wav_name} has {n_frames} frames')
#     # print(f'[*INFO] {wav_name} has dimensions {fbank.shape} and type {fbank.dtype}')
    
#     # import matplotlib.pyplot as plt

#     # plt.imshow(fbank.T, aspect='auto', origin='lower')
#     # plt.title('Mel Spectrogram')
#     # plt.xlabel('Time')
#     # plt.ylabel('Mel Frequency')
#     # plt.colorbar(format='%+2.0f dB')
#     # plt.show()
    
#     # print(f'fbank min: {fbank.min()}, fbank max: {fbank.max()}')
    
#     p = target_length - n_frames
#     if p > 0:
#         m = torch.nn.ZeroPad2d((0, 0, 0, p))
#         fbank = m(fbank)
#     elif p < 0:
#         fbank = fbank[0:target_length, :]

#     fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
#     return fbank


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

audio_model = models.ASTModel(label_dim=12, fstride=10, tstride=10, input_fdim=128,
input_tdim=1024, imagenet_pretrain=True,
audioset_pretrain=False, model_size='base384')

# Assume each input spectrogram has 1024 time frames
input_tdim = 1024
# checkpoint_path = f"{home_dir}exp/birdclef_audio1/models/best_audio_model.pth'

# exp folder is subfolder of audioset_pretrained.
checkpoint_path = f"{model_path}"

# # now load the visualization model
# ast_mdl = ASTModelVis(label_dim=12, input_tdim=input_tdim, imagenet_pretrain=True, audioset_pretrain=False)
print(f'[*INFO] load checkpoint: {checkpoint_path}')
checkpoint = torch.load(checkpoint_path, map_location=device)

# Remove 'module.' prefix from keys
state_dict = checkpoint
new_state_dict = {}
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
    new_state_dict[name] = v

audio_model.load_state_dict(new_state_dict)
audio_model = audio_model.to(device)
audio_model.eval()          

# Load the AudioSet label set
label_csv = f"{home_dir}src/bird_class_labels_indices.csv"     # label and indices for audioset data
labels = load_label(label_csv)


# def validate(audio_model, data_loader, args, mode):
#     for i, (audio_input, labels) in enumerate(data_loader):
#         # Resolve the path before loading the audio file
#         audio_input = [resolve_path(f) for f in audio_input]
#         # Your existing processing code
#         for f in audio_input:
#             if not os.path.exists(f):
#                 raise FileNotFoundError(f"File not found: {f}")
#         # Continue with your existing code
#         B = audio_input.size(0)
#         audio_input = audio_input.to(device, non_blocking=True)
#         labels = labels.to(device, non_blocking=True)
        
def validate(audio_model, data_loader, args, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    # switch to evaluate mode
    audio_model.eval()

    end = time.time()
    A_predictions = []
    A_targets = []
    A_loss = []
    with torch.no_grad():
        for i, (audio_input, labels) in enumerate(data_loader):

            # print(audio_input.shape)
            # print(labels.shape)

            # print(type(audio_input))
            # print(type(labels))
            
            audio_input = audio_input.to(device)

            # compute output
            audio_output = audio_model(audio_input)
            audio_output = torch.sigmoid(audio_output)
            predictions = audio_output.to('cpu').detach()

            A_predictions.append(predictions)
            A_targets.append(labels)

            # compute the loss
            labels = labels.to(device)
            if isinstance(args.loss_fn, torch.nn.CrossEntropyLoss):
                loss = args.loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
            else:
                loss = args.loss_fn(audio_output, labels)
            A_loss.append(loss.to('cpu').detach())

            batch_time.update(time.time() - end)
            end = time.time()

        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)
        stats = calculate_stats(audio_output, target)

        # save the prediction here
        exp_dir = args.exp_dir
        # if os.path.exists(exp_dir+'/predictions/target.csv') == False:
        os.mkdir(exp_dir+'/predictions')
        np.savetxt(exp_dir+'/predictions/target.csv', target, delimiter=',')
        print('Saving target to ' + exp_dir+'/predictions/target.csv')
        np.savetxt(exp_dir+'/predictions/predictions_' + str(epoch) + '.csv', audio_output, delimiter=',')
        print('Saving predictions to ' + exp_dir+'/predictions/predictions_' + str(epoch) + '.csv')
    return stats, loss


print("dataloader.py file: " + dataloader.__file__);

class Args:
    pass

args = Args()
args.alen = 5
args.audio_length = 1024
args.audioset_pretrain = False
args.bal ='bal'
args.batch_size =12
args.data_eval = te_data
args.data_train = os.path.join(home_dir,"Data/train_audio_nfalic01.json") #f"{home_dir}Data/train_audio.json" 
args.data_val = os.path.join(home_dir,"Data/val_audio_nfalic01.json") #f"{home_dir}Data/val_audio.json"
args.dataset ='audioset'
args.dataset_mean =-4.2677393
args.dataset_std =4.5689974
args.exp_dir = os.path.dirname(os.path.dirname(model_path))
args.freqm =48
args.fstride =10
args.imagenet_pretrain =True
args.label_csv = label_csv
args.loss ='BCE'
args.lr =1e-05
args.lrscheduler_decay =0.5
args.lrscheduler_start =2
args.lrscheduler_step =1
args.metrics ='mAP'
args.mixup =0.5
args.model ='ast'
args.n_class =12
args.n_epochs =5
args.n_print_steps =100
args.noise =False
args.num_workers =0
args.optim ='adam'
args.save_model =True
args.timem =192
args.tstride =10
args.wa =True
args.wa_end =5
args.wa_start =1
args.warmup =True
if args.loss == 'BCE':
    args.loss_fn = nn.BCEWithLogitsLoss()
elif args.loss == 'CE':
    args.loss_fn = nn.CrossEntropyLoss()

# sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

te_audio_conf = {'num_mel_bins': 128, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': 'imageNet', 'mode':'evaluation', 'mean':-4.2677393, 'std':4.5689974, 'noise':False}

print("class_indices: " + label_csv)

input_tdim = 1024  # Define input_tdim

te_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetDataset(te_data, label_csv=label_csv, audio_conf=te_audio_conf),
    batch_size=12*2, shuffle=False, num_workers=0, pin_memory=True)
# sampler=sampler, - only used for train loader, not val or test

# for i, (audio_input, labels) in enumerate(te_loader):
#     B = audio_input.size(0)
#     audio_input = audio_input.to(device, non_blocking=True)
#     labels = labels.to(device, non_blocking=True)

#     feats = make_features('/Users/avtar/Library/CloudStorage/OneDrive-Tufts/Tufts CS/CS152 L3D/Project/Code/audioSet-Pretrained/sample_audios/XC37740.ogg', mel_bins=128) 

#     feats_data = feats.expand(1, input_tdim, 128)  # reshape the feature
#     feats_data = feats_data.to(device)

#     # Make the prediction
#     with torch.no_grad():
#         with autocast():
#             output = audio_model.forward(feats_data)
#             output = torch.sigmoid(output)
#     result_output = output.data.cpu().numpy()[0]
#     sorted_indexes = np.argsort(result_output)[::-1]

#     # Print audio tagging top probabilities
#     print('Prediction results:')
#     for k in range(10):
#         print('- {}: {:.4f}'.format(np.array(labels)[sorted_indexes[k]], result_output[sorted_indexes[k]]))

print("Current working directory:", os.getcwd())

# Load the JSON file to get the paths

with open(te_data, 'r') as f:
    data = json.load(f)

# Check if the first file exists
first_file_path = data['data'][0]['wav']
abs_path = os.path.abspath(first_file_path)

if os.path.exists(abs_path):
    print("\nSample file from json " + abs_path + " exists.")
else:
    print("\nSample file from json " + abs_path + " does not exist.")

# for i, (audio_input, labels) in enumerate(te_loader):
#     # Resolve the path before loading the audio file
#     print("Before resolving path:", audio_input)
#     audio_input = [resolve_path(f) for f in audio_input]


stats, _ = validate(audio_model, te_loader, args, run_name)

predictionsDir = f"{home_dir}exp/{modelDirSubPath}predictions/"
print("\nPredictions directory: " + predictionsDir)
target = np.loadtxt(predictionsDir + 'target.csv', delimiter=',')
audio_output = np.loadtxt(predictionsDir + 'predictions_' + str(run_name) + '.csv', delimiter=',')

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
        incorrect_labels += 1
    else:
        correct_labels += 1

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
df = pd.DataFrame(stats, index=class_names)

# Print the DataFrame as a table
print(df)

# Save the DataFrame to a CSV file
df.to_csv(predictionsDir + '/stats_output_' + str(run_name) + '.csv', index=True)

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
plt.savefig(predictionsDir + '/test_confusion_matrix_' + str(run_name) + '.png')

# Show the plot
plt.show()

