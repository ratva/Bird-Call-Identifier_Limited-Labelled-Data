import sys
import os, csv, argparse, wget
import torch, torchaudio, timm
from torch import nn
from torch.utils.data import WeightedRandomSampler
from torch.cuda.amp import autocast
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

# ===================== Change this =====================
home_dir = '/cluster/tufts/cs152l3dclass/arekhi01/Bird-Call-Identifier---Limited-Labelled-Data/audioSet-Pretrained/'

modelDirSubPath = 'birdclef_audio25/'
modelName = 'best_audio_model.pth'
model_path = f"{home_dir}exp/{modelDirSubPath}models/{modelName}"

te_data = f"{home_dir}Data/test_audio_nfalic01.json"

run_name = "test_25"
# =======================================================

sys.path.insert(0, f"{home_dir}src/")
sys.path.append('../')

import models
from utilities import *
from utilities.stats import calculate_stats
from traintest import train, validate
import json
from sklearn.metrics import precision_recall_curve
import importlib.util
spec = importlib.util.spec_from_file_location("dataloader", f"{home_dir}src/dataloader.py")
dataloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataloader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== Function Definitions =====================
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
        if os.path.exists(exp_dir+'/predictions') == False:
            os.mkdir(exp_dir+'/predictions')
        np.savetxt(exp_dir+'/predictions/target.csv', target, delimiter=',')
        print('Saving target to ' + exp_dir+'/predictions/target.csv')
        np.savetxt(exp_dir+'/predictions/predictions_' + str(epoch) + '.csv', audio_output, delimiter=',')
        print('Saving predictions to ' + exp_dir+'/predictions/predictions_' + str(epoch) + '.csv')
    return stats, loss

# ===================== Function Definitions =====================

audio_model = models.ASTModel(label_dim=12, fstride=10, tstride=10, input_fdim=128,
input_tdim=1024, imagenet_pretrain=True,
audioset_pretrain=False, model_size='base384')

# Assume each input spectrogram has 1024 time frames
input_tdim = 1024

checkpoint_path = f"{model_path}"

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

print("dataloader.py file: " + dataloader.__file__);

class Args:
    pass

args = Args()
args.exp_dir = os.path.dirname(os.path.dirname(model_path))
args.loss ='BCE'
if args.loss == 'BCE':
    args.loss_fn = nn.BCEWithLogitsLoss()
elif args.loss == 'CE':
    args.loss_fn = nn.CrossEntropyLoss()

# Mean and std for the test dataset calculated from src/get_norm_stats.py
te_audio_conf = {'num_mel_bins': 128, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': 'imageNet', 'mode':'evaluation', 'mean':-5.3776608, 'std':4.507704, 'noise':False}

print("class_indices: " + label_csv)

input_tdim = 1024  # Define input_tdim

# Dataloader will generate normalized fbanks
te_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetDataset(te_data, label_csv=label_csv, audio_conf=te_audio_conf),
    batch_size=12*2, shuffle=False, num_workers=0, pin_memory=True)

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

stats, _ = validate(audio_model, te_loader, args, run_name)

# ===================== Assess predictions =====================

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

print(f"\nCorrectly predicted labels: {correct_labels}")
print(f"Incorrectly predicted labels: {incorrect_labels}\n")

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

# ===================== Plots =====================

# Calculate precision and recall for each class
precision = dict()
recall = dict()
for i in range(len(class_names)):
    precision[i], recall[i], _ = precision_recall_curve(target[:, i], audio_output[:, i])

# Plot the precision-recall curve for each class
plt.figure(figsize=(12, 10))
for i in range(len(class_names)):
    plt.plot(recall[i], precision[i], lw=2, label=f'Class {class_names[i]}')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.grid(True)

# Save the figure
plt.savefig(predictionsDir + '/precision_recall_curve_' + str(run_name) + '.png')

# Show the plot
# plt.show()

# ===================== ROC Curve =====================
# Calculate the false positive rate and true positive rate for each class
fpr = dict()
tpr = dict()
for i in range(len(class_names)):
    fpr[i], tpr[i], _ = metrics.roc_curve(target[:, i], audio_output[:, i])

# Plot the ROC curve for each class
plt.figure(figsize=(12, 10))
for i in range(len(class_names)):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {class_names[i]}')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.grid(True)

# Save the figure
plt.savefig(predictionsDir + '/roc_curve_' + str(run_name) + '.png')

# Show the plot
# plt.show()

# ===================== Confusion Matrix =====================
# Calculate the confusion matrix
conf_matrix = confusion_matrix(true_classes, pred_classes, normalize='true')
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
plt.savefig(predictionsDir + '/confusion_matrix_' + str(run_name) + '.png')

# Show the plot
# plt.show()

