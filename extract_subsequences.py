#!/usr/bin/env python
# coding: utf-8
from torch.utils.data import TensorDataset, DataLoader
from model import *
from utils import *


SLIDING_WINDOW_LENGTH = 10
SELECT_N_SUBSEQUENCES_AGGREGATE = 1000
# n x 10 subsequences will be selected, where n is the number of gained sequences
SELECT_N_SUBSEQUENCES_INDIVIDUAL = 10
AGGREGATE = True


model = MSResNet()
checkpoint = torch.load('pretrained_weights.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()

data, target_labels = get_data()
data = data[:num_original_gained_sites]
target_labels = target_labels[:num_original_gained_sites]

target_labels = np.ones(len(data))
gained_ids = get_gained_sequence_ids()
gained_ids = sorted(gained_ids)

X_train = torch.FloatTensor(np.array(data))
y_train = torch.FloatTensor(target_labels)
id_train = torch.LongTensor(gained_ids)

train_loader = build_dataloader(X_train, y_train, id_train, batch_size=1)

if AGGREGATE:
    # Choose how to extract subsequences: from all 150 gained sequences
    bedfile = build_subsequences_aggregate(data,
                                           trained_model=model,
                                           dataloader=train_loader,
                                           length=SEQLEN,

                                           window_size=SLIDING_WINDOW_LENGTH)
else:
    # Choose how to extract subsequences: n subsequences from individual gained sequences
    bedfile = build_subsequences_individual(trained_model=model,
                                            dataloader=train_loader,
                                            length=SEQLEN,
                                            window_size=SLIDING_WINDOW_LENGTH,
                                            select_n_subsequences=SELECT_N_SUBSEQUENCES_INDIVIDUAL)

print("DeepLIFT minimum score: {}: ".format(bedfile['DeepLIFT'].values.min()))
print("DeepLIFT maximum score: {}: ".format(bedfile['DeepLIFT'].values.max()))

# Select subsequences with n most positive (or most negative) DeepLIFT scores
bedfile_filtered = bedfile[bedfile["DeepLIFT"] > 0].nlargest(SELECT_N_SUBSEQUENCES_AGGREGATE, ['DeepLIFT'])
bedfile_filtered = bedfile_filtered.drop(["sequence", "DeepLIFT"], axis=1)
bedfile_filtered['void'] = np.nan
bedfile_filtered['strand'] = '+'

# Use the txt file for HOMER.
# HOMER command: findMotifsGenome.pl top_subsequences.txt hg38 path/to/output_dir -size 200
bedfile_filtered.to_csv("top_subsequences.txt", sep='\t', index=False, header=False)
