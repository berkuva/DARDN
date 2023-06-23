import pandas as pd
import numpy as np
import torch
import sys


np.random.seed(0)
torch.manual_seed(0)


print("PyTorch version:", torch.__version__)
print("Python version:", sys.version)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


ENCODING_PATH = ""

# Chromosome 1
mat1 = np.load(ENCODING_PATH+'chr1_len5045.npz')
encoding1 = mat1['encoding']
ids1 = mat1['ids']

# # Chromosome 2
# mat2 = np.load(ENCODING_PATH+'chr2_len5045.npz')
# encoding2 = mat2['encoding']
# ids2 = mat2['ids']

# # Chromosome 3
# mat3 = np.load(ENCODING_PATH+'chr3_len5045.npz')
# encoding3 = mat3['encoding']
# ids3 = mat3['ids']

# # Chromosome 4
# mat4 = np.load(ENCODING_PATH+'chr4_len5045.npz')
# encoding4 = mat4['encoding']
# ids4 = mat4['ids']

# # Chromosome 5
# mat5 = np.load(ENCODING_PATH+'chr5_len5045.npz')
# encoding5 = mat5['encoding']
# ids5 = mat5['ids']

# # Chromosome 6
# mat6 = np.load(ENCODING_PATH+'chr6_len5045.npz')
# encoding6 = mat6['encoding']
# ids6 = mat6['ids']

# # Chromosome 7
# mat7 = np.load(ENCODING_PATH+'chr7_len5045.npz')
# encoding7 = mat7['encoding']
# ids7 = mat7['ids']

# # Chromosome 8
# mat8 = np.load(ENCODING_PATH+'chr8_len5045.npz')
# encoding8 = mat8['encoding']
# ids8 = mat8['ids']

# # Chromosome 9
# mat9 = np.load(ENCODING_PATH+'chr9_len5045.npz')
# encoding9 = mat9['encoding']
# ids9 = mat9['ids']

# # Chromosome 10
# mat10 = np.load(ENCODING_PATH+'chr10_len5045.npz')
# encoding10 = mat10['encoding']
# ids10 = mat10['ids']

# # Chromosome 11
# mat11 = np.load(ENCODING_PATH+'chr11_len5045.npz')
# encoding11 = mat11['encoding']
# ids11 = mat11['ids']

# # Chromosome 12
# mat12 = np.load(ENCODING_PATH+'chr12_len5045.npz')
# encoding12 = mat12['encoding']
# ids12 = mat12['ids']

# # Chromosome 13
# mat13 = np.load(ENCODING_PATH+'chr13_len5045.npz')
# encoding13 = mat13['encoding']
# ids13 = mat13['ids']

# # Chromosome 14
# mat14 = np.load(ENCODING_PATH+'chr14_len5045.npz')
# encoding14 = mat14['encoding']
# ids14 = mat14['ids']

# # Chromosome 15
# mat15 = np.load(ENCODING_PATH+'chr15_len5045.npz')
# encoding15 = mat15['encoding']
# ids15 = mat15['ids']

# # Chromosome 16
# mat16 = np.load(ENCODING_PATH+'chr16_len5045.npz')
# encoding16 = mat16['encoding']
# ids16 = mat16['ids']

# # Chromosome 17
# mat17 = np.load(ENCODING_PATH+'chr17_len5045.npz')
# encoding17 = mat17['encoding']
# ids17 = mat17['ids']

# # Chromosome 18
# mat18 = np.load(ENCODING_PATH+'chr18_len5045.npz')
# encoding18 = mat18['encoding']
# ids18 = mat18['ids']

# # Chromosome 19
# mat19 = np.load(ENCODING_PATH+'chr19_len5045.npz')
# encoding19 = mat19['encoding']
# ids19 = mat19['ids']

# # Chromosome 20
# mat20 = np.load(ENCODING_PATH+'chr20_len5045.npz')
# encoding20 = mat20['encoding']
# ids20 = mat20['ids']

# # Chromosome 21
# mat21 = np.load(ENCODING_PATH+'chr21_len5045.npz')
# encoding21 = mat21['encoding']
# ids21 = mat21['ids']

# # Chromosome 22
# mat22 = np.load(ENCODING_PATH+'chr22_len5045.npz')
# encoding22 = mat22['encoding']
# ids22 = mat22['ids']

# # Chromosome X
# matX = np.load(ENCODING_PATH+'chrX_len5045.npz')
# encodingX = matX['encoding']
# idsX = matX['ids']

# # Chromosome Y
# matY = np.load(ENCODING_PATH+'chrY_len5045.npz')
# encodingY = matY['encoding']
# idsY = matY['ids']


SEQLEN = 10000
dl_attri = torch.load("attributions.pt", map_location='cpu')
dl_attri = dl_attri.reshape(-1,4,SEQLEN)

# Data Loading.
df = pd.read_csv('T-ALL_binding_features.csv',
                 sep=',',
                 usecols=["id", "chr", "occupancy_score", "cancer_vs_other_stats", "start", "end"])

# This will use 150 gained sites specific to T_ALL.
num_gained_sites = 150
gained = df.sort_values(by='cancer_vs_other_stats', ascending=False).iloc[
    np.concatenate([np.arange(0, num_gained_sites)])]

gained_ids = gained.id.values

occupancy_threshold_score = 616
constitutive = df[df["occupancy_score"] >= occupancy_threshold_score]
constitutive = constitutive.sort_values(by="occupancy_score", ascending=False)

binding_file = 'union_binding_occupancy_score_GT3.csv'
binding_df = pd.read_csv(binding_file, sep=',', index_col=3)
