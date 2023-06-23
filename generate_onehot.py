import os
import numpy as np
import pandas as pd
import twobitreader
from Bio.Seq import Seq


def one_hot_encoded(seq):
    # one hot encoding of the input sequence
    seq_int_mapping = dict(zip("ACGTN", range(5)))
    seq_encoded_mapping = np.append(np.eye(4),[[0,0,0,0]],axis=0)
    seq = [seq_int_mapping[i] for i in seq]
    return seq_encoded_mapping[seq]


def return_enconded_seq(binding_df, ii, genome, length):
    # one hot encoding of input CBS
    chrom = binding_df.loc[ii, 'chr']
    mid_position = binding_df.loc[ii, 'mid_position']
    start = max(mid_position-length, 0)
    end = mid_position+length
    sequence = Seq(genome[chrom][start:end]).upper()

    one_hot = one_hot_encoded(sequence)
    return one_hot


def main(chrom, length):
    outdir = 'one_hot_encoding/'
    os.makedirs(outdir, exist_ok=True)

    # read 2bit file
    hg38_2bit_file = './tall/hg38.2bit'
    genome = twobitreader.TwoBitFile(hg38_2bit_file)

    # binding info for each chrom
    binding_file = './tall/union_binding_occupancy_score_GT3.csv'
    binding_df = pd.read_csv(binding_file, sep=',', index_col=3)
    binding_df = binding_df[binding_df.chr == chrom]

    # get the one-hot-encoding of each CBS
    ids = binding_df.index.to_numpy()
    one_hot_array = [[return_enconded_seq(binding_df, ii, genome, length)] for ii in binding_df.index]
    one_hot_array = np.concatenate(one_hot_array)

    # Saved file contains two dictionary keys: 'encoding' and 'ids'
    np.savez_compressed('{}/{}_len{}'.format(outdir, chrom, length), encoding=one_hot_array, ids=ids)


if __name__ == '__main__':
    chrms = []
    for i in range(1, 23):
        chrms.append("chr{}".format(i))
    chrms.append("chrX")
    chrms.append("chrY")

    # Total length is 10090 which allows margin for shifting sequence to left and right
    one_side_length = 5045

    for chrm in chrms:
        main(chrm, one_side_length)
        print(chrm)

