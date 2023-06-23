from load_data import *
import torch.nn.functional as F
import random
import tqdm as tqdm
from captum.attr import DeepLift
from torch.utils.data import TensorDataset, DataLoader
import os


def shift_sequence(array, shiftby=4, negative=False):
    # stochastically shift between 4/4 = 1 bp to 20/4 = 5 bps.
    linspace = np.array([shiftby]) #,8,12,16,20
    randn = int(np.random.choice(linspace, size=1, replace=False))

    if negative == False:
        return np.roll(array, randn)
    else:
        return np.roll(array, -randn)


def reverse_complement(array):
    array = np.flip(array, axis=0)
    result = []
    for base in array:
        # A > T
        if (base == [1., 0., 0., 0.]).all():
            result.append([0., 0., 0., 1.])
        # C > G
        elif (base == [0., 1., 0., 0.]).all():
            result.append([0., 0., 1., 0.])
        # G > C
        elif (base == [0., 0., 1., 0.]).all():
            result.append([0., 1., 0., 0.])
        # T > A
        elif (base == [0., 0., 0., 1.]).all():
            result.append([1., 0., 0., 0.])

    return result


def data_augment(st):
    shift_one_right = shift_sequence(st, 4)
    shift_one_right = shift_one_right[45:-45]
    shift_one_left = shift_sequence(st, 4, True)
    shift_one_left = shift_one_left[45:-45]
    return shift_one_right, shift_one_left


def get_gained_ids():
    return gained.id.values


def get_constitutive_ids():
    return constitutive.id.values

gained_ids = get_gained_ids()
constitutive_ids = get_constitutive_ids()

def get_gained_sites():
    X = []
    X_prime = []
    for i, uid in enumerate(ids1):
        if uid in gained_ids:
            s = encoding1[i]
            
            shift_one_right, shift_one_left = data_augment(s)
            X.append(s[45:-45])

            X_prime.append(shift_one_right)
            X_prime.append(shift_one_left)
            
#     for i, uid in enumerate(ids2):
#         if uid in gained_ids:
#             s = encoding2[i]
            
#             shift_one_right, shift_one_left = data_augment(s)
#             X.append(s[45:-45])

#             X_prime.append(shift_one_right)
#             X_prime.append(shift_one_left)

#     for i, uid in enumerate(ids3):
#         if uid in gained_ids:
#             s = encoding3[i]
            
#             shift_one_right, shift_one_left = data_augment(s)
#             X.append(s[45:-45])

#             X_prime.append(shift_one_right)
#             X_prime.append(shift_one_left)
            
#     for i, uid in enumerate(ids4):
#         if uid in gained_ids:
#             s = encoding4[i]
            
#             shift_one_right, shift_one_left = data_augment(s)
#             X.append(s[45:-45])

#             X_prime.append(shift_one_right)
#             X_prime.append(shift_one_left)

#     for i, uid in enumerate(ids5):
#         if uid in gained_ids:
#             s = encoding5[i]
            
#             shift_one_right, shift_one_left = data_augment(s)
#             X.append(s[45:-45])

#             X_prime.append(shift_one_right)
#             X_prime.append(shift_one_left)

#     for i, uid in enumerate(ids6):
#         if uid in gained_ids:
#             s = encoding6[i]
            
#             shift_one_right, shift_one_left = data_augment(s)
#             X.append(s[45:-45])

#             X_prime.append(shift_one_right)
#             X_prime.append(shift_one_left)

#     for i, uid in enumerate(ids7):
#         if uid in gained_ids:
#             s = encoding7[i]
            
#             shift_one_right, shift_one_left = data_augment(s)
#             X.append(s[45:-45])

#             X_prime.append(shift_one_right)
#             X_prime.append(shift_one_left)

#     for i, uid in enumerate(ids8):
#         if uid in gained_ids:
#             s = encoding8[i]
            
#             shift_one_right, shift_one_left = data_augment(s)
#             X.append(s[45:-45])

#             X_prime.append(shift_one_right)
#             X_prime.append(shift_one_left)
            
#     for i, uid in enumerate(ids9):
#         if uid in gained_ids:
#             s = encoding9[i]
            
#             shift_one_right, shift_one_left = data_augment(s)
#             X.append(s[45:-45])

#             X_prime.append(shift_one_right)
#             X_prime.append(shift_one_left)

#     for i, uid in enumerate(ids10):
#         if uid in gained_ids:
#             s = encoding10[i]
            
#             shift_one_right, shift_one_left = data_augment(s)
#             X.append(s[45:-45])

#             X_prime.append(shift_one_right)
#             X_prime.append(shift_one_left)

#     for i, uid in enumerate(ids11):
#         if uid in gained_ids:
#             s = encoding11[i]
            
#             shift_one_right, shift_one_left = data_augment(s)
#             X.append(s[45:-45])

#             X_prime.append(shift_one_right)
#             X_prime.append(shift_one_left)

#     for i, uid in enumerate(ids12):
#         if uid in gained_ids:
#             s = encoding12[i]
            
#             shift_one_right, shift_one_left = data_augment(s)
#             X.append(s[45:-45])

#             X_prime.append(shift_one_right)
#             X_prime.append(shift_one_left)

#     for i, uid in enumerate(ids13):
#         if uid in gained_ids:
#             s = encoding13[i]
            
#             shift_one_right, shift_one_left = data_augment(s)
#             X.append(s[45:-45])

#             X_prime.append(shift_one_right)
#             X_prime.append(shift_one_left)

#     for i, uid in enumerate(ids14):
#         if uid in gained_ids:
#             s = encoding14[i]
            
#             shift_one_right, shift_one_left = data_augment(s)
#             X.append(s[45:-45])

#             X_prime.append(shift_one_right)
#             X_prime.append(shift_one_left)

#     for i, uid in enumerate(ids15):
#         if uid in gained_ids:
#             s = encoding15[i]
            
#             shift_one_right, shift_one_left = data_augment(s)
#             X.append(s[45:-45])

#             X_prime.append(shift_one_right)
#             X_prime.append(shift_one_left)

#     for i, uid in enumerate(ids16):
#         if uid in gained_ids:
#             s = encoding16[i]
            
#             shift_one_right, shift_one_left = data_augment(s)
#             X.append(s[45:-45])

#             X_prime.append(shift_one_right)
#             X_prime.append(shift_one_left)

#     for i, uid in enumerate(ids17):
#         if uid in gained_ids:
#             s = encoding17[i]
            
#             shift_one_right, shift_one_left = data_augment(s)
#             X.append(s[45:-45])

#             X_prime.append(shift_one_right)
#             X_prime.append(shift_one_left)

#     for i, uid in enumerate(ids18):
#         if uid in gained_ids:
#             s = encoding18[i]
            
#             shift_one_right, shift_one_left = data_augment(s)
#             X.append(s[45:-45])

#             X_prime.append(shift_one_right)
#             X_prime.append(shift_one_left)

#     for i, uid in enumerate(ids19):
#         if uid in gained_ids:
#             s = encoding19[i]
            
#             shift_one_right, shift_one_left = data_augment(s)
#             X.append(s[45:-45])

#             X_prime.append(shift_one_right)
#             X_prime.append(shift_one_left)

#     for i, uid in enumerate(ids20):
#         if uid in gained_ids:
#             s = encoding20[i]
            
#             shift_one_right, shift_one_left = data_augment(s)
#             X.append(s[45:-45])

#             X_prime.append(shift_one_right)
#             X_prime.append(shift_one_left)

#     for i, uid in enumerate(ids21):
#         if uid in gained_ids:
#             s = encoding21[i]
            
#             shift_one_right, shift_one_left = data_augment(s)
#             X.append(s[45:-45])

#             X_prime.append(shift_one_right)
#             X_prime.append(shift_one_left)

#     for i, uid in enumerate(ids22):
#         if uid in gained_ids:
#             s = encoding22[i]
            
#             shift_one_right, shift_one_left = data_augment(s)
#             X.append(s[45:-45])

#             X_prime.append(shift_one_right)
#             X_prime.append(shift_one_left)

#     for i, uid in enumerate(idsX):
#         if uid in gained_ids:
#             s = encodingX[i]
            
#             shift_one_right, shift_one_left = data_augment(s)
#             X.append(s[45:-45])

#             X_prime.append(shift_one_right)
#             X_prime.append(shift_one_left)

#     for i, uid in enumerate(idsY):
#         if uid in gained_ids:
#             s = encodingY[i]
            
#             shift_one_right, shift_one_left = data_augment(s)
#             X.append(s[45:-45])

#             X_prime.append(shift_one_right)
#             X_prime.append(shift_one_left)
            
            
#     try:
#         X = np.concatenate([X, X_prime])
#     except:
#         pass
    return np.array(X)
    

def get_constitutive_sites():
    X_ = []

    for i, uid in enumerate(ids1):
        if uid in constitutive_ids:
            X_.append(encoding1[i][45:-45])

#     for i, uid in enumerate(ids2):
#         if uid in constitutive_ids:
#             X_.append(encoding2[i][45:-45])

#     for i, uid in enumerate(ids3):
#         if uid in constitutive_ids:
#             X_.append(encoding3[i][45:-45])

#     for i, uid in enumerate(ids4):
#         if uid in constitutive_ids:
#             X_.append(encoding4[i][45:-45])

#     for i, uid in enumerate(ids5):
#         if uid in constitutive_ids:
#             X_.append(encoding5[i][45:-45])

#     for i, uid in enumerate(ids6):
#         if uid in constitutive_ids:
#             X_.append(encoding6[i][45:-45])

#     for i, uid in enumerate(ids7):
#         if uid in constitutive_ids:
#             X_.append(encoding7[i][45:-45])

#     for i, uid in enumerate(ids8):
#         if uid in constitutive_ids:
#             X_.append(encoding8[i][45:-45])

#     for i, uid in enumerate(ids9):
#         if uid in constitutive_ids:
#             X_.append(encoding9[i][45:-45])

#     for i, uid in enumerate(ids10):
#         if uid in constitutive_ids:
#             X_.append(encoding10[i][45:-45])

#     for i, uid in enumerate(ids11):
#         if uid in constitutive_ids:
#             X_.append(encoding11[i][45:-45])

#     for i, uid in enumerate(ids12):
#         if uid in constitutive_ids:
#             X_.append(encoding12[i][45:-45])

#     for i, uid in enumerate(ids13):
#         if uid in constitutive_ids:
#             X_.append(encoding13[i][45:-45])

#     for i, uid in enumerate(ids14):
#         if uid in constitutive_ids:
#             X_.append(encoding14[i][45:-45])

#     for i, uid in enumerate(ids15):
#         if uid in constitutive_ids:
#             X_.append(encoding15[i][45:-45])

#     for i, uid in enumerate(ids16):
#         if uid in constitutive_ids:
#             X_.append(encoding16[i][45:-45])

#     for i, uid in enumerate(ids17):
#         if uid in constitutive_ids:
#             X_.append(encoding17[i][45:-45])

#     for i, uid in enumerate(ids18):
#         if uid in constitutive_ids:
#             X_.append(encoding18[i][45:-45])

#     for i, uid in enumerate(ids19):
#         if uid in constitutive_ids:
#             X_.append(encoding19[i][45:-45])

#     for i, uid in enumerate(ids20):
#         if uid in constitutive_ids:
#             X_.append(encoding20[i][45:-45])

#     for i, uid in enumerate(ids21):
#         if uid in constitutive_ids:
#             X_.append(encoding21[i][45:-45])

#     for i, uid in enumerate(ids22):
#         if uid in constitutive_ids:
#             X_.append(encoding22[i][45:-45])

#     for i, uid in enumerate(idsX):
#         if uid in constitutive_ids:
#             X_.append(encodingX[i][45:-45])

#     for i, uid in enumerate(idsY):
#         if uid in constitutive_ids:
#             X_.append(encodingY[i][45:-45])

    return np.array(X_)


def get_data():
    X = get_gained_sites()
    X_ = get_constitutive_sites()

    gained_labels = np.ones(len(X))
    constitutive_labels = np.zeros(len(X_))

    X = np.concatenate([X, X_])
    y = np.concatenate([gained_labels, constitutive_labels])
    return X, y


def build_dataloader(X, y, z=None, batch_size=32):
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)
    if z is not None:
        dataset = TensorDataset(X, y, z)
    else:
        dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


def calculate_loss(inputs, targets):
    BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets)
    return BCE_loss


def get_constitutive_baseline(data, percentage=0.80):
    r_ind = random.sample(range(1, len(data)), int(len(data) * percentage)-1)
    sites = np.array(data)[r_ind]
    mean = sites.mean(axis=0)
    output = torch.FloatTensor(mean).reshape(1, 1, 4, SEQLEN).to(device)
    return output


def correct_sample(model, deeplift_input, target_label):
    output = model(deeplift_input.reshape(-1, 1, 4, SEQLEN))
    am = torch.argmax(output)
    return (target_label == am).long().item()


def get_gained_sequence_ids():
    return gained_ids


def convert_listofints_to_letters(subseq):
    cur_letter = ''
    subseq = subseq.flatten()
    for k in range(0, len(subseq), 4):
        now = subseq[k:k + 4]
        if (now == torch.FloatTensor([1.0, 0.0, 0.0, 0.0])).all():
            cur_letter += 'A'
        elif (now == torch.FloatTensor([0.0, 1.0, 0.0, 0.0])).all():
            cur_letter += 'C'
        elif (now == torch.FloatTensor([0.0, 0.0, 1.0, 0.0])).all():
            cur_letter += 'G'
        elif (now == torch.FloatTensor([0.0, 0.0, 0.0, 1.0])).all():
            cur_letter += 'T'
        else:
            print(now)

    return cur_letter


def find_chrom_from_id(query_id):
    return df[df['id'] == query_id]['chr'].values[0]


def gate(idx, attributions, data):
    dl_weight_vec = attributions[idx].reshape(-1)
    x = data[idx]
    gated_values = []
    i = 0
    for base in x:
        nuc = dl_weight_vec[i:i + 4]
        gate = torch.matmul(torch.FloatTensor(base), nuc)
        gated_values.append(gate.data)
        i += 4
    gated_values = torch.FloatTensor(np.array(gated_values))
    return gated_values


def build_subsequences_individual(data,
                                  trained_model,
                                  dataloader,
                                  num_sites_to_use=num_gained_sites,
                                  length=SEQLEN,
                                  window_size=20,
                                  select_n_subsequences=10):
    """
    :param data: gained sites
    :param dataloader: train data loader
    :param length: sequence length
    :param window_size: scan size for each DeepLIFT score
    :param select_n_subsequences: number of subsequences per gained sequence
    :return: dataframe containing top subsequences: genomic coordinates, sequence ids, subsequences, DeepLIFT scores
    """

    bedfile = pd.DataFrame()
    incorrect_count = 0

    for i, (deeplift_input, label, site_id) in tqdm.tqdm(enumerate(dataloader), position=0, leave=True):
        bedfile_temp = pd.DataFrame()
        chrom = []
        start = []
        ends = []
        sequences = []
        scores = []
        unique_site_ids = []

        if not correct_sample(trained_model, deeplift_input, label):
            incorrect_count += 1

        chromosome = find_chrom_from_id(site_id.item())

        mid_position = binding_df.loc[site_id.item()]["mid_position"]
        start_left = mid_position - length // 2

        attribution = gate(idx=i, attributions=dl_attri, data=data[:num_sites_to_use])

        for cur in np.arange(0, length):
            end_position = cur + window_size
            if end_position > length:
                break
            numbases = deeplift_input.squeeze().long()[cur:cur+window_size]

            subsequence = convert_listofints_to_letters(numbases)

            chrom.append(chromosome)
            start.append(cur + start_left + 1)
            ends.append(cur + start_left + window_size)
            sequences.append(subsequence)

            lower_bound_cur = max(0, cur)
            upper_bound_cur = cur + window_size
            mean_att_score = attribution[lower_bound_cur:upper_bound_cur].mean().item()

            scores.append(mean_att_score)
            unique_site_ids.append(site_id.item())

        zipped = list(zip(chrom, start, ends, unique_site_ids, sequences, scores))
        bedfile_temp = bedfile_temp.append(pd.DataFrame(zipped,
                                                        columns=['chr', 'start', 'end', 'id', 'sequence', 'DeepLIFT']),
                                                        ignore_index=True)


        bedfile = pd.concat([bedfile, bedfile_temp.nlargest(select_n_subsequences, ['DeepLIFT'])])

        print("----", "Number of incorrectly classified sequences: ", incorrect_count, "----")
        return bedfile


def build_subsequences_aggregate(data,
                                 trained_model,
                                 dataloader,
                                 num_sites_to_use=num_gained_sites,
                                 length=SEQLEN,
                                 window_size=20):
    """
    :param dataloader: train data loader
    :param length: sequence length
    :param window_size: scan size for each DeepLIFT score
    :return: dataframe containing top subsequences: genomic coordinates, sequence ids, subsequences, DeepLIFT scores
    """
    bedfile = pd.DataFrame()
    chrom = []
    start = []
    ends = []
    sequences = []
    scores = []

    unique_site_ids = []
    incorrect_count = 0

    for i, (deeplift_input, label, site_id) in tqdm.tqdm(enumerate(dataloader), position=0, leave=True):
        if not correct_sample(trained_model, deeplift_input, label):
            incorrect_count += 1

        chromosome = find_chrom_from_id(site_id.item())

        mid_position = binding_df.loc[site_id.item()]["mid_position"]
        start_left = mid_position - length // 2

        attribution = gate(idx=i, attributions=dl_attri, data=data[:num_sites_to_use])

        for cur in np.arange(0, length):
            end_position = cur + window_size
            if end_position > length:
                break
            numbases = deeplift_input.squeeze().long()[cur:cur + window_size]

            subsequence = convert_listofints_to_letters(numbases)

            chrom.append(chromosome)
            start.append(cur + start_left + 1)
            ends.append(cur + start_left + window_size)
            sequences.append(subsequence)

            lower_bound_cur = max(0, cur)
            upper_bound_cur = cur + window_size
            mean_att_score = attribution[lower_bound_cur:upper_bound_cur].mean().item()

            scores.append(mean_att_score)
            unique_site_ids.append(site_id.item())

    print("----", incorrect_count, "----")
    zipped = list(zip(chrom, start, ends, unique_site_ids, sequences, scores))
    bedfile = bedfile.append(pd.DataFrame(zipped, columns=['chr', 'start', 'end', 'id', 'sequence', 'DeepLIFT']),
                             ignore_index=True)

    return bedfile



def run_deeplift(model, data, labels, epoch, num_gained_sites, const_baseline):
    """
    Calculates both positive and negative DeepLIFT constirubions for output neuron 0 and 1.
    :param model: trained model
    :param epoch: epoch number
    :param num_gained_sites: number of gained sites to use
    :param const_baseline: reference baseline to use for DeepLIFT
    :return: DeepLIFT attributions
    """
    print("Running DeepLIFT")
    model.eval()

    deeplift_dataloader = build_dataloader(np.array(data[:num_gained_sites]),
                                           labels[:num_gained_sites],
                                           batch_size=1)

    dl = DeepLift(model)
    attributions_0 = []
    attributions_1 = []

    for _, (dl_input, _) in tqdm.tqdm(enumerate(deeplift_dataloader), position=0, leave=True):
        dl_input = dl_input.reshape(-1, 1, 4, SEQLEN).to(device)

        attribution_0 = dl.attribute(inputs=dl_input, baselines=const_baseline, target=0)
        attribution_0 = attribution_0.squeeze()

        attributions_0.append(attribution_0)

        attribution_1 = dl.attribute(inputs=dl_input, baselines=const_baseline, target=1)
        attribution_1 = attribution_1.squeeze()

        attributions_1.append(attribution_1)

    # Create folder for attributions
    if not os.path.exists("attributions"):
        os.makedirs("attributions")

    torch.save(torch.FloatTensor(np.array([a.cpu().detach().numpy() for a in attributions_0])),
        "attributions/gained_0_{}.pt".format(epoch))

    torch.save(torch.FloatTensor(np.array([a.cpu().detach().numpy() for a in attributions_1])),
        "attributions/gained_1_{}.pt".format(epoch))
