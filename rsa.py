import torch
import numpy as np
import scipy.spatial
import scipy.stats
from typing import Dict, Any
import json

import torch
import numpy as np

n_attributes = 5
n_values = 3

def entropy_dict(freq_table):
    H = 0
    n = sum(v for v in freq_table.values())

    for m, freq in freq_table.items():
        p = freq_table[m] / n
        H += -p * np.log(p)
    return H / np.log(2)


def entropy(messages):
    from collections import defaultdict

    freq_table = defaultdict(float)

    for m in messages:
        m = _hashable_tensor(m)
        freq_table[m] += 1.0

    return entropy_dict(freq_table)


def _hashable_tensor(t):
    if isinstance(t, tuple):
        return t
    if isinstance(t, int):
        return t

    try:
        t = t.item()
    except:
        t = tuple(t.view(-1).tolist())
    return t


def mutual_info(xs, ys):
    e_x = entropy(xs)
    e_y = entropy(ys)

    xys = []

    for x, y in zip(xs, ys):
        xy = (_hashable_tensor(x), _hashable_tensor(y))
        xys.append(xy)

    e_xy = entropy(xys)

    return e_x + e_y - e_xy


def one_hot(a):
    ncols = a.max() + 1
    out = np.zeros((a.size, ncols), dtype=np.uint8)
    out[np.arange(a.size), a.ravel()] = 1
    out.shape = a.shape + (ncols,)
    return out

def information_gap_representation(meanings, representations):
    gaps = torch.zeros(representations.size(1))
    non_constant_positions = 0.0

    for j in range(representations.size(1)):
        symbol_mi = []
        h_j = None
        for i in range(meanings.size(1)):
            x, y = meanings[:, i], representations[:, j]
            info = mutual_info(x, y)
            symbol_mi.append(info)

            if h_j is None:
                h_j = entropy(y)

        symbol_mi.sort(reverse=True)

        if h_j > 0.0:
            gaps[j] = (symbol_mi[0] - symbol_mi[1]) / h_j
            non_constant_positions += 1

    score = gaps.sum() / non_constant_positions
    return score.item()


def histogram(strings, vocab_size):
    batch_size = strings.size(0)

    histogram = torch.zeros(batch_size, vocab_size, device=strings.device)

    for v in range(vocab_size):
        histogram[:, v] = strings.eq(v).sum(dim=-1)

    return histogram


def information_gap_vocab(n_attributes, n_values,  dataset, sender, device, vocab_size):
    attributes, strings, _meanings = ask_sender(
        n_attributes, n_values, dataset, sender, device)

    histograms = histogram(strings, vocab_size)
    return information_gap_representation(attributes, histograms[:, 1:])


def edit_dist(_list):
    distances = []
    count = 0
    for i, el1 in enumerate(_list[:-1]):
        for j, el2 in enumerate(_list[i+1:]):
            count += 1
            # Normalized edit distance (same in our case as length is fixed)
            distances.append(editdistance.eval(el1, el2) / len(el1))
    return distances


def cosine_dist(_list):
    distances = []
    for i, el1 in enumerate(_list[:-1]):
        for j, el2 in enumerate(_list[i+1:]):
            distances.append(spatial.distance.cosine(el1, el2))
    return distances


def representation_similarity_analysis(
    test_images,
    test_metadata,
    generated_messages,
    hidden_sender,
    hidden_receiver,
    vocab_size,
    samples=5000,
    tre=False,
):
    """
    Calculates RSA scores of the two agents (ρS/R),
    and of each agent with the input (ρS/I and ρR/I),
    and Topological Similarity between metadata/generated messages
    where S refers to Sender,R to Receiver,I to input.
    Args:
        test_set: encoded test set metadata info describing the image
        generated_messages: generated messages output from eval on test
        hidden_sender: encoded representation in sender
        hidden_receiver: encoded representation in receiver
        vocab_size: size of vocabulary
        samples (int, optional): default 5000 - number of pairs to sample
        tre (bool, optional): default False - whether to also calculate pseudo-TRE
    @TODO move to metrics repo
    """
    # one hot encode messages by taking padding into account and transforming to one hot
    messages = one_hot(generated_messages)

    assert test_metadata.shape[0] == messages.shape[0]

    sim_image_features = np.zeros(samples)
    sim_metadata = np.zeros(samples)
    sim_messages = np.zeros(samples)
    sim_hidden_sender = np.zeros(samples)
    sim_hidden_receiver = np.zeros(samples)

    for i in range(samples):
        rnd = np.random.choice(len(test_metadata), 2, replace=False)
        s1, s2 = rnd[0], rnd[1]

        sim_image_features[i] = scipy.spatial.distance.cosine(
            test_images[s1].detach(), test_images[s2].detach()
        )
        sim_metadata[i] = scipy.spatial.distance.cosine(
            test_metadata[s1], test_metadata[s2]
        )

        sim_messages[i] = scipy.spatial.distance.cosine(
            messages[s1].flatten(), messages[s2].flatten()
        )
        sim_hidden_sender[i] = scipy.spatial.distance.cosine(
            hidden_sender[s1].flatten(), hidden_sender[s2].flatten()
        )
        sim_hidden_receiver[i] = scipy.spatial.distance.cosine(
            hidden_receiver[s1].flatten(), hidden_receiver[s2].flatten()
        )

    rsa_sr = scipy.stats.pearsonr(sim_hidden_sender, sim_hidden_receiver)[0]
    rsa_si = scipy.stats.pearsonr(sim_hidden_sender, sim_image_features)[0]
    rsa_ri = scipy.stats.pearsonr(sim_hidden_receiver, sim_image_features)[0]
    topological_similarity = scipy.stats.pearsonr(sim_messages, sim_metadata)[0]

    dis_samples = min(10*samples, len(test_metadata))
    rnd = np.random.choice(len(test_metadata), dis_samples, replace=False)
    attributes = torch.Tensor(test_metadata[rnd]).view(-1, n_attributes, n_values).argmax(dim=-1)
    # strings = messages.view(messages.size(0), -1).detach()
    strings = torch.Tensor(messages[rnd]).argmax(dim=-1)

    positional_disent = information_gap_representation(attributes, strings)
    histograms = histogram(strings, vocab_size)
    bos_disent = information_gap_representation(attributes, histograms[:, 1:])


    if tre:
        pseudo_tre = np.linalg.norm(sim_metadata - sim_messages, ord=1)
        return rsa_sr, rsa_si, rsa_ri, topological_similarity, pseudo_tre
    else:
        return rsa_sr, rsa_si, rsa_ri, topological_similarity, positional_disent, bos_disent


# def representation_similarity_analysis_freq(
#     test_images,
#     test_metadata,
#     generated_messages,
#     hidden_sender,
#     hidden_receiver,
#     samples=5000,
#     tre=False,
# ):
#     # one hot encode messages by taking padding into account and transforming to one hot
#     messages = one_hot(generated_messages)

#     assert test_metadata.shape[0] == messages.shape[0]
#     assert len(test_images) == test_metadata.shape[0], '{} != {}'.format(len(test_images), test_metadata.shape[0])
#     assert test_metadata.shape[0] == hidden_sender.shape[0]
#     assert test_metadata.shape[0] == hidden_receiver.shape[0]

#     # color_shape
#     shape_color_to_indices = {
#         '0_0':[], '0_1':[], '0_2':[],
#         '1_0':[], '1_1':[], '1_2':[],
#         '2_0':[], '2_1':[], '2_2':[],
#     }

#     # Get indices per shape-color combination
#     for i,m in enumerate(test_metadata):
#         if m[0] == 1 and m[3+0] == 1:
#             shape_color_to_indices['0_0'].append(i)
#         elif m[0] == 1 and m[3+1] == 1:
#             shape_color_to_indices['0_1'].append(i)
#         elif m[0] == 1 and m[3+2] == 1:
#             shape_color_to_indices['0_2'].append(i)
#         elif m[1] == 1 and m[3+0] == 1:
#             shape_color_to_indices['2_0'].append(i)
#         elif m[1] == 1 and m[3+1] == 1:
#             shape_color_to_indices['2_1'].append(i)
#         elif m[1] == 1 and m[3+2] == 1:
#             shape_color_to_indices['2_2'].append(i)
#         elif m[2] == 1 and m[3+0] == 1:
#             shape_color_to_indices['2_0'].append(i)
#         elif m[2] == 1 and m[3+1] == 1:
#             shape_color_to_indices['2_1'].append(i)
#         elif m[2] == 1 and m[3+2] == 1:
#             shape_color_to_indices['2_2'].append(i)


#     rsa_sr_dict = {k:-1 for k in shape_color_to_indices.keys()}
#     rsa_si_dict = {k:-1 for k in shape_color_to_indices.keys()}
#     rsa_ri_dict = {k:-1 for k in shape_color_to_indices.keys()}
#     topological_similarity_dict = {k:-1 for k in shape_color_to_indices.keys()}


#     for key, indices in shape_color_to_indices.items():

#         if len(indices) == 0:
#             continue

#         sim_image_features = np.zeros(samples)
#         sim_metadata = np.zeros(samples)
#         sim_messages = np.zeros(samples)
#         sim_hidden_sender = np.zeros(samples)
#         sim_hidden_receiver = np.zeros(samples)

#         for i in range(samples):
#             rnd = np.random.choice(indices, 2, replace=False)
#             s1, s2 = rnd[0], rnd[1]

#             sim_image_features[i] = scipy.spatial.distance.cosine(
#                 test_images[s1], test_images[s2]
#             )
#             sim_metadata[i] = scipy.spatial.distance.cosine(
#                 test_metadata[s1], test_metadata[s2]
#             )

#             sim_messages[i] = scipy.spatial.distance.cosine(
#                 messages[s1].flatten(), messages[s2].flatten()
#             )
#             sim_hidden_sender[i] = scipy.spatial.distance.cosine(
#                 hidden_sender[s1].flatten(), hidden_sender[s2].flatten()
#             )
#             sim_hidden_receiver[i] = scipy.spatial.distance.cosine(
#                 hidden_receiver[s1].flatten(), hidden_receiver[s2].flatten()
#             )

#         rsa_sr = scipy.stats.pearsonr(sim_hidden_sender, sim_hidden_receiver)[0]
#         rsa_si = scipy.stats.pearsonr(sim_hidden_sender, sim_image_features)[0]
#         rsa_ri = scipy.stats.pearsonr(sim_hidden_receiver, sim_image_features)[0]
#         topological_similarity = scipy.stats.pearsonr(sim_messages, sim_metadata)[0]

#         rsa_sr_dict[key] = rsa_sr
#         rsa_si_dict[key] = rsa_si
#         rsa_ri_dict[key] = rsa_ri
#         topological_similarity_dict[key] = topological_similarity


#     freqs = {k:len(v) for k,v in shape_color_to_indices.items()}

#     return freqs, rsa_sr_dict, rsa_si_dict, rsa_ri_dict, topological_similarity_dict
