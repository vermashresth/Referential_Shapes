import torch
import numpy as np
import scipy.spatial
import scipy.stats


def one_hot(a):
    ncols = a.max() + 1
    out = np.zeros((a.size, ncols), dtype=np.uint8)
    out[np.arange(a.size), a.ravel()] = 1
    out.shape = a.shape + (ncols,)
    return out


def representation_similarity_analysis(
    test_images,
    test_metadata,
    generated_messages,
    hidden_sender,
    hidden_receiver,
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
            test_images[s1], test_images[s2]
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

    if tre:
        pseudo_tre = np.linalg.norm(sim_metadata - sim_messages, ord=1)
        return rsa_sr, rsa_si, rsa_ri, topological_similarity, pseudo_tre
    else:
        return rsa_sr, rsa_si, rsa_ri, topological_similarity


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