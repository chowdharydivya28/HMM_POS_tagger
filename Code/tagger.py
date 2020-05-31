import numpy as np

from util import accuracy
from hmm import HMM


# TODO:
def model_training(train_data, tags):
    """
    Train HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - tags: (1*num_tags) a list of POS tags

    Returns:
    - model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
    """
    model = None
    ###################################################

    # all_words = []
    # all_transitions = {}
    # all_states_words_tuples = {}
    # unique_0_tags_cnt = {t: 0.0 for t in tags}
    # for i in train_data:
    #     all_words = np.unique(np.append(all_words, i.words))
    #     unique_0_tags_cnt[i.tags[0]] = unique_0_tags_cnt[i.tags[0]] + 1
    #     for l in range(0, len(i.tags) - 1, 1):
    #         if (i.tags[l], i.tags[l + 1]) in list(all_transitions.keys()):
    #             all_transitions[(i.tags[l], i.tags[l + 1])] = all_transitions[(i.tags[l], i.tags[l + 1])] + 1.0
    #         else:
    #             all_transitions[(i.tags[l], i.tags[l + 1])] = 1.0
    #     for k in range(0, len(i.words)):
    #         if (i.tags[k], i.words[k]) in list(all_states_words_tuples.keys()):
    #             all_states_words_tuples[(i.tags[k], i.words[k])] = all_states_words_tuples[(i.tags[k], i.words[k])] + \
    #                                                                1.0
    #         else:
    #             all_states_words_tuples[(i.tags[k], i.words[k])] = 1.0

    all_words = set()
    unique_0_tags_cnt = {t: 0.0 for t in tags}
    for i in train_data:
        unique_0_tags_cnt[i.tags[0]] = unique_0_tags_cnt[i.tags[0]] + 1
        for k in i.words:
            if k not in all_words:
                all_words.add(k)
    obs_dict = dict(zip(sorted(all_words), np.arange(0, len(all_words), 1)))
    state_dict = dict(zip(tags, np.arange(0, len(tags), 1)))
    pi = np.array(list(unique_0_tags_cnt.values())) / len(train_data)
    S = len(list(state_dict.values()))
    L = len(list(obs_dict.values()))
    A = np.zeros([S, S])
    B = np.zeros([S, L])
    for i in train_data:
        for l in range(len(i.tags)):
            if l != len(i.tags)-1:
                A[state_dict[i.tags[l]], state_dict[i.tags[l+1]]] += 1.0
            B[state_dict[i.tags[l]], obs_dict[i.words[l]]] += 1.0
    A = A / A.sum(axis=1, keepdims=True)
    B = B / B.sum(axis=1, keepdims=True)
    model = HMM(pi, A, B, obs_dict, state_dict)
    # print("A", A)
    # print("A shape:", A.shape)
    # print("B", B)
    # print("B shape:", B.shape)
    # print("pi", pi)
    # print("state dict", state_dict)
    # print("obs dict", obs_dict)

    # all_words = []
    # unique_0_tags_cnt = {t: 0.0 for t in tags}
    # state_dict = dict(zip(tags, np.arange(0, len(tags), 1)))
    # A = np.zeros([len(tags), len(tags)])
    # B_dict = {t: {} for t in list(state_dict.values())}
    # for i in train_data:
    #     all_words = np.unique(np.append(all_words, i.words))
    #     unique_0_tags_cnt[i.tags[0]] = unique_0_tags_cnt[i.tags[0]] + 1
    #     tag_seq = [state_dict[k] for k in i.tags]
    #     for l in range(len(tag_seq)-1):
    #         A[tag_seq[l], tag_seq[l+1]] = A[tag_seq[l], tag_seq[l+1]] + 1.0
    #         if i.words[l] not in list(B_dict[tag_seq[l]].keys()):
    #             B_dict[tag_seq[l]][i.words[l]] = 1.0
    #         else:
    #             B_dict[tag_seq[l]][i.words[l]] += 1.0
    # print("B dict", B_dict)
    # A = A / A.sum(axis=1, keepdims=True)
    # pi = np.array(list(unique_0_tags_cnt.values())) / len(train_data)
    # obs_dict = dict(zip(all_words, np.arange(0, len(all_words), 1)))
    # S = len(tags)
    # obs_symbols = len(obs_dict)
    # B = np.zeros([S, obs_symbols])
    # for s in range(S):
    #     word_idx = [obs_dict[k] for k in list(B_dict[s].keys())]
    #     word_counts = list(B_dict[s].values())
    #     B[s, :][word_idx] = word_counts
    # B = B / B.sum(axis=1, keepdims=True)
    # model = HMM(pi, A, B, obs_dict, state_dict)
    ###################################################
    return model


# TODO:
def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - model: an object of HMM class

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    # ###################################################
    # print("model A:", model.A.shape)
    # print("model B:", model.B.shape)
    # print("model pi:", model.pi.shape)
    new_col = np.repeat(10 ** (-6), len(model.state_dict.keys()))
    # print("new col:", new_col)
    for i in range(len(test_data)):
        for k in range(len(test_data[i].words)):
            if test_data[i].words[k] not in model.obs_dict.keys():
                model.B = np.c_[model.B, new_col]
                model.obs_dict[test_data[i].words[k]] = len(model.obs_dict.keys())
        tagging.append(model.viterbi(test_data[i].words))
    # print("final tagging :", tagging)
    ###################################################
    return tagging
