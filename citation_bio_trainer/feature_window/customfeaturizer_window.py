import numpy as np

def get_custom_feats(data_ls):
    # feat_arr = np.zeros((len(data_ls), maxlen, 5), dtype='int8') #[[[0]* 5]* maxlen]* len(data_ls)
    data_feat = []
    for ind in range(len(data_ls)):
        sent_ls = data_ls[ind]
        sent_len = len(sent_ls)
        sent_feat = []
        for i in range(sent_len):
            token_feat = [0] * 5
            if i == 0:
                token_feat[0] = 1
            elif (i == (sent_len - 1)) or (i == (sent_len - 2)):
                pass
            elif sent_ls[i - 1] == "\n":
                if (
                    sent_ls[i].isdigit()
                    and len(sent_ls[i]) <= 2
                    and sent_ls[i + 1] in (".", ")")
                ):
                    token_feat[1] = 1
                elif (
                    sent_ls[i].isalpha()
                    and len(sent_ls[i]) == 1
                    and sent_ls[i + 1] in (".", ")")
                ):
                    token_feat[2] = 1
                elif (
                    sent_ls[i] in ("[", "(")
                    and (sent_ls[i + 1].isdigit() and len(sent_ls[i + 1]) <= 2)
                    and sent_ls[i + 2] in ("]", ")")
                ):
                    token_feat[3] = 1
                elif (
                    sent_ls[i] in ("[", "(")
                    and (sent_ls[i + 1].isalpha() and len(sent_ls[i + 1]) == 2)
                    and sent_ls[i + 2] in ("]", ")")
                ):
                    token_feat[4] = 1
            sent_feat.append(token_feat)
        data_feat.append(sent_feat)
    return data_feat


def pad_custom_feats(feats, num_feats, maxlen):
    for i in range(len(feats)):
        for j in range(len(feats[i])):
            if len(feats[i][j]) < maxlen:
                num_pads = maxlen - len(feats[i][j])
                feats[i][j] += [[0] * num_feats] * num_pads
    return feats
