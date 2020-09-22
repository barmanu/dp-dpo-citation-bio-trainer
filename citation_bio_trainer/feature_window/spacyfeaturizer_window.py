import spacy as sp
import os

os.system("python3 -m spacy download en_core_web_sm")
import numpy as np
import pandas as pd
from spacy.matcher import PhraseMatcher
import dask.dataframe as dd
from dask.multiprocessing import get
from spacy.tokenizer import Tokenizer


class SpacyFeaturizer_window(object):
    def __init__(self):
        self.spacy_vorn_attributes = [
            "DEP",
            "POS",
            "TAG",
            "ENT_TYPE",
        ]
        self.spacy_vocab_attributes = [
            "ORTH",
            "LOWER",
            "SHAPE",
            "LEMMA",
        ]
        self.spacy_num_attributes = [
            "IS_ALPHA",
            "IS_ASCII",
            "IS_DIGIT",
            "IS_LOWER",
            "IS_UPPER",
            "IS_TITLE",
            "IS_PUNCT",
            "IS_SPACE",
            "IS_STOP",
            "LIKE_NUM",
            "LIKE_URL",
            "LIKE_EMAIL",  # "LENGTH",
        ]
        self.GZT_LISTS = {
            "manuscript_affiliation": [
                "department of",
                "department for",
                "faculty of",
                "faculty for",
                "institute of",
                "institute for",
                "division for",
                "division of",
                "research group",
                "college of",
                "college for",
                "laboratory of",
                "laboratory for",
                "academy of",
                "academy for",
                "research center",
                "research centre",
                "reliability center",
                "centre of",
                "centre for",
                "research institute",
                "center of",
                "center for",
                "school of",
            ],
            "manuscript_author_degrees_before": [
                "CPT",
                "Captain",
                "Colonel (Ret)",
                "Colonel",
                "Dr.",
                "Dr",
                "Mag.",
                "Major",
                "Mr.",
                "Mr",
                "Ms.",
                "Pr.",
                "Prof.",
                "Prof",
                "Professor",
                "Profs.",
            ],
            "manuscript_section_heading": [
                "Abstract",
                "Reference",
                "Keywords",
                "Highlights",
                "Acknowledgments",
                "Acknowledgements",
                "Introduction",
                "Conclusions",
            ],
            "manuscript_keyword": ["keyword", "key word", "indexed terms"],
            "manuscript_tabfig": ["tab", "fig", "scheme", "box"],
            "manuscript_corresponds": ["correspond"],
        }
        self.num_features = True
        self.cat_features = False
        self.gzt_features = False
        self.nlp = nlp
        self.nlp.tokenizer = Tokenizer(nlp.vocab)
    def get_array_from_df_combined(self, df):

        rows = df.text.tolist()
        rows = [
            t.replace("\n", "क") for t in rows
        ]  ## as spacy cannot handle consecutive newlines
        sep = " "
        text = sep.join(rows)

        if nlp.max_length < len(text):
            nlp.max_length = 1 + len(text)
        rows_token_indexes_in_text = list(
            np.cumsum([len(a) for a in nlp.tokenizer.pipe(rows)])
        )
        total_tokens = rows_token_indexes_in_text.pop()
        # assert(total_tokens == 1 + len(list(nlp.tokenizer.pipe([text]))[0]) ) - len(rows_token_indexes_in_text)

        def set_custom_boundaries(doc):
            for token_index in rows_token_indexes_in_text:
                doc[token_index].is_sent_start = True
            return doc

        nlp.add_pipe(set_custom_boundaries, before="tagger")
        doc = nlp(text)

        result_df = pd.DataFrame([], columns=["spacy_bin", "spacy_cat"])

        ## bigint features
        if self.cat_features:
            spacy_bigint_attributes = (
                self.spacy_vorn_attributes + self.spacy_vocab_attributes
            )
            tokens_features_bigint = doc.to_array(
                spacy_bigint_attributes
            ).astype("object")
            for i in range(tokens_features_bigint.shape[0]):
                for j in range(tokens_features_bigint.shape[1]):
                    tokens_features_bigint[i][j] = nlp.vocab[
                        tokens_features_bigint[i][j]
                    ].text
            tokens_features_big = np.split(
                tokens_features_bigint, rows_token_indexes_in_text
            )
            result_df["spacy_cat"] = tokens_features_big

        small_feat_list = []

        ## smallint features
        if self.num_features:
            tokens_features_smallint = doc.to_array(
                self.spacy_num_attributes
            ).astype("int8")
            small_feat_list.append(tokens_features_smallint)

        ## gzt features
        if self.gzt_features:
            phrase_matcher = PhraseMatcher(nlp.vocab)
            gzt_attributes = [a.upper() for a in list(self.GZT_LISTS.keys())]
            gzt_index_map = dict()
            for i, a in enumerate(gzt_attributes):
                gzt_index_map[nlp.vocab.strings[a]] = i

            gzt_patterns = list()
            for label, terms in self.GZT_LISTS.items():
                patterns = [nlp.make_doc(text) for text in terms]
                phrase_matcher.add(label.upper(), None, *patterns)

            gzt_matches = phrase_matcher(doc)

            token_gzt_features = np.zeros(
                shape=[len(doc), len(gzt_attributes)], dtype="int8"
            )

            for match_id, start, end in gzt_matches:
                gzt_attribute_index = gzt_index_map[match_id]

                span = doc[start:end]
                if span is not None:
                    for token in span:
                        # print(token.i, token)
                        token_gzt_features[token.i, gzt_attribute_index] = 1

            small_feat_list.append(token_gzt_features)

        # tokens_features_small = np.concatenate((tokens_features_smallint, token_gzt_features), axis=1)
        if len(small_feat_list) > 0:
            tokens_features_small = np.hstack(small_feat_list)
            tokens_features_small = np.split(
                tokens_features_small, rows_token_indexes_in_text
            )
            result_df["spacy_bin"] = tokens_features_small

        return result_df

    def get_spacy_dask(self, df, blocksize=100):
        partitions = 1
        if int(df.shape[0] / blocksize) > 1:
            partitions = int(df.shape[0] / blocksize)
        print("total partitions = %d" % partitions)
        spacy_df = (
            dd.from_pandas(df, npartitions=partitions)
            .map_partitions(
                self.get_array_from_df_combined,
                meta=[("spacy_bin", "object"), ("spacy_cat", "object")],
            )
            .compute(scheduler="processes")
        )
        return spacy_df


def pad_spacy_feats(feats, num_feats, maxlen):
    for i in range(len(feats)):
        if len(feats[i]) < maxlen:
            num_pads = maxlen - len(feats[i])
            feats[i] += [[0] * num_feats] * num_pads
    return feats
