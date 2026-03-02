from __future__ import annotations

import itertools
import math
import random
import re
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

from utils import read_json

glove_dim = 300
glove_path = f"data/glove.6B.{glove_dim}d.txt"

possible_segments = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
for seg in itertools.combinations(range(6), 2):
    possible_segments.append(seg)

vocab_file = "data/vocab_glove_complete.txt"


def word_tokenize(sentence: str) -> List[str]:
    sent = sentence.lower()
    sent = re.sub(r"[^A-Za-z0-9\s]+", " ", sent)
    return sent.split()


def sentences_to_words(sentences: Sequence[str]) -> List[str]:
    words: List[str] = []
    for sentence in sentences:
        words.extend(word_tokenize(str(sentence.lower())))
    return words


class glove_embedding(object):
    """Loads GloVe vectors into a dictionary keyed by token."""

    def __init__(self, glove_file: str = glove_path):
        glove_dict: Dict[str, np.ndarray] = {}
        with open(glove_file, "r", encoding="utf-8") as handle:
            for line in handle:
                pieces = line.rstrip("\n").split(" ")
                if len(pieces) != glove_dim + 1:
                    continue
                word = pieces[0]
                vec = np.asarray(pieces[1:], dtype=np.float32)
                glove_dict[word] = vec
        self.glove_dict = glove_dict
        self.glove_words = list(glove_dict.keys())


class zero_language_vector(object):
    def __init__(self, data):
        self.dim = glove_dim

    def get_vector_dim(self) -> int:
        return self.dim

    def get_vocab_size(self) -> int:
        return 0

    def preprocess(self, data):
        embedding = np.zeros((self.get_vector_dim(),), dtype=np.float32)
        for datum in data:
            datum["language_input"] = embedding
            datum["gt"] = (datum["gt"][0], datum["gt"][1])
        return data


class recurrent_language(object):
    def get_vocab_size(self) -> int:
        return len(self.vocab_dict.keys())

    def preprocess_sentence(self, words: Sequence[str]) -> np.ndarray:
        vector_dim = self.get_vector_dim()
        sentence_mat = np.zeros((len(words), vector_dim), dtype=np.float32)
        count_words = 0
        for word in words:
            if word in self.vocab_dict:
                sentence_mat[count_words, :] = self.vocab_dict[word]
                count_words += 1
            elif "<unk>" in self.vocab_dict:
                sentence_mat[count_words, :] = self.vocab_dict["<unk>"]
                count_words += 1
        sentence_mat = sentence_mat[:count_words]
        return sentence_mat


class recurrent_word(recurrent_language):
    def __init__(self, data):
        self.data = data
        with open(vocab_file, "r", encoding="utf-8") as handle:
            vocab = [line.strip() for line in handle]
        if "<unk>" not in vocab:
            vocab.append("<unk>")
        self.vocab_dict = {word: idx for idx, word in enumerate(vocab)}

    def get_vector_dim(self) -> int:
        return 1


class recurrent_embedding(recurrent_language):
    def read_embedding(self):
        print("Reading glove embedding")
        self.embedding = glove_embedding(glove_path)

    def get_vector_dim(self) -> int:
        return glove_dim

    def __init__(self, data):
        self.read_embedding()
        self.data = data
        with open(vocab_file, "r", encoding="utf-8") as handle:
            vocab = [line.strip() for line in handle]
        if "<unk>" in vocab:
            vocab.remove("<unk>")

        vocab_dict: Dict[str, np.ndarray] = {}
        for word in vocab:
            if word in self.embedding.glove_dict:
                vocab_dict[word] = self.embedding.glove_dict[word]
            else:
                print(f"{word} not in glove embedding")
        self.vocab_dict = vocab_dict

    def preprocess(self, data):
        for datum in data:
            datum["language_input"] = sentences_to_words([datum["description"]])
        return data

    def get_vocab_dict(self):
        return self.vocab_dict


def feature_process_base(start: int, end: int, features: np.ndarray) -> np.ndarray:
    return np.mean(features[start : end + 1, :], axis=0)


def feature_process_norm(start: int, end: int, features: np.ndarray) -> np.ndarray:
    base_feature = np.mean(features[start : end + 1, :], axis=0)
    return base_feature / (np.linalg.norm(base_feature) + 0.00001)


def feature_process_context(start: int, end: int, features: np.ndarray) -> np.ndarray:
    feature_dim = features.shape[1]
    full_feature = np.zeros((feature_dim * 2,), dtype=np.float32)
    if np.sum(features[5, :]) > 0:
        full_feature[:feature_dim] = feature_process_norm(0, 6, features)
    else:
        full_feature[:feature_dim] = feature_process_norm(0, 5, features)
    full_feature[feature_dim : feature_dim * 2] = feature_process_norm(start, end, features)
    return full_feature


feature_process_dict = {
    "feature_process_base": feature_process_base,
    "feature_process_norm": feature_process_norm,
    "feature_process_context": feature_process_context,
}


class extractData(object):
    """General class to extract data."""

    def increment(self):
        # Uses iteration, batch_size, data_list, and num_data to extract next batch identifiers.
        next_batch = [None] * self.batch_size
        if self.iteration + self.batch_size >= self.num_data:
            next_batch[: self.num_data - self.iteration] = self.data_list[self.iteration :]
            next_batch[self.num_data - self.iteration :] = self.data_list[
                : self.batch_size - (self.num_data - self.iteration)
            ]
            random.shuffle(self.data_list)
            self.iteration = self.num_data - self.iteration
        else:
            next_batch = self.data_list[self.iteration : self.iteration + self.batch_size]
            self.iteration += self.batch_size
        assert self.iteration > -1
        assert len(next_batch) == self.batch_size
        return next_batch


class extractLanguageFeatures(extractData):
    def __init__(self, dataset, params, result=None):
        self.data_list = list(range(len(dataset)))
        self.num_data = len(self.data_list)
        self.dataset = dataset
        self.iteration = 0

        self.vocab_dict = params["vocab_dict"]
        self.batch_size = params["batch_size"]
        first_vec = next(iter(self.vocab_dict.values()))
        self.num_glove_centroids = first_vec.shape[0]
        self.T = params["sentence_length"]

        if isinstance(result, dict):
            self.result = result
            self.query_key = params["query_key"]
            self.cont_key = params["cont_key"]
            self.top_keys = [self.query_key, self.cont_key]
            self.top_shapes = [
                (self.T, self.batch_size, self.num_glove_centroids),
                (self.T, self.batch_size),
            ]
        else:
            print("Will only be able to run in test mode")

    def get_features(self, query: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        feature = np.zeros((self.T, self.num_glove_centroids), dtype=np.float32)
        cont = np.zeros((self.T,), dtype=np.float32)

        len_query = min(len(query), self.T)
        if len_query < len(query):
            query = query[:len_query]
        for count_word, word in enumerate(query):
            if word in self.vocab_dict:
                feature[-len_query + count_word, :] = self.vocab_dict[word]
            else:
                feature[-len_query + count_word, :] = np.zeros((glove_dim,), dtype=np.float32)
        if len_query > 1:
            cont[-(len_query - 1) :] = 1
        if len_query > 0:
            assert np.sum(feature[:-len_query, :]) == 0

        return feature, cont

    def get_data_test(self, data):
        query = data["language_input"]
        return self.get_features(query)

    def get_data(self, next_batch):
        data = self.dataset
        query_mat = np.zeros((self.T, self.batch_size, self.num_glove_centroids), dtype=np.float32)
        cont = np.zeros((self.T, self.batch_size), dtype=np.float32)

        for i, nb in enumerate(next_batch):
            query = data[nb]["language_input"]
            query_mat[:, i, :], cont[:, i] = self.get_features(query)

        self.result[self.query_key] = query_mat
        self.result[self.cont_key] = cont


class extractVisualFeatures(extractData):
    def __init__(self, dataset, params, result):
        self.data_list = list(range(len(dataset)))
        self.feature_process_algo = params["feature_process"]
        self.loc_feature = params["loc_feature"]
        self.num_data = len(self.data_list)
        self.dataset = dataset
        self.iteration = 0
        self.loc = params["loc_feature"]
        loss_type = params["loss_type"]
        if loss_type not in {"triplet", "inter", "intra"}:
            raise ValueError("loss_type must be one of triplet/inter/intra")

        self.inter = loss_type in {"triplet", "inter"}
        self.intra = loss_type in {"triplet", "intra"}

        self.batch_size = params["batch_size"]
        self.num_glove_centroids = params["num_glove_centroids"]

        with h5py.File(params["features"], "r") as features_h5py:
            features = {key: np.array(features_h5py[key]) for key in features_h5py.keys()}
        self.features = features

        if self.feature_process_algo not in feature_process_dict:
            raise ValueError(f"Unknown feature_process '{self.feature_process_algo}'")
        self.feature_process = feature_process_dict[self.feature_process_algo]

        self.feature_dim = self.feature_process(0, 0, self.features[self.dataset[0]["video"]]).shape[-1]
        self.result = result

        self.feature_key_p = params["feature_key_p"]
        self.feature_time_stamp_p = params["feature_time_stamp_p"]
        self.feature_time_stamp_n = params["feature_time_stamp_n"]

        self.top_keys = [self.feature_key_p, self.feature_time_stamp_p, self.feature_time_stamp_n]
        self.top_shapes = [(self.batch_size, self.feature_dim), (self.batch_size, 2), (self.batch_size, 2)]

        if self.inter:
            self.feature_key_inter = "features_inter"
            self.top_keys.append(self.feature_key_inter)
            self.top_shapes.append((self.batch_size, self.feature_dim))
        if self.intra:
            self.feature_key_intra = "features_intra"
            self.top_keys.append(self.feature_key_intra)
            self.top_shapes.append((self.batch_size, self.feature_dim))

        self.possible_annotations = possible_segments

    def get_data_test(self, data):
        video_feats = self.features[data["video"]]
        features = np.zeros((len(self.possible_annotations), self.feature_dim), dtype=np.float32)
        loc_feats = np.zeros((len(self.possible_annotations), 2), dtype=np.float32)
        for i, annotation in enumerate(self.possible_annotations):
            features[i, :] = self.feature_process(annotation[0], annotation[1], video_feats)
            loc_feats[i, :] = [annotation[0] / 6.0, annotation[1] / 6.0]
        return features, loc_feats

    def get_data(self, next_batch):
        feature_process = self.feature_process
        data = self.dataset
        features_p = np.zeros((self.batch_size, self.feature_dim), dtype=np.float32)
        if self.inter:
            features_inter = np.zeros((self.batch_size, self.feature_dim), dtype=np.float32)
        if self.intra:
            features_intra = np.zeros((self.batch_size, self.feature_dim), dtype=np.float32)

        features_time_stamp_p = np.zeros((self.batch_size, 2), dtype=np.float32)
        features_time_stamp_n = np.zeros((self.batch_size, 2), dtype=np.float32)

        for i, nb in enumerate(next_batch):
            rint = random.randint(0, len(data[nb]["times"]) - 1)
            gt_s = data[nb]["times"][rint][0]
            gt_e = data[nb]["times"][rint][1]
            possible_n = list(set(self.possible_annotations) - {(gt_s, gt_e)})
            random.shuffle(possible_n)
            n = possible_n[0]

            video = data[nb]["video"]
            feats = self.features[video]

            if self.inter:
                other_video = data[nb]["video"]
                while other_video == video:
                    other_video_index = int(random.random() * len(data))
                    other_video = data[other_video_index]["video"]
                feats_inter = self.features[other_video]

            features_p[i, :] = feature_process(gt_s, gt_e, feats)
            if self.intra:
                features_intra[i, :] = feature_process(n[0], n[1], feats)
            if self.inter:
                features_inter[i, :] = feature_process(gt_s, gt_e, feats_inter)

            if self.loc:
                features_time_stamp_p[i, 0] = gt_s / 6.0
                features_time_stamp_p[i, 1] = gt_e / 6.0
                features_time_stamp_n[i, 0] = n[0] / 6.0
                features_time_stamp_n[i, 1] = n[1] / 6.0
            else:
                features_time_stamp_p[i, 0] = 0
                features_time_stamp_p[i, 1] = 0
                features_time_stamp_n[i, 0] = 0
                features_time_stamp_n[i, 1] = 0

            assert not math.isnan(np.mean(self.features[data[nb]["video"]][n[0] : n[1] + 1, :]))
            assert not math.isnan(np.mean(self.features[data[nb]["video"]][gt_s : gt_e + 1, :]))

        self.result[self.feature_key_p] = features_p
        self.result[self.feature_time_stamp_p] = features_time_stamp_p
        self.result[self.feature_time_stamp_n] = features_time_stamp_n
        if self.inter:
            self.result[self.feature_key_inter] = features_inter
        if self.intra:
            self.result[self.feature_key_intra] = features_intra


class batchAdvancer(object):
    def __init__(self, extractors):
        self.extractors = extractors
        self.increment_extractor = extractors[0]

    def __call__(self):
        # The batch advancer just calls each extractor.
        next_batch = self.increment_extractor.increment()
        for extractor in self.extractors:
            extractor.get_data(next_batch)


language_feature_process_dict = {
    "zero_language": zero_language_vector,
    "recurrent_embedding": recurrent_embedding,
}


def build_preprocessed_data(descriptions_json: str, language_feature: str):
    if language_feature not in language_feature_process_dict:
        raise ValueError(f"Unsupported language feature process '{language_feature}'")

    data_orig = read_json(descriptions_json)
    random.shuffle(data_orig)
    language_processor = language_feature_process_dict[language_feature](data_orig)
    data = language_processor.preprocess(data_orig)
    return data, language_processor

