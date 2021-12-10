# !pip install fasttext loguru utilpy

import os
import re
import random
import re
from collections import defaultdict
from dataclasses import dataclass
import csv
from pprint import pprint

import fasttext
from loguru import logger
from tqdm import tqdm
from utilpy import files


def mask_nums(x):
    """preprocessing the texts and masking numbers/amounts"""
    tmp = re.sub(r"\d", "X", x)
    tmp = re.sub(r"-", " ", tmp)
    tmp = re.sub(r"@", " ", tmp)
    tmp = re.sub(r":", " ", tmp)
    tmp = re.sub(r"\n", " ", tmp)
    return re.sub(r"\/", " ", tmp)


@dataclass
class TrainFasttext:
    file_path: str
    model_id: str
    split_ratio: float = 0.8
    version: str = "v1"
    model_file: str = os.path.join("tmp",  "model.bin")
    train_file: str = os.path.join("tmp", "train.txt")
    test_file: str = os.path.join("tmp", "test.txt")
    files.make_dirs('tmp')


    def save_model(self):
        self.model.save_model(self.model_file)
        logger.info(f"Model saved in path: {self.model_file} !!")

    def create_dataset(self, text_name: str, labels_name: str):
        self.data_folder = 'tmp'
        with open(self.file_path, "r", encoding='latin-1') as file:
            data = [line for line in csv.DictReader(file)]

        # make files to store data
        for i in ["train", "test"]:
            with open(f"{self.data_folder}/{i}.txt", "w") as f:
                if i == "train":
                    self.train_file = f"{self.data_folder}/{i}.txt"
                else:
                    self.test_file = f"{self.data_folder}/{i}.txt"

        for d in tqdm(data):
            text = d.get(text_name)
            label = d.get(labels_name)
            if text and label:
                line = self._get_text_and_labels(text, label)

                dataset_type = (
                    self.train_file
                    if random.random() <= self.split_ratio
                    else self.test_file
                )
                with open(dataset_type, "a", encoding="utf-8") as f:
                    f.write(line)
                    f.write("\n")
        logger.info("dataset created!!")

    def train(self, train_file=None, test_file=None, save_quantize=False, **kwargs):
        """
        file-format for train and test files: label name followed by space and the corresponding text
        __label__1 Text
        __label__2 Text
        .
        .
        """
        if not self.test_file and not train_file:
            raise ValueError(f"No training file found in path {train_file} !!")
        elif train_file:
            self.train_file = train_file

        if not self.test_file and test_file:
            self.test_file = test_file

        # get params
        self.model_json = {}
        model_size = kwargs.get('model_size', '2M')
        tune_time = kwargs.get('tune_time', 600)

        # start training
        if os.path.isfile(self.train_file):
            logger.info("Training Started!!")
            if not save_quantize:
                self.model = fasttext.train_supervised(
                    self.train_file,
                    autotuneValidationFile=self.test_file,
                    autotuneDuration=tune_time)
                logger.info("Training Completed!!")
            else:
                model_size = kwargs.get('model_size', '2M')
                self.model = fasttext.train_supervised(
                    self.train_file, autotuneValidationFile=self.test_file, autotuneModelSize=model_size)

            if self.test_file is not None:
                if os.path.isfile(self.test_file):
                    logger.info("Calculating metrics in test file...")
                    result = self.model.test(self.test_file)
                    n, p, r = result
                    labels = self._accuracy_matrix_tf(self.test_file)
                    self.model_json["test_metrices"] = {
                        "summary": {"n": n, "p": p, "r": r},
                        "labels": labels,
                    }

                    logger.info("Calculating metrics in train file...")
                    result = self.model.test(self.train_file)
                    n, p, r = result
                    labels = self._accuracy_matrix_tf(self.train_file)
                    self.model_json["train_metrices"] = {
                        "summary": {"n": n, "p": p, "r": r},
                        "labels": labels,
                    }
                    print("Training Matrix {}".format(self.model_json))

        else:
            ValueError(f"No training file found in path {self.test_file} !!")

    def predict(self, text):
        if not self.model:
            raise ValueError("Use load_model to load model")

        # make list for predication
        if type(text) == str:
            text = [text]
        predicted_labels = self.model.predict(text)

        for i, j in zip(predicted_labels[0], predicted_labels[1]):
            logger.info(f"Predicted Label: {i} with accuracy: {j}")

    def predict_test_file(self, test_file):
        # read testing dataset
        if not self.model:
            raise ValueError("Use load_model to load model")

        with open(test_file, "r") as f:
            lines = f.readlines()

        # read file and make a predication
        actual_label = [
            re.search(r"__label\w+", line.strip()).group() for line in lines
        ]
        texts = [re.sub(r"__label\w+\s", "", line.strip()) for line in lines]
        predicted_labels = self.model.predict(texts)
        pred_data = []
        for t, i, j, k in zip(
            texts, predicted_labels[0], actual_label, predicted_labels[1]
        ):
            logger.info(f"Predicted Label: {i}, true label {j} with accuracy: {k}")
            pred_data.append(
                [t, j.replace("__label__", ""), i[0].replace("__label__", "")]
            )
        return pred_data

    def load_model(self, path=None):
        if not path:
            path = self.model_file

        self.model = fasttext.load_model(path)

    def _get_text_and_labels(self, text, label):
        if not type(label) == "str":
            label = str(label)
        if text:
            # sort and convert to required format
            text = mask_nums(str(text))
            data = "__label__" + label.replace(" ", "_") + " " + text[:1000]
            return data
        else:
            return ""

    def _accuracy_matrix_tf(self, test_path):
        # read testing dataset
        with open(test_path, "r") as f:
            lines = f.readlines()

        # read file and make a predication
        actual_label = [
            re.search(r"__label\w+", line.strip()).group() for line in lines
        ]
        texts = [re.sub(r"__label\w+\s", "", line.strip()) for line in lines]
        predicted_labels = self.model.predict(texts)

        # clean get accuracy matrix
        actual_label = [j.replace("__label__", "") for j in actual_label]
        predicted_labels = [i[0].replace("__label__", "") for i in predicted_labels[0]]
        acc_label_dict = defaultdict(list)

        for i, j in zip(actual_label, predicted_labels):
            acc_label_dict[i].append(i == j)

        # get data
        acc_matrix = {
            i: {"n": len(j), "accuarcy": sum(j) / len(j)}
            for i, j in acc_label_dict.items()
        }

        return acc_matrix


if __name__ == "__main__":
    file_path = "datasets/spam_or_ham/train.csv"

    model = TrainFasttext(file_path=file_path, model_id="1234")
    model.create_dataset("Message","Category")
    model.train(save_quantize=False, tune_time=100)
    model.save_model()
    pprint(model.model_json)

    model.load_model()
    model.predict("OMGGGGG WHAT IS GOINGGG ONNNNNN!!!!")
