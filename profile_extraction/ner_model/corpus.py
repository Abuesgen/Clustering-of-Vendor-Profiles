"""
This module contains all classes and methods for reading Jsonl corpora (doccano format)
"""
import json
from pathlib import Path
from typing import Any, List, Optional, Union

from flair.data import Corpus, FlairDataset, Sentence
from flair.datasets.base import find_train_dev_test_files
from flair.tokenization import SegtokTokenizer
from torch.utils.data import ConcatDataset, Dataset


class MultiFileJsonlCorpus(Corpus):
    """
    This class represents a generic Jsonl corpus with multiple files for
    train, dev and test dataset.
    """

    def __init__(
        self,
        train_files=None,
        test_files=None,
        dev_files=None,
        text_column_name: str = "text",
        label_column_name: str = "labels",
        **corpusargs,
    ):  # pylint: disable=too-many-arguments
        """
        Initializes a MuliFileJsonlCorpus.
        Note at least one of train_files, test_files and dev_files must contain
        at least one path, otherwise an Error will be thrown.

        :param corpusargs: Additional arguments for Corpus initialization
        :param train_files: List of paths to training files
        :param test_files: List of paths to test files
        :param dev_files: List of paths to dev files
        :param text_column_name: Name of the text column inside the jsonl files.
        :param label_column_name: Name of the label column inside the jsonl files.

        :raises RuntimeError: If no paths are given
        """
        train: Optional[Dataset] = (
            ConcatDataset(
                [
                    JsonlDataset(train_file, text_column_name=text_column_name, label_column_name=label_column_name)
                    for train_file in train_files
                ]
            )
            if train_files and train_files[0]
            else None
        )

        # read in test file if exists
        test: Optional[Dataset] = (
            ConcatDataset(
                [
                    JsonlDataset(test_file, text_column_name=text_column_name, label_column_name=label_column_name)
                    for test_file in test_files
                ]
            )
            if test_files and test_files[0]
            else None
        )

        # read in dev file if exists
        dev: Optional[Dataset] = (
            ConcatDataset(
                [
                    JsonlDataset(dev_file, text_column_name=text_column_name, label_column_name=label_column_name)
                    for dev_file in dev_files
                ]
            )
            if dev_files and dev_files[0]
            else None
        )
        super().__init__(train, dev, test, **corpusargs)


class JsonlCorpus(MultiFileJsonlCorpus):
    """
    Represents a JsonlCorpus with one file per dataset.
    It provides some convenience functions such as automatically finding
    train, dev, and test files if they are named accordingly.
    """

    def __init__(
        self,
        data_folder: Union[str, Path],
        train_file: Optional[Union[str, Path]] = None,
        test_file: Optional[Union[str, Path]] = None,
        dev_file: Optional[Union[str, Path]] = None,
        text_column_name: str = "text",
        label_column_name: str = "labels",
        autofind_splits: bool = True,
        name: Optional[str] = None,
        **corpusargs,
    ):  # pylint: disable=too-many-arguments
        """
        Initializes a JsonlCorpus

        :param data_folder: Folder in which the files are stored
        :param train_file: Name of the train file. If None and autofind_splits is true a
                            file with filename containing 'train' is chosen.
        :param test_file: Name of the test file. If None and autofind_splits is true a
                            file with filename containing 'test' or 'testb' is chosen.
        :param dev_file: Name of the dev file. If None and autofind_splits is true a
                            file with filename containing 'dev' or 'testa' is chosen.
        :param text_column_name: Name of the text column inside the jsonl file.
        :param label_column_name: Name of the label column inside the jsonl file.
        :param autofind_splits: Whether train, test and dev file should be determined automatically
        :param name: name of the Corpus see flair.data.Corpus
        """
        # find train, dev and test files if not specified
        dev_file, test_file, train_file = find_train_dev_test_files(
            data_folder, dev_file, test_file, train_file, autofind_splits
        )
        super().__init__(
            dev_files=[dev_file] if dev_file else [],
            train_files=[train_file] if train_file else [],
            test_files=[test_file] if test_file else [],
            text_column_name=text_column_name,
            label_column_name=label_column_name,
            name=name if data_folder is None else str(data_folder),
            **corpusargs,
        )


class JsonlDataset(FlairDataset):
    """
    Reads a given JsonlDataset and Provides its sentences and NE-annotations as FlairDataset.

    A Dataset represents one set of examples.
    Typically, it corresponds to one file in the filesystem.
    """

    def __init__(
        self,
        path_to_json_file: Union[str, Path],
        text_column_name: str = "text",
        label_column_name: str = "labels",
    ):
        """
        Reads a given Jsonl file in doccano format.
        The expected file format is:

        { "<text_column_name>": "<text>", "label_column_name": [[<start_char_index>, <end_char_index>, <label>],...] }

        :param path_to_json_file: File to read
        :param text_column_name: Name of the text column
        :param label_column_name: Name of the label column

        """
        path_to_json_file = Path(path_to_json_file)

        self.text_column_name = text_column_name
        self.label_column_name = label_column_name
        self.path_to_json_file = path_to_json_file

        self.sentences: List[Sentence] = []
        with path_to_json_file.open(encoding="utf-8") as jsonl_fp:
            for line in jsonl_fp:
                current_line = json.loads(line)
                raw_text = current_line[text_column_name]
                current_labels = current_line[label_column_name]
                current_sentence = Sentence(raw_text, use_tokenizer=SegtokTokenizer())

                JsonlDataset._add_labels_to_sentence(raw_text, current_sentence, current_labels)

                self.sentences.append(current_sentence)

    @staticmethod
    def _add_labels_to_sentence(raw_text: str, sentence: Sentence, labels: List[List[Any]]):
        for label in labels:
            JsonlDataset._add_label_to_sentence(raw_text, sentence, label[0], label[1], label[2])
        for token in sentence:
            if token.get_tag("ner").value == "":
                token.add_tag("ner", "O")

    @staticmethod
    def _add_label_to_sentence(text: str, sentence: Sentence, start: int, end: int, label: str):
        """
        Adds a NE label to a given sentence.

        :param text: raw sentence (with all whitespaces etc.). Is used to determine the token indices.
        :param sentence: Tokenized flair Sentence.
        :param start: Start character index of the label.
        :param end: End character index of the label.
        :param label: Label to assign to the given range.
        :return: Nothing. Changes sentence as INOUT-param
        """

        annotated_part = text[start:end]

        while annotated_part.startswith(" "):
            start += 1
            annotated_part = text[start:end]

        while annotated_part.endswith(" "):
            end -= 1
            annotated_part = text[start:end]

        start_idx = -1
        end_idx = -1
        for token in sentence:
            if token.start_pos <= start <= token.end_pos and start_idx == -1:
                start_idx = token.idx - 1

            if token.start_pos <= end <= token.end_pos and end_idx == -1:
                end_idx = token.idx - 1
        if end_idx == -1:
            end_idx = sentence[-1].idx - 1

        if start_idx == -1 or start_idx > end_idx:
            raise ValueError(
                f"Could not create token span from char span.\n\
                    Sen: {sentence}\nStart: {start}, End: {end}, Label: {label}\n\
                        Ann: {annotated_part}\nRaw: {text}\nCo: {start_idx}, {end_idx}"
            )

        prefix = "B"
        for token in sentence[start_idx : end_idx + 1]:
            token.add_tag("ner", f"{prefix}-{label}")
            prefix = "I"

    def is_in_memory(self) -> bool:
        """
        Currently all Jsonl Datasets are stored in Memory
        """
        return True

    def __len__(self):
        """
        Number of sentences in the Dataset
        """
        return len(self.sentences)

    def __getitem__(self, index: int = 0) -> Sentence:
        """
        Returns the sentence at a given index
        """
        return self.sentences[index]
