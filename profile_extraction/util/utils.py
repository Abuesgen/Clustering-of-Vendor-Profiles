"""
This module contains helper functions
"""
from typing import Any, Dict, List, Optional, Tuple

from sklearn.model_selection import train_test_split


def train_dev_test_split(
    dataset: List[Any], dev_split: float, test_split: float, random_state: Optional[int] = None
) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Splits a given dataset randomly into train, dev and testsets

    :param dataset: dataset to split
    :param dev_split: portion of entries to put into devset
    :param test_split: portion of entries to put into testset
    :param random_state: allows you to set a random seed

    :return: a tuple of 3 lists containing train, dev and testdata
    """
    if dev_split + test_split >= 1:
        raise ValueError("The sum of the splits must be less than 1.")

    dataset_len = len(dataset)
    dev_testset_len = round(dataset_len * (dev_split + test_split))
    testset_len = round(dev_testset_len * (test_split / (dev_split + test_split)))
    if dev_testset_len >= dataset_len or testset_len == 0:
        raise ValueError("The given split is not possible.")

    trainset, test_dev_set = train_test_split(dataset, test_size=dev_testset_len, random_state=random_state)
    devset, testset = train_test_split(test_dev_set, test_size=testset_len, random_state=random_state)
    return trainset, devset, testset


def filter_dataset(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    A helper function for filtering data points which should not be included.
    Further it removes the NO_ANNOTATION labels as they are solely helper labels.

    :param dataset: the dataset to filter
    :return: the filtered dataset
    """
    filtered_dataset: List[Dict[str, Any]] = []

    for entry in dataset:
        labels = {label[2] for label in entry["labels"]}
        if "NOT_INCLUDED" in labels:
            pass
        elif "NO_ANNOTATION" in labels:
            entry["labels"] = []
            filtered_dataset.append(entry)
        else:
            filtered_dataset.append(entry)

    return filtered_dataset
