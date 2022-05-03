"""
This module provides functions and commands for training FlairEmbeddings
"""

import click
from flair.data import Dictionary
from flair.embeddings import FlairEmbeddings
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
from torch.optim import AdamW


@click.command()
@click.option("--forward/--backward", default=True)
@click.argument("data-path", type=click.Path(dir_okay=True, file_okay=False), required=True)
@click.argument("output-path", type=click.Path(dir_okay=True, file_okay=False), required=True)
def train_language_model(data_path: str, output_path: str, forward: bool):
    """
    Trains a language model with a given dictionary

    :param data_path: path to train, dev and testdata
    :param output_path: path to save the trained model to
    :param forward: whether to train a forward or backward language model
    """

    # instantiate an existing LM, such as one from the FlairEmbeddings
    language_model = FlairEmbeddings(f'de-{"forward" if forward else "backward" }').lm

    # are you fine-tuning a forward or backward LM?
    is_forward_lm = language_model.is_forward_lm

    # get the dictionary from the existing language model
    dictionary: Dictionary = language_model.dictionary

    # get your corpus, process forward and at the character level
    corpus = TextCorpus(data_path, dictionary, is_forward_lm, character_level=True)

    # train your language model
    trainer = LanguageModelTrainer(language_model, corpus, optimizer=AdamW)

    trainer.train(
        output_path, sequence_length=250, mini_batch_size=100, max_epochs=100, patience=10, learning_rate=1e-3
    )
