"""
This module provides commands to extract profiles from a JSON chat export
using a trained NER model
"""
from typing import TextIO

import click
from flair.models import SequenceTagger

from profile_extraction.profile_creation.extraction.profile_extractor import (
    extract_profiles,
)
from profile_extraction.profile_creation.reader import ChatReaderFactory, DataType


@click.command()
@click.option(
    "--model",
    type=click.Path(dir_okay=False, file_okay=True),
    required=True,
    help="Path to a SequenceTagger providing NER for the Profile Extraction",
)
@click.option("--chat", type=click.File(), required=True, help="File containing a JSON chat export")
@click.option("--output", type=click.File("w"), required=True, help="File to write the result to")
@click.option("--summary-only/--full-profiles", default=False, type=bool)
def profile_creation(model: str, chat: TextIO, output: TextIO, summary_only: bool):
    """
    Generates Chat Profiles from a given chat export

    :param model: Model to use for NamedEntityRecognition
    :param chat: Chat file to analyze
    :param output: File to save the generated profiles to
    :param summary_only: Whether the application should only generate a summary for each user
    """
    reader = ChatReaderFactory.get_instance(DataType.JSON)
    sequence_tagger = SequenceTagger.load(model)

    profiles = extract_profiles(reader(chat), sequence_tagger=sequence_tagger, summary_only=summary_only)
    print(profiles.json(indent=4), file=output)
