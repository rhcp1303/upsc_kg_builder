import json

from django.core.management.base import BaseCommand
import logging
from ...helpers import spacy_model_helper as helper

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'This is a utility management command for training spacy model'

    def handle(self, *args, **options):
        with open("temp/training_data.json", "r") as f:
            train_data_json = f.read()
            train_data = json.loads(train_data_json)
        output_directory = "trained_spacy_model"
        helper.train_spacy_ner(train_data, output_directory, n_iter=100)
