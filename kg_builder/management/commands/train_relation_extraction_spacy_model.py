from django.core.management.base import BaseCommand
import logging
from ...helpers import relation_extraction_spacy_model_helper as helper

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'This is a utility management command for training relation extraction spacy model'

    def handle(self, *args, **options):
        output_dir = "temp/relations_training_data.json"
        helper.train_relation_extraction("train_relations.json", output_dir)