import time

from django.core.management.base import BaseCommand
import logging
from ...helpers import prepare_ner_model_training_data_helper as helper
import  json
from ...helpers import extract_text_helper as eth
from langchain.text_splitter import CharacterTextSplitter

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'This is a utility management command for getting custom named entities for upsc content'

    def handle(self, *args, **options):
        pdf_extractor = eth.select_pdf_extractor("digital", 1, "no")
        extracted_text = pdf_extractor.extract_text("/Users/ankit.anand/Desktop/hac.pdf")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator=".")
        pdf_chunks = text_splitter.split_text(extracted_text)
        l = []
        for i in range(len(pdf_chunks)):
            training_datapoint = helper.get_ner_labels_from_llm_for_spacy(pdf_chunks[i])
            l.append(training_datapoint)
            time.sleep(5)
        with open("temp/training_data.json", "w") as file:
            file.write(json.dumps(l))