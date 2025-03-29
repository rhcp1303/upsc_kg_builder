import time
from django.core.management.base import BaseCommand
import logging
from ...helpers import prepare_relation_extraction_model_training_data_helper as relations_helper
import json
from ...helpers import extract_text_helper as eth
from langchain.text_splitter import CharacterTextSplitter
import spacy

logger = logging.getLogger(__name__)

nlp_trained = spacy.load("trained_spacy_model")


class Command(BaseCommand):
    help = 'This is a utility management command for preparing training data for relationship extraction model for upsc content'

    def handle(self, *args, **options):
        pdf_extractor = eth.select_pdf_extractor("scanned", 2, "no")
        extracted_text = pdf_extractor.extract_text("/Users/ankit.anand/Desktop/old_ancient.pdf")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator=".")
        pdf_chunks = text_splitter.split_text(extracted_text)
        l = []
        for i in range(len(pdf_chunks)):
            doc = nlp_trained(pdf_chunks[i])
            spacy_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
            training_datapoint = relations_helper.get_relations_from_llm_for_spacy(pdf_chunks[i], spacy_entities)
            l.append(training_datapoint)
            time.sleep(5)
        with open("temp/nitin_singhania_relation_training_data.json", "w") as file:
            file.write(json.dumps(l))

