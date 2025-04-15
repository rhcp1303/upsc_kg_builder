import time

from django.core.management.base import BaseCommand
import logging
from ...helpers import prepare_kg_relations_data_helper as helper
import json
from ...helpers import extract_text_helper as eth
from langchain.text_splitter import CharacterTextSplitter

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'This is a utility management command for preparing training data for custom named entities recognition model for upsc content'

    def handle(self, *args, **options):
        pdf_extractor = eth.select_pdf_extractor("digital", 1, "no")
        extracted_text = pdf_extractor.extract_text("/Users/ankit.anand/Downloads/ca2025/budget26.pdf")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator=".")
        pdf_chunks = text_splitter.split_text(extracted_text)
        l = []
        for i in range(len(pdf_chunks)):
            text_with_relationships = helper.get_entities_and_relations_from_llm(pdf_chunks[i])
            if text_with_relationships:
                text, result_dict = text_with_relationships
                result_json = json.dumps(result_dict, indent=2)
                print("Relationships with Labels:", result_json)
                l.append(result_dict)
            time.sleep(5)
        with open("temp/budget26.json", "w") as file:
            file.write(json.dumps(l))
