import time

from django.core.management.base import BaseCommand
import logging
from ...helpers import get_upsc_cutom_entities_helper as helper
import  json
from ...helpers import extract_text_helper as eth
from langchain.text_splitter import CharacterTextSplitter

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'This is a utility management command for getting custom named entities for upsc content'

    def handle(self, *args, **options):
        upsc_custom_entities = [
            ("Mahajanapadas", "HISTORICAL_PERIOD"),
            ("Indian Constitution", "POLITICAL_DOCUMENT"),
            ("Article 370", "CONSTITUTIONAL_ARTICLE"),
            ("Paris Agreement", "INTERNATIONAL_AGREEMENT"),
            ("Bhimbetka", "ARCHAEOLOGICAL_SITE"),
            ("Planning Commission", "INDIAN_INSTITUTION"),
            ("Green Revolution", "AGRICULTURAL_EVENT"),
        ]

        sample_text = """
            The Mauryan Empire followed the Mahajanapadas in ancient India. The Indian Constitution lays down the framework for governance. Article 370 was a special provision for Jammu and Kashmir. The Paris Agreement aims to combat climate change. Excavations at Bhimbetka reveal prehistoric rock paintings.
            """
        pdf_extractor = eth.select_pdf_extractor("digital", 1, "no")
        extracted_text = pdf_extractor.extract_text("/Users/ankit.anand/Desktop/hac.pdf")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator=".")
        pdf_chunks = text_splitter.split_text(extracted_text)
        for i in range(len(pdf_chunks)):
            ner_labels = helper.get_ner_labels_from_llm(pdf_chunks[i], upsc_custom_entities)
            with open("temp/data.json","a+") as file:
                file.write(str(ner_labels))
            time.sleep(5)

        # print(json.dumps(ner_labels, indent=2))
