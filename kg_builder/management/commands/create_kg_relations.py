from django.core.management.base import BaseCommand
import logging
from neo4j import GraphDatabase
from ...helpers import create_kg_relations_helper as kgh

logger = logging.getLogger(__name__)

uri = "bolt://localhost:7687"
user = "neo4j"
password = "neo4j1234!"
database_name = "neo4j"


class Command(BaseCommand):
    help = 'This is a utility management command for creating knowledge graph relations and inserting it into database'

    def handle(self, *args, **options):
        graph_db = GraphDatabase.driver(uri, auth=(user, password))
        kgh.create_and_insert_relations_into_kg("temp/pmfias_south_america.json")
        graph_db.close()
