from django.core.management.base import BaseCommand
import logging
from ...helpers import query_kg_helper as helper

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'This is a utility management command for querying knowledge graph'

    def handle(self, *args, **options):
        target_entity_text = "Kanishka"
        connected_info = helper.get_neighbors_any_label(target_entity_text)
        print(f"Nodes directly connected to node with text '{target_entity_text}' (any label):")
        for connection in connected_info:
            print(
                f"  Source: {connection['source_node']['text']} ({connection['source_node'].get('label', 'No Label')})")
            print(f"  Relation: {connection['relation']}")
            print(
                f"  Target: {connection['target_node']['text']} ({connection['target_node'].get('label', 'No Label')})")
            print("-" * 20)

        all_paths = helper.get_paths_from_entity_any_label(target_entity_text, max_depth=3)
        print(f"\nAll paths starting from node with text '{target_entity_text}' (any label, up to depth 3):")
        for path in all_paths:
            print("  Path:")
            for segment in path:
                print(f"    Start: {segment['start_node']['text']} ({segment['start_node'].get('label', 'No Label')})")
                print(f"    Relation: {segment['relation']}")
                print(f"    End: {segment['end_node']['text']} ({segment['end_node'].get('label', 'No Label')})")
            print("-" * 20)
