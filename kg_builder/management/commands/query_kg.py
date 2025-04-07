from django.core.management.base import BaseCommand
import logging
from ...helpers import query_kg_helper as helper

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'This is a utility management command for querying knowledge graph'

    def handle(self, *args, **options):
        target_entity_text = "Kanishka"
        all_paths = helper.get_paths_from_entity_any_label(target_entity_text, max_depth=3)
        # print(all_paths)
        print(f"\nAll paths starting from node with text '{target_entity_text}' (any label, up to depth 3):")
        for path in all_paths[0:20]:
            print("  Path:")
            for segment in path:
                print(f"    Start: {segment['start_node']['text']} ({segment['start_node'].get('label', 'No Label')})")
                print(f"    Relation: {segment['relation']['name']}")
                print(f"    End: {segment['end_node']['text']} ({segment['end_node'].get('label', 'No Label')})")
            print("-" * 20)
