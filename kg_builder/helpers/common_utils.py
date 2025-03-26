import json
from enum import Enum
import textwrap


class SubjectCode(Enum):
    MODERN_INDIAN_HISTORY = "MIH"
    HISTORY_ART_AND_CULTURE = "HAC"
    POLITY = "POL"
    INTERNATIONAL_RELATIONS_AND_SECURITY = "IRS"
    ECONOMICS = "ECO"
    SCIENCE_AND_TECH = "SNT"
    ENVIRONMENT = "ENV"
    GEOGRAPHY = "GEO"
    SOCIAL_ISSUES = "SOCI"
    MISCELLANEOUS = "MISC"


class QuestionContentType(Enum):
    STATIC = 'STATIC'
    CURRENT_AFFAIRS = 'CA'


class PatternType(Enum):
    SINGLE_STATEMENT = "SINGLE_STATEMENT"
    TWO_STATEMENTS = "TWO_STATEMENTS"
    THREE_STATEMENTS = "THREE_STATEMENTS"
    IDENTIFY_FEATURES = "IDENTIFY_FEATURES"
    MATCH_THE_PAIRS = "MATCH_THE_PAIRS"
    # ASSERTION_REASONING = "ASSERTION_REASONING"


class PDFFileType(Enum):
    SCANNED = 'scanned'
    DIGITAL = 'digital'


def write_to_json(data, output_path):
    with open(output_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)


def merge_json_lists(list1_path, list2_path, output_file_path):
    try:
        with open(list1_path, 'r') as f1:
            list1 = json.load(f1)
        with open(list2_path, 'r') as f2:
            list2 = json.load(f2)
        if not isinstance(list1, list):
            raise ValueError("First file does not contain a JSON list.")
        if not isinstance(list2, list):
            raise ValueError("Second file does not contain a JSON list.")
        merged_list = list1 + list2
        with open(output_file_path, 'w') as outfile:
            json.dump(merged_list, outfile, indent=4)
        print(f"Successfully merged {list1_path} and {list2_path} into {output_file_path}")
    except FileNotFoundError:
        print(f"Error: One or both input files not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except ValueError as e:
        print(f"Error: {e}")


def wrap_text(text, width=70, break_long_words=True, break_on_hyphens=True):
    try:
        wrapped_text = textwrap.fill(text, width=width, break_long_words=break_long_words,
                                     break_on_hyphens=break_on_hyphens)
        return wrapped_text
    except Exception as e:
        print(f"Error during text wrapping: {e}")
        return text
