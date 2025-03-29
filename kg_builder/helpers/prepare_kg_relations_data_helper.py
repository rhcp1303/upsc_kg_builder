from langchain_google_genai import ChatGoogleGenerativeAI
import json
import os
import spacy

api_key = "AIzaSyBq2_GdMf0KhowSVSb0hn4Z_8B81kBewXY"
os.environ["GOOGLE_API_KEY"] = api_key
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
nlp_trained = spacy.load("trained_spacy_model")


def get_entities_and_relations_from_llm(text):
    prompt = f"""
Identify named entities and their relationships in the following text, focusing on relevance to the UPSC syllabus.

Text:
{text}

**Output Format**:

Provide the output as a single JSON object without any backticks. The JSON object should have one key: "relationships".

* **"relationships"**: A JSON list of dictionaries, where each dictionary describes a relationship and includes the entity text and their NER labels. Each relationship dictionary should have the following structure:
  {{
    "entity1": {{ "entity_text": "text of entity 1", "label": "NER label of entity 1" }},
    "entity2": {{ "entity_text": "text of entity 2", "label": "NER label of entity 2" }},
    "relation": "concise label describing the relationship"
  }}

**Constraints**:

* Only include relationships between entities that are highly relevant to the UPSC context. Do not include trivial relationships.

**Example Output:**
{{
  "relationships": [
    {{
      "entity1": {{ "entity_text": "Indian Constitution", "label": "POLITICAL_DOCUMENT" }},
      "entity2": {{ "entity_text": "Fundamental Rights", "label": "LEGAL_CONCEPT" }},
      "relation": "contains"
    }},
    {{
      "entity1": {{ "entity_text": "Indus Valley Civilization", "label": "HISTORICAL_PERIOD" }},
      "entity2": {{ "entity_text": "Harappa", "label": "ARCHAEOLOGICAL_SITE" }},
      "relation": "site_of"
    }},
    {{
      "entity1": {{ "entity_text": "Climate Change", "label": "ENVIRONMENTAL_ISSUE" }},
      "entity2": {{ "entity_text": "Global Warming", "label": "ENVIRONMENTAL_PROCESS" }},
      "relation": "causes"
    }}
  ]
}}
"""

    try:
        response = llm.invoke(prompt)
        llm_output = response.content.strip()
        try:
            llm_data = json.loads(llm_output)
            relationships_with_labels = []

            for rel_data in llm_data.get("relationships", []):
                entity1_info = rel_data.get("entity1")
                entity2_info = rel_data.get("entity2")
                relation_label = rel_data.get("relation")

                if entity1_info and entity2_info and relation_label:
                    relationships_with_labels.append({
                        "entity1": {"entity_text": entity1_info.get("entity_text"), "label": entity1_info.get("label")},
                        "entity2": {"entity_text": entity2_info.get("entity_text"), "label": entity2_info.get("label")},
                        "relation": relation_label
                    })

            return [text,  relationships_with_labels]

        except json.JSONDecodeError:
            print(f"Error decoding JSON from LLM: {llm_output}")
            return [text, []]
    except Exception as e:
        print(f"Error calling Google Gemini API: {e}")
        return [text, []]


