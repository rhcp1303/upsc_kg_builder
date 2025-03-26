from langchain_google_genai import ChatGoogleGenerativeAI
import json
import os
import spacy

api_key = "AIzaSyBq2_GdMf0KhowSVSb0hn4Z_8B81kBewXY"
os.environ["GOOGLE_API_KEY"] = api_key
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
nlp = spacy.load("en_core_web_sm")


def get_ner_labels_from_llm_for_spacy(text):
    prompt = f"""
    Identify named entities in the following text that are relevant to the UPSC syllabus.
    Text:
    {text}

    Provide the entities as a JSON list of dictionaries, where each dictionary has "entity" (the exact text of the entity) and "label".
    Only include entities that are highly relevant to the UPSC context.
    
    **Instructions**:
    1. Give the output without any backtick.

    Example Output:
    [
      {{ "entity": "Mahajanapadas", "label": "HISTORICAL_PERIOD" }},
      {{ "entity": "Indian Constitution", "label": "POLITICAL_DOCUMENT" }},
      {{ "entity": "Article 370", "label": "CONSTITUTIONAL_ARTICLE" }},
      {{ "entity": "Paris Agreement", "label": "INTERNATIONAL_AGREEMENT" }},
      {{ "entity": "Bhimbetka", "label": "ARCHAEOLOGICAL_SITE" }}
    ]
    """

    try:
        response = llm.invoke(prompt)
        llm_output = response.content.strip()
        try:
            llm_entities = json.loads(llm_output)
            annotations = {"entities": []}
            doc = nlp(text)

            for llm_entity_data in llm_entities:
                entity_text = llm_entity_data.get("entity")
                entity_label = llm_entity_data.get("label")

                if entity_text and entity_label:
                    for token in doc:
                        if token.text == entity_text:
                            start_char = token.idx
                            end_char = start_char + len(entity_text)
                            annotations["entities"].append((start_char, end_char, entity_label))
                            break

            return [text, annotations]

        except json.JSONDecodeError:
            print(f"Error decoding JSON from LLM: {llm_output}")
            return [text, {"entities": []}]
    except Exception as e:
        print(f"Error calling Google Gemini API: {e}")
        return [text, {"entities": []}]
