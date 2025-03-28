from langchain_google_genai import ChatGoogleGenerativeAI
import json
import os
import spacy

api_key = "AIzaSyBq2_GdMf0KhowSVSb0hn4Z_8B81kBewXY"
os.environ["GOOGLE_API_KEY"] = api_key
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
nlp_trained = spacy.load("trained_spacy_model")


def get_relations_from_llm_for_spacy(text, entities):
    if not entities:
        return text, {"relations": []}

    entity_mentions = []
    for start, end, label in entities:
        entity_mentions.append(f"'{text[start:end]}' ({label})")

    prompt = f"""
       Identify and extract relationships between the following named entities present in the text below, focusing on relationships relevant to the UPSC syllabus.

       Text:
       {text}

       Entities:
       {', '.join(entity_mentions)}

       Provide the relationships as a JSON list of dictionaries, where each dictionary has "entity1" (the text of the first entity), "entity2" (the text of the second entity), and "relation" (a concise label describing the relationship between them).

       Only include relationships that are highly relevant to the UPSC context. Do not include trivial relationships.

       **Instructions**:
       1. Give the output without any backtick.
       2. Ensure that both entity1 and entity2 are present in the provided Entities list.

       Example Output:
       [
         {{ "entity1": "Indian Constitution", "entity2": "Fundamental Rights", "relation": "contains" }},
         {{ "entity1": "Indus Valley Civilization", "entity2": "Harappa", "relation": "part_of" }},
         {{ "entity1": "Climate Change", "entity2": "Global Warming", "relation": "cause_of" }}
       ]
       """
    try:
        response = llm.invoke(prompt)
        llm_output = response.content.strip()
        try:
            llm_relations = json.loads(llm_output)
            relations = []
            input_entity_map = {text[start:end]: i for i, (start, end, label) in enumerate(entities)}

            for relation_data in llm_relations:
                entity1_text = relation_data.get("entity1")
                entity2_text = relation_data.get("entity2")
                relation_label = relation_data.get("relation")

                if entity1_text and entity2_text and relation_label:
                    head_index = input_entity_map.get(entity1_text)
                    tail_index = input_entity_map.get(entity2_text)

                    if head_index is not None and tail_index is not None:
                        relations.append((head_index, tail_index, relation_label))

            return [text, {"relations": relations}]

        except json.JSONDecodeError:
            print(f"Error decoding JSON from LLM (Relations): {llm_output}")
            return text, {"relations": []}
    except Exception as e:
        print(f"Error calling Google Gemini API (Relations): {e}")
        return text, {"relations": []}
