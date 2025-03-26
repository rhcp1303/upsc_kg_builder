from langchain_google_genai import ChatGoogleGenerativeAI
import json
import os

api_key = "AIzaSyBq2_GdMf0KhowSVSb0hn4Z_8B81kBewXY"
os.environ["GOOGLE_API_KEY"] = api_key
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

def get_ner_labels_from_llm(text, custom_entities):
    entity_definitions = "\n".join([f"- {entity}: {label}" for entity, label in custom_entities])
    prompt = f"""
    Identify named entities in the following text that are relevant to the UPSC syllabus.
    Pay close attention to the following custom entities and their definitions:
    {entity_definitions}

    Text:
    {text}

    Provide the entities as a JSON list of dictionaries, where each dictionary has "entity" and "label" keys.
    Only include entities that match the provided custom entities or are common general entities (PERSON, ORG, GPE, DATE, etc.) if they are highly relevant to the UPSC context.
    
    **Instruction**
    1. Give output without any backtick
    
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
        llm_output = response.content
        try:
            return json.loads(llm_output)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from LLM: {llm_output}")
            return []
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return []