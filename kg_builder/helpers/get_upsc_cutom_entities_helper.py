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

    Text:
    {text}

    Provide the entities the following format, where each dictionary has "entity" and "label" keys.
    Only include entities that are highly relevant to the UPSC context.
    
    **Instruction**
    1. Give output without any backtick
    
    Example Output:
      {{ "entity": "Mahajanapadas", "label": "HISTORICAL_PERIOD" }},
      {{ "entity": "Indian Constitution", "label": "POLITICAL_DOCUMENT" }},
      {{ "entity": "Article 370", "label": "CONSTITUTIONAL_ARTICLE" }},
      {{ "entity": "Paris Agreement", "label": "INTERNATIONAL_AGREEMENT" }},
      {{ "entity": "Bhimbetka", "label": "ARCHAEOLOGICAL_SITE" }}
    """

    response = llm.invoke(prompt)
    llm_output = response.content
    return llm_output
