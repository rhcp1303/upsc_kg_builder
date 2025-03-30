import json

from neo4j import GraphDatabase, Result

uri = "bolt://localhost:7687"
username = "neo4j"
password = "neo4j1234!"
database_name = "neo4j"

driver = GraphDatabase.driver(uri, database=database_name, auth=(username, password))


def create_entity_node(tx, entity_text, label):
    query = """
    MERGE (e:Entity {text: $entity_text, label: $label})
    RETURN e
    """
    result = tx.run(query, entity_text=entity_text, label=label)
    return result.single()[0]


def create_relationship(tx, entity1_data, entity2_data, relation):
    query = """
    MERGE (e1:Entity {text: $entity1_text, label: $entity1_label})
    MERGE (e2:Entity {text: $entity2_text, label: $entity2_label})
    MERGE (e1)-[r:RELATION {name: $relation}]->(e2)
    RETURN r
    """
    tx.run(
        query,
        entity1_text=entity1_data["entity_text"],
        entity1_label=entity1_data["label"],
        entity2_text=entity2_data["entity_text"],
        entity2_label=entity2_data["label"],
        relation=relation
    )

def insert_relationships(tx, relationships_list):
    for relationship_data in relationships_list:
        print(relationship_data)
        create_relationship(tx, relationship_data["entity1"], relationship_data["entity2"],
                            relationship_data["relation"])


def create_and_insert_relations_into_kg(json_file_path):
    with open(json_file_path, 'r') as f:
        json_data = f.read()
    data = json.loads(json_data)
    with driver.session() as session:
        for relationships in data:
            try:
                session.execute_write(insert_relationships, relationships)
            except Exception as e:
                print(e)
                continue

    print("Relationships inserted successfully into Neo4j.")
    driver.close()
