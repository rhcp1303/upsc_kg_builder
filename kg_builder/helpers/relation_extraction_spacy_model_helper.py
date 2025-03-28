import spacy
import json
import os
from spacy.training import Example
from spacy.language import Language
from spacy.util import load_config
from spacy.cli.train import train as spacy_train
from spacy.tokens import Doc, Span
from spacy.pipeline import Pipe
from typing import List, Tuple, Dict, Any
from thinc.api import chain, concatenate, Linear, Relu, Softmax, Adam
from thinc.model import Model
import numpy.typing as npt
import numpy as np

@Language.factory("relation_extractor")
class RelationExtractorComponent(Pipe):
    def __init__(self, nlp: Language, name: str = "relation_extractor", model: Model = None):
        super().__init__(name=name)
        self.nlp = nlp
        if model is None:
            self.model = self.create_model()
        self.model.initialize()

    def create_model(self) -> Model:
        embedding_width = 128
        hidden_width = 64
        num_relations = 1

        return Model(
            "relation_extractor",
            chain(
                concatenate(
                    self.nlp.create_pipeline("tok2vec")[0],
                    Linear(embedding_width, nI=self.nlp.get_pipe("tok2vec").model.get_dim("nO")),
                    nO=embedding_width * 2
                ),
                Relu(hidden_width, nI=embedding_width * 2),
                Linear(num_relations, nO=hidden_width),
                Softmax()
            ),
            predict=self.predict_relations
        )

    def predict_relations(self, docs: List[Doc]) -> List[Doc]:
        for doc in docs:
            relations = []
            if doc.ents:
                entity_pairs = [(ent1, ent2) for i, ent1 in enumerate(doc.ents) for j, ent2 in enumerate(doc.ents) if i != j]
                if entity_pairs:
                    inputs = self._get_inputs(doc, entity_pairs)
                    if inputs is not None:
                        scores = self.model.predict(inputs)
                        for (ent1, ent2), score in zip(entity_pairs, scores):
                            if score[0] > 0.5:
                                relations.append((ent1, ent2, "HAS_RELATION")) # Replace with your relation label
            doc._.set("relations", relations)
        return docs

    def _get_inputs(self, doc: Doc, entity_pairs: List[Tuple[Span, Span]]) -> npt.NDArray[np.float32] | None:
        if not entity_pairs:
            return None
        inputs = []
        for ent1, ent2 in entity_pairs:
            max_len = 5
            ent1_start = ent1.start
            ent1_end = ent1.end
            ent2_start = ent2.start
            ent2_end = ent2.end

            context_before1 = doc[max(0, ent1_start - max_len):ent1_start].vector
            context_ent1 = ent1.vector
            context_after1 = doc[ent1_end:min(len(doc), ent1_end + max_len)].vector
            context_before2 = doc[max(0, ent2_start - max_len):ent2_start].vector
            context_ent2 = ent2.vector
            context_after2 = doc[ent2_end:min(len(doc), ent2_end + max_len)].vector

            features = np.concatenate([
                context_before1.mean(axis=0),
                context_ent1,
                context_after1.mean(axis=0),
                context_before2.mean(axis=0),
                context_ent2,
                context_after2.mean(axis=0),
            ])
            inputs.append(features)
        return np.array(inputs, dtype="f")

    def update(self, examples: List[Example], *, drop=0.0, sgd=None, losses: Dict[str, float] = None) -> None:
        if losses is None:
            losses = {}
        losses.setdefault(self.name, 0.0)

        for eg in examples:
            doc = eg.x
            gold_relations = eg.y._.get("relations") if eg.y._.has("relations") else []
            entity_pairs = [(ent1, ent2) for i, ent1 in enumerate(doc.ents) for j, ent2 in enumerate(doc.ents) if i != j]

            if entity_pairs:
                inputs = self._get_inputs(doc, entity_pairs)
                if inputs is not None:
                    targets = np.zeros((len(entity_pairs), 1), dtype="f")
                    for i, (ent1, ent2) in enumerate(entity_pairs):
                        for head, tail, _ in gold_relations:
                            if head.start == ent1.start and head.end == ent1.end and tail.start == ent2.start and tail.end == ent2.end:
                                targets[i, 0] = 1.0
                                break

                    def backprop(d_scores: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
                        losses[self.name] += np.mean((self.model.predict(inputs) - targets) ** 2)
                        return d_scores

                    self.model.ops.grad_softmax(self.model.predict(inputs), targets, backprop)
                    self.model.finish_update(sgd)

def load_relation_data(json_file: str) -> List[Tuple[str, Dict[str, Any]]]:
    with open(json_file, 'r') as f:
        data = json.load(f)
    training_data = []
    for item in data:
        text = item['text']
        relations = item.get('relations', [])
        annotations = {"relations": []}
        for rel in relations:
            head_text = rel['head']
            tail_text = rel['tail']
            relation_label = rel['relation']

            head_start = text.find(head_text)
            tail_start = text.find(tail_text)

            if head_start != -1 and tail_start != -1:
                annotations["relations"].append((head_start, tail_start, relation_label))
        training_data.append((text, annotations))
    return training_data

def create_relation_examples(nlp: spacy.Language, training_data: List[Tuple[str, Dict[str, Any]]]) -> List[Example]:
    examples = []
    for text, annotations in training_data:
        doc = nlp.make_doc(text)
        relations = annotations.get("relations", [])
        if relations:
            doc.ents = [doc.char_span(start, end, label="ENTITY") for start, end, label in get_entity_spans(text, relations)]
            if doc.ents:
                relations_data = []
                for head_start, tail_start, label in relations:
                    head_ents = [ent for ent in doc.ents if ent.start_char == head_start]
                    tail_ents = [ent for ent in doc.ents if ent.start_char == tail_start]
                    if head_ents and tail_ents:
                        relations_data.append((head_ents[0], tail_ents[0], label))
                doc._.set("relations", relations_data)
                examples.append(Example.from_doc(doc, doc))
    return examples

def get_entity_spans(text: str, relations: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    entity_spans = set()
    for head_start, tail_start, label in relations:
        head_end = head_start + len(text[head_start:text.find(' ', head_start) if ' ' in text[head_start:] else None])
        tail_end = tail_start + len(text[tail_start:text.find(' ', tail_start) if ' ' in text[tail_start:] else None])
        entity_spans.add((head_start, head_end, "ENTITY"))
        entity_spans.add((tail_start, tail_end, "ENTITY"))
    return list(entity_spans)

def train_relation_extraction(train_file: str, output_path: str, base_config: str = "en_core_web_sm"):
    config = load_config(base_config)
    config["pipeline"] = [
        {"name": "relation_extractor", "factory": "relation_extractor"}
    ]
    config["components"]["relation_extractor"]["model"] = {
        "@architectures": "relation_extractor",
    }
    config["train"]["pipeline"] = ["relation_extractor"]
    config["train"]["batch_size"] = 32
    config["train"]["iterations"] = 20
    config["train"]["optimizer"] = {"@optimizers": "Adam.v1", "learn_rate": 0.001}
    nlp = spacy.blank(config["nlp"]["lang"])
    nlp.add_pipe("relation_extractor")
    training_data = load_relation_data(train_file)
    examples = create_relation_examples(nlp, training_data)
    config_path = os.path.join(output_path, "config.cfg")
    config.to_disk(config_path)
    spacy_train(config_path, output_path, overrides={"paths.train": train_file, "paths.dev": train_file})