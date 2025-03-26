import spacy
from spacy.training import Example
import random

def train_spacy_ner(train_data, output_path, n_iter=100):
    nlp = spacy.load("en_core_web_sm")
    ner = nlp.get_pipe("ner")
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.resume_training()
        for itn in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            for text, annotations in train_data:
                example = Example.from_dict(nlp.make_doc(text), annotations)
                nlp.update([example], drop=0.5, losses=losses, sgd=optimizer)
            print(f"Losses at iteration {itn}: {losses}")

    nlp.to_disk(output_path)
    print(f"Trained model saved to {output_path}")
