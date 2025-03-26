from django.core.management.base import BaseCommand
import logging
import spacy

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'This is a utility management command for training spacy model'

    def handle(self, *args, **options):
        output_directory = "trained_spacy_model"
        nlp_trained = spacy.load(output_directory)
        test_text = '''

        In the subsequent century, stupas were elaborately built with certain additions like the enclosing of the circumambulatory path with railings and sculptural decoration. There were numerous stupas constructed earlier but expansions or new additions were made in the second century BCE. The stupa consists of a cylindrical drum and a circular anda with a harmika and chhatra on the top which remain consistent throughout with minor variations and changes in shape and size. Apart from the circumambulatory path, gateways were added. Thus, with the elaborations in stupa architecture, there was ample space for the architects and sculptors to plan elaborations and to carve out images.
During the early phase of Buddhism, Buddha is depicted symbolically through footprints, stupas, lotus throne, chakra, etc. This indicates either simple worship, or paying respect, or at times depicts historisisation of life events. Gradually narrative became a part of the Buddhist tradition. Thus events from the life of the Buddha, the Jataka stories, were depicted on the railings and torans of the stupas. Mainly synoptic narrative, continuous narrative and episodic narrative are used in the pictorial tradition. While events from the life of the Buddha became an important theme in all the Buddhist monuments, the Jataka stories also became equally important for sculptural decorations. The main events associated with the Buddha’s life which were frequently depicted were events related to the birth, renunciation, enlightenment, dhammachakra- pravartana, and mahaparinibbana (death). Among the Jataka stories that are frequently depicted are Chhadanta Jataka, Vidurpundita Jataka, Ruru Jataka, Sibi Jataka, Vessantara Jataka and Shama Jataka.
  Stupa worship, Bharhut
EXERCISE
1. Do you think that the art of making sculptures in India began during the Mauryan period?
2. What was the significance of the stupa and how did stupa architecture develop?
3. Which were the four events in the life of the Buddha which have been depicted in different forms of Buddhist art? What did these events symbolise?
4. What are the Jatakas? How do the Jatakas relate to Buddhism? Find out.

     POST-MAURYAN TREND4S IN INDIAN ART AND ARCHITECTURE
 FROM the second century BCE onwards, various rulers established their control over the vast Mauryan Empire: the Shungas, Kanvas, Kushanas and Guptas in the north and parts of central India; the Satvahanas, Ikshavakus, Abhiras, Vakataks in southern and western India. Incidentally, the period of the second century BCE also marked the rise of the main Brahmanical sects such as the Vaishnavas and the Shaivas. There are numerous sites dating back to the second century BCE in India. Some of the prominent examples of the finest sculpture are found at Vidisha, Bharhut (Madhya Pradesh), Bodhgaya (Bihar), Jaggayyapeta (Andhra Pradesh), Mathura (Uttar Pradesh), Khandagiri-Udaigiri (Odisha), Bhaja near Pune and Pavani near Nagpur (Maharashtra).
Bharhut
Bharhut sculptures are tall like the images of Yaksha and Yakhshini in the Mauryan period, modelling of the sculptural volume is in low relief maintaining linearity. Images stick to the picture plane. In the relief panels depicting narratives, illusion of three-dimensionality is shown with tilted perspective. Clarity in the narrative is enhanced by selecting main events. At Bharhut, narrative panels are shown with fewer characters but as the time progresses, apart from the main character in the story, others also start appearing in the picture space. At times more than one event at one geographical place is clubbed in the picture space or only a single main event is depicted in the pictorial space.
Availability of the space is utilised to the maximum by the sculptors. Folded hands in the narratives as well as single figures of the Yakhshas and Yakshinis are shown flat clinging to the chest. But in some cases, especially in later times, the hands are shown with the natural projection against the chest. Such examples show how artisans who were working at a collective level had to
Yakshini, Bharhut

        '''

        doc = nlp_trained(test_text)
        for ent in doc.ents:
            print(ent.text, ent.label_)
