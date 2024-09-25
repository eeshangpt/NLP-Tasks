import os
from pprint import pprint
from warnings import filterwarnings

from dotenv import load_dotenv

# from classification import (
#     SentimentClassifier,
#     TFSentimentClassifier,
#     ZeroShotClassifier,
#     TextClassifier,
# )
# from named_entity import NamedEntityRecognizer, TFNamedEntityRecognizer
from question_answering import SimpleOracle

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "dev.env"))
# sys.path.append(os.getenv("PACKAGE_DIR"))
filterwarnings("ignore")

# zs_classifier = ZeroShotClassifier(categories=["sci-fi", "mystery", "romance"])
# zs_classifier_m = ZeroShotClassifier(
#     categories=["sci-fi", "thriller", "mystery", "romance"], is_multi_label=True
# )
#
# ner_sentences_2 = [
#     """The story follows ten people who are brought together, for various reasons, to an empty mansion on an island. The mysterious hosts of this strange party are not present, but left instructions for two of the ten to tend the house as the housekeeper and cook. As the days unfold in accordance with the lyrics of a nursery rhyme, each invitee is forced to face the music (literally) and bear the consequences of their troubling pasts, as death will come for them one by one.""",
#     """Gone Girl is the ultimate mystery puzzle for the modern media age. Devoted wife Amy’s sudden disappearance throws Nick Dunne into a hailstorm of suspicion — from her parents to his neighbours to the investigators, everyone leans towards believing that he is somehow responsible. Nick himself becomes aware of how his wife viewed him, as well as how little he knows of her, when stories of her emerge from friends he’s never heard of.""",
# ]
# for _, text in enumerate(ner_sentences_2):
#     single_class = zs_classifier_m.predict(text)
#     print(_, single_class, sep="\n", end="\n\n")
#
#
# sentiment_classifier = SentimentClassifier()
# sentiment_sentences = ["I love this movie!", "I hate this movie!"]
# print(sentiment_classifier.predict(sentiment_sentences))
#
# tf_snet_class = TFSentimentClassifier()
# print(tf_snet_class.predict("I love this movie!"))
#
# multi_sentiment_classifier = TextClassifier()
# multiple_sentiment_sentences = [
#     "The team's hard work and dedication paid off with a successful project completion.",
#     "I'm disappointed with the service I received at the restaurant yesterday.",
#     "The meeting is scheduled for 2:00 PM.",
#     "Yeah. That crap of a car caused a lot of inconvenience.",
#     "I really liked the coffee, the tea and most importantly the company.",
# ]
# print(multi_sentiment_classifier.predict(multiple_sentiment_sentences))
#
# finance_classifier = TextClassifier(model_name="ProsusAI/finbert")
# finance_sentences = [
#     "Stock prices plummeted amid uncertainties surrounding US election.",
#     "Investors are optimistic about the future, contributing to a bullish trend in the stock market with a gain of 150 points.",
# ]
# print(finance_classifier.predict(finance_sentences))
#
# ner_model = NamedEntityRecognizer(aggregation_strategy="simple")
# ner_sentences_1 = [
#     "Apple Inc. is set to launch its latest iPhone this month.",
#     "We went to The Louvre and the Eiffel Tower",
# ]
# pprint(ner_model.predict(ner_sentences_1))
#
# tf_ner_model = TFNamedEntityRecognizer()
# ner_sentences_2 = [
#     "Apple Inc. is set to launch its latest iPhone this month.",
#     "Apple Inc. was founded by Steve Wozniac, Steve Jobs, and Ronald Wayne.",
# ]
# for text in ner_sentences_2:
#     pprint(tf_ner_model.predict(text))

oracle = SimpleOracle()
contexts = [
    """In the vast realm of space exploration, humanity has embarked on a journey to unravel the mysteries of the cosmos.
Over the decades, numerous missions have been launched to explore our solar system and beyond. One notable achievement
is the exploration of Mars, where rovers like Curiosity have provided valuable data about the Martian surface. 
Additionally, the Cassini-Huygens mission offered insights into the majestic rings of Saturn and its enigmatic moon, Titan.
The International Space Station (ISS) serves as a microgravity and space environment research laboratory, fostering
international collaboration in space. Advancements in space telescopes, such as the Hubble Space Telescope,
have expanded our understanding of distant galaxies and the birth of stars. As we continue to push the boundaries of
space exploration, questions arise about potential extraterrestrial life, the feasibility of human colonization on other
planets, and the future of interstellar travel."""
]
questions = [
    "Which rover played a crucial role in this exploration?",
    "Which celestial body was the focus of NASA's recent mission involving the Perseverance rover?",
    "What insights did the Cassini-Huygens mission offer, and which celestial bodies were the focus of its exploration?",
    "Why is the Curiosity rover significant in the field of space exploration?",
]
answers = oracle.predict(
    questions=questions,
    contexts=contexts,
)
pprint(answers.to_dict())
