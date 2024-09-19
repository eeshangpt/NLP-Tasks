import os
from pprint import pprint
from warnings import filterwarnings

from dotenv import load_dotenv

from classification import (
    SentimentClassifier,
    TFSentimentClassifier,
    ZeroShotClassifier,
    TextClassifier,
)
from named_entity import NamedEntityRecognizer
from named_entity import TFNamedEntityRecognizer

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "dev.env"))

filterwarnings("ignore")

zs_classifier = ZeroShotClassifier(categories=["sci-fi", "mystery", "romance"])
zs_classifier_m = ZeroShotClassifier(
    categories=["sci-fi", "thriller", "mystery", "romance"], is_multi_label=True
)

texts = [
    """The story follows ten people who are brought together, for various reasons, to an empty mansion on an island. The mysterious hosts of this strange party are not present, but left instructions for two of the ten to tend the house as the housekeeper and cook. As the days unfold in accordance with the lyrics of a nursery rhyme, each invitee is forced to face the music (literally) and bear the consequences of their troubling pasts, as death will come for them one by one.""",
    """Gone Girl is the ultimate mystery puzzle for the modern media age. Devoted wife Amy’s sudden disappearance throws Nick Dunne into a hailstorm of suspicion — from her parents to his neighbours to the investigators, everyone leans towards believing that he is somehow responsible. Nick himself becomes aware of how his wife viewed him, as well as how little he knows of her, when stories of her emerge from friends he’s never heard of.""",
]
for _, text in enumerate(texts):
    single_class = zs_classifier_m.predict(text)
    print(_, single_class, sep="\n", end="\n\n")


sentiment_classifier = SentimentClassifier()
print(sentiment_classifier.predict(["I love this movie!", "I hate this movie!"]))

tf_snet_class = TFSentimentClassifier()
print(tf_snet_class.predict("I love this movie!"))

multi_sentiment_classifier = TextClassifier()
print(
    multi_sentiment_classifier.predict(
        [
            "The team's hard work and dedication paid off with a successful project completion.",
            "I'm disappointed with the service I received at the restaurant yesterday.",
            "The meeting is scheduled for 2:00 PM.",
            "Yeah. That crap of a car caused a lot of inconvenience.",
            "I really liked the coffee, the tea and most importantly the company.",
        ]
    )
)

finance_classifier = TextClassifier(model_name="ProsusAI/finbert")
print(
    finance_classifier.predict(
        [
            "Stock prices plummeted amid uncertainties surrounding US election.",
            "Investors are optimistic about the future, contributing to a bullish trend in the stock market with a gain of 150 points.",
        ]
    )
)

ner_model = NamedEntityRecognizer(aggregation_strategy="simple")
pprint(
    ner_model.predict(
        [
            "Apple Inc. is set to launch its latest iPhone this month.",
            "We went to The Louvre and the Eiffel Tower",
        ]
    )
)

tf_ner_model = TFNamedEntityRecognizer()
texts = [
    "Apple Inc. is set to launch its latest iPhone this month.",
    "Apple Inc. was founded by Steve Wozniac, Steve Jobs, and Ronald Wayne.",
]
for text in texts:
    pprint(tf_ner_model.predict(text))
    print("===+===+===")
