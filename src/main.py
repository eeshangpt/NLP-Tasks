from warnings import filterwarnings

from classification import MultiClassSentimentClassifier

# from classification import SentimentClassifier
# from classification import TFSentimentClassifier
# from classification import ZeroShotClassifier

filterwarnings("ignore")

# zs_classifier = ZeroShotClassifier(categories=["sci-fi", "mystery", "romance"])
# zs_classifier_m = ZeroShotClassifier(categories=["sci-fi", 'thriller', "mystery", "romance"], is_multi_label=True)
# #
# texts = [
#     """The story follows ten people who are brought together, for various reasons, to an empty mansion on an island. The mysterious hosts of this strange party are not present, but left instructions for two of the ten to tend the house as the housekeeper and cook. As the days unfold in accordance with the lyrics of a nursery rhyme, each invitee is forced to face the music (literally) and bear the consequences of their troubling pasts, as death will come for them one by one.""",
#     """Gone Girl is the ultimate mystery puzzle for the modern media age. Devoted wife Amy’s sudden disappearance throws Nick Dunne into a hailstorm of suspicion — from her parents to his neighbours to the investigators, everyone leans towards believing that he is somehow responsible. Nick himself becomes aware of how his wife viewed him, as well as how little he knows of her, when stories of her emerge from friends he’s never heard of.""",
# ]
# for _, text in enumerate(texts):
#     single_class = zs_classifier_m.predict(text)
#     print(_, single_class, sep='\n', end='\n\n')


# sentiment_classifier = SentimentClassifier()
# print(sentiment_classifier.predict(["I love this movie!", "I hate this movie!"]))

# tf_snet_class = TFSentimentClassifier()
# print(tf_snet_class.predict("I love this movie!"))

multi_sentiment_classifier = MultiClassSentimentClassifier()
print(
    multi_sentiment_classifier.predict(
        [
            "The team's hard work and dedication paid off with a successful project completion.",
            "I'm disappointed with the service I received at the restaurant yesterday.",
            "The meeting is scheduled for 2:00 PM.",
            "Yeah. That crap of a car caused a lot of inconvenience.",
            "I really liked the coffee and company."
        ]
    )
)
