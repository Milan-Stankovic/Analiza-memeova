
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
ss = sid.polarity_scores("ONE_DOES_NOT_SIMPLY,THE NUMBER OF PEOPLE WHO USE THE WRONG TEMPLATE IS TOO DAMN HIGH")



print(ss)

