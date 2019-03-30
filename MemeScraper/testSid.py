
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
ss = sid.polarity_scores("ONE_DOES_NOT_SIMPLY,THE NUMBER OF PEOPLE WHO USE THE WRONG TEMPLATE IS TOO DAMN HIGH")



print(ss)

x=['19', '88', '11', '7']
y=['0.452', '0.524'  '0.91', '0.503']
plot(y,x)

