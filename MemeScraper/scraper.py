# import libraries
import urllib.request  as urllib2
from bs4 import BeautifulSoup
import csv
from datetime import datetime

url = 'https://imgflip.com/meme/Hide-the-Pain-Harold'

# query the website and return the html to the variable ‘page’
req = urllib2.Request(url, headers={'User-Agent' : "Magic Browser"})
page = urllib2.urlopen( req )

# parse the html using beautiful soup and store in variable `soup`
soup = BeautifulSoup(page, 'html.parser')

# Take out the <div> of name and get its value
full_meme = soup.findAll('img', attrs={'class': 'base-img'})

count_info = soup.findAll('div', attrs={'class': 'base-view-count'})


#Ostalo je dodati sve forove

#print(full_meme)

full_meme = str(full_meme)

i = full_meme.index('|')


meme_text = full_meme[ i+1: full_meme.index('|',i+1)]

print(meme_text)



count_info = str(count_info)

i = count_info.index('<')
j = count_info.find('>')
h = count_info.find("<", i+1)



#Sve informacije o pregledima, lajkovima i komentarima
number_info = count_info[j+1 : h]


#print(number_info)

i = number_info.find(' ')

views = number_info[0:i+1]

print(views)

j = number_info.index(' ', i+1)

h = number_info.index(' ', j+1)

upvotes = number_info[j+1: h]

print(upvotes)

j = number_info.index(' ', h+1)

h = number_info.index(' ', j+1)

comments = number_info[j+1:h]

print(comments)




# open a csv file with append, so old data will not be erased
#with open('index.csv', 'a') as csv_file:
# writer = csv.writer(csv_file)
# writer.writerow([name, price, datetime.now()])


