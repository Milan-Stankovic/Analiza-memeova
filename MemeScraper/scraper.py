# import libraries
import urllib.request  as urllib2
from bs4 import BeautifulSoup
import csv
from datetime import datetime

url = 'https://imgflip.com/meme/Hide-the-Pain-Harold?page='
#url = url+1
all_url = []

for i in range(0,100) :
    all_url.append(url+str(i))



for url in all_url:



    # query the website and return the html to the variable ‘page’
    req = urllib2.Request(url, headers={'User-Agent' : "Magic Browser"})
    page = urllib2.urlopen( req )


    # parse the html using beautiful soup and store in variable `soup`
    soup = BeautifulSoup(page, 'html.parser')

    # Take out the <div> of name and get its value
    all_memes = soup.findAll('img', attrs={'class': 'base-img'})

    all_count_info = soup.findAll('div', attrs={'class': 'base-view-count'})


    data = soup.findAll('div', attrs={'class': 'base-unit clearfix'})

    i=0

    skip =[]
    for d in data:
        d= str(d)
        if "base-nsfw-msg" in d:
            skip.append(i)
        i=i+1

    #Ostalo je dodati sve forove

    #print(full_meme)

    txt = []

    for full_meme in all_memes:

        full_meme = str(full_meme)

        i = full_meme.index('|')

        meme_text = full_meme[ i+2: full_meme.index('|',i+1)-1]

        txt.append(meme_text)
        #print(meme_text)


    #print(all_count_info)

    v =[]
    u = []
    c = []
    for count_info in all_count_info:
        count_info = str(count_info)

        i = count_info.index('<')
        j = count_info.find('>')
        h = count_info.find("<", i+1)



        #Sve informacije o pregledima, lajkovima i komentarima
        number_info = count_info[j+1 : h]


        #print(number_info)

        i = number_info.find(' ')

        views = number_info[0:i]

        if views is None:
            views =0;

        v.append(int(views.replace(',', '')))

       # print(views)

        j = number_info.index(' ', i+1)

        h = number_info.index(' ', j+1)

        upvotes = number_info[j+1: h]

        if upvotes is None:
            upvotes =0;

        u.append(int(upvotes.replace(',', '')))
        #print(upvotes)

        try:
            j = number_info.index(' ', h+1)

            h = number_info.index(' ', j+1)
            comments = number_info[j + 1:h]
            c.append(int(comments.replace(',', '')))
        except:
            comments =0;
            c.append(int(comments))



       # print(comments)



    skipAdd =0
    # open a csv file with append, so old data will not be erased


    with open('index1.csv', 'a') as csv_file:
        writer = csv.writer(csv_file)
        for idx, item in enumerate(txt):
            if idx in skip:
                skipAdd = skipAdd + 1
            writer.writerow(['Hide the Pain Harold', item, v[idx + skipAdd], u[idx + skipAdd], c[idx + skipAdd]])


