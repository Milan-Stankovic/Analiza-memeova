# import libraries
import urllib.request  as urllib2
from bs4 import BeautifulSoup
import csv
import io
from datetime import datetime

url = 'https://memegenerator.net/One-Does-Not-Simply/images/popular'
#url = url+1
all_url = []

all_url.append(url);
url=url+"/alltime/page/"

for i in range(2,20):
    all_url.append(url+str(i))



for url in all_url:

    print(url);

    # query the website and return the html to the variable ‘page’
    req = urllib2.Request(url, headers={'User-Agent' : "Magic Browser"})
    page = urllib2.urlopen( req )

    # parse the html using beautiful soup and store in variable `soup`
    soup = BeautifulSoup(page, 'html.parser')

    # Take out the <div> of name and get its value
    all_memes = soup.findAll('div', attrs={'class': 'gallery-img'})

    txt = [];

    #print(all_memes);

    for them_memes in all_memes:
        temparr = ['One dones not simply']
        the_meme = them_memes.findAll('div', attrs={'class': 'only-above-768'})

        first_text = the_meme[0].findAll('div', attrs={'class': 'optimized-instance-text0'})
        secont_text = the_meme[0].findAll('div', attrs={'class': 'optimized-instance-text1'})

        if not first_text:
            print("1-Empty tag error!!!")
        if not secont_text:
            print("2-Empty tag error!!!")

        if len(first_text)>0 and len(secont_text)>0:
            temp = ""
            if len(first_text[0].contents)>0:
               temp = temp + first_text[0].contents[0]+ " ";
            else:
                print("Content error.......")
            if len(secont_text[0].contents)>0:
               temp = temp + secont_text[0].contents[0];
            else:
                print("Content error.......")
            temparr.append(temp.upper())
        else:
            print("Strange erorr!!!")

        score = them_memes.findAll('div', attrs={'class': 'score'})
        #dodao sam da format fajla bude isti, posto ovde nema views, stavio sam 0
        temparr.append(0)
        temparr.append(int(score[0].contents[0].strip().replace(',', '')))

        coment_count = them_memes.findAll('span', attrs={'class': 'comments-count'})
        temparr.append(int(coment_count[0].contents[0].strip().replace(',', '')))
        txt.append(temparr)

    with open('test.csv', 'a', encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        for idx, item in enumerate(txt):
            writer.writerow(item)
