import re
import requests
from bs4 import BeautifulSoup
import urllib.request
import os

BASE_CHAR_PAGE = 'https://southpark.fandom.com/wiki/Portal:Characters'
BASE_DOWNLOAD_PATH = 'data/southpark/{number}.{imgformat}'
BASE_DOWNLOAD_PATH = os.path.join(os.getcwd(), BASE_DOWNLOAD_PATH)
print(BASE_DOWNLOAD_PATH)

def download_image(url, current_index, format):
    try:
        download_path = BASE_DOWNLOAD_PATH.format(number=current_index,imgformat=format)
        #print(download_path)
        urllib.request.urlretrieve(url, download_path)
    except Exception as e:
        print("couldn't download image " + url)
        print(e)

def get_all_image_urls():
    all_urls = []
    response = requests.get(BASE_CHAR_PAGE)
    soup = BeautifulSoup(response.text, 'html.parser')
    img_tags = soup.find_all('img')
    urls = [img['src'] for img in img_tags]
    print(len(urls))
    j = 0
    for url in urls:
        #print(url)
        if "https://static.wikia.nocookie.net/southpark/images" in url and "revision" in url:
            short_url = url.split("/revision")[0]
            all_urls.append(short_url)
            j = j+1
    return all_urls

def download_all_images():
    all_urls = get_all_image_urls()
    with open("myimglist.txt", "w") as o: 
        for url in all_urls:    
            o.write(url + '\n')
        
    # i = 0
    # for url in all_urls:
    #     if "jpeg" in url:
    #         download_image(url, i, "jpeg")
    #     else:
    #         download_image(url, i, "png")
    #     i = i + 1


download_all_images()