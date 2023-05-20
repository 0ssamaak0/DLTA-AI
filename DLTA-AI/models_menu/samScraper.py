import requests
from bs4 import BeautifulSoup
import json
import requests

url = 'https://github.com/facebookresearch/segment-anything/blob/main/README.md'
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')

# get all ul inside article tag
ul = soup.find('article').find_all('ul')

models_json = []
# get all li inside ul
li = ul[0].find_all('li')
for i in li:
    model = {}
    #print(i.find('a').text.split(" ")[0]) # get text inside a tag (model name)
    name =  i.find('a').text.split(" ")[0]
    name = name.replace("-", "_").lower()
    model['name'] = name
    #print(i.find('a')['href']) # get href inside a tag)
    model['url'] = i.find('a')['href']
    checkpoint = "mmdetection/checkpoints/" + i.find('a')['href'].split("/")[-1]
    model['checkpoint'] = checkpoint
    models_json.append(model)

with open ("sam_models.json", "w") as f:
            json.dump(models_json, f, indent=4)