import requests
from bs4 import BeautifulSoup
import json
import requests
from tqdm import tqdm

url = 'https://raw.githubusercontent.com/open-mmlab/mmdetection/2.x/README.md'
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')

# get all ul tags
ul_tags = soup.find_all('ul')
instance_seg = ul_tags[1]

# for the all ul tags, make a dict of each li tag contents and href and append to a list
li_tags = instance_seg.find_all('li')
li_tags_list = []
for li in li_tags:
    li_tags_list.append(
        {'name': li.find('a').text, 'link': "https://github.com/open-mmlab/mmdetection/tree/2.x/" + li.find('a')['href'], })
model_id = 0
col_names = ["id", "Model", "Model Name", "Backbone", "Lr schd", "Memory (GB)",
             "Inference Time (fps)", "Box AP", "Mask AP", "Config", "Checkpoint_link"]

tr_tags_list = []
# =================================================================================================

# Mask R-CNN (ICCV'2017)
url = li_tags_list[0]['link']
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')

# get all tr tags from the first table in the page, for each tr tag, get all td tags and append to a dictionary, append all dictionaries to a list
table = soup.find_all('table')[0]
tr_tags = table.find_all('tr')

for tr in tr_tags[1:]:
    td_tags = tr.find_all('td')
    td_tag_dict = {}
    td_tag_dict["id"] = model_id
    model_id += 1
    td_tag_dict["Model"] = "Mask R-CNN"
    td_tag_dict["Model Name"] = "Mask R-CNN"
    for i in range(len(td_tags)):
        if i == 0:
            td_tag_dict["Backbone"] = td_tags[i].text
        elif i == 1:
            td_tag_dict["Style"] = td_tags[i].text
        elif i == 2:
            td_tag_dict["Lr schd"] = td_tags[i].text
        elif i == 3:
            td_tag_dict["Memory (GB)"] = td_tags[i].text
        elif i == 4:
            td_tag_dict["Inference Time (fps)"] = td_tags[i].text
        elif i == 5:
            td_tag_dict["box AP"] = td_tags[i].text
        elif i == 6:
            td_tag_dict["mask AP"] = td_tags[i].text
        elif i == 7:
            td_tag_dict["Config"] = td_tags[i].find('a')['href']
        elif i == 8:
            td_tag_dict["Checkpoint_link"] = td_tags[i].find('a')['href']

    tr_tags_list.append(td_tag_dict)

# =================================================================================================

# Cascade Mask R-CNN (CVPR'2018)
url = li_tags_list[1]['link']
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')

# get all tr tags from the second table in the page, for each tr tag, get all td tags and append to a dictionary, append all dictionaries to a list
table = soup.find_all('table')[1]
tr_tags = table.find_all('tr')

for tr in tr_tags[1:]:
    td_tags = tr.find_all('td')
    td_tag_dict = {}
    td_tag_dict["id"] = model_id
    model_id += 1
    td_tag_dict["Model"] = "Cascade Mask R-CNN"
    td_tag_dict["Model Name"] = "Cascade Mask R-CNN"
    for i in range(len(td_tags)):
        if i == 0:
            td_tag_dict["Backbone"] = td_tags[i].text
        elif i == 1:
            td_tag_dict["Style"] = td_tags[i].text
        elif i == 2:
            td_tag_dict["Lr schd"] = td_tags[i].text
        elif i == 3:
            td_tag_dict["Memory (GB)"] = td_tags[i].text
        elif i == 4:
            td_tag_dict["Inference Time (fps)"] = td_tags[i].text
        elif i == 5:
            td_tag_dict["box AP"] = td_tags[i].text
        elif i == 6:
            td_tag_dict["mask AP"] = td_tags[i].text
        elif i == 7:
            td_tag_dict["Config"] = td_tags[i].find('a')['href']
        elif i == 8:
            td_tag_dict["Checkpoint_link"] = td_tags[i].find('a')['href']

    tr_tags_list.append(td_tag_dict)
# =================================================================================================
# Mask Scoring R-CNN (CVPR'2019)
url = li_tags_list[2]['link']
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')

# get all tr tags from the second table in the page, for each tr tag, get all td tags and append to a dictionary, append all dictionaries to a list
table = soup.find_all('table')[0]
tr_tags = table.find_all('tr')

for tr in tr_tags[1:]:
    td_tags = tr.find_all('td')
    td_tag_dict = {}
    td_tag_dict["id"] = model_id
    model_id += 1
    td_tag_dict["Model"] = "Mask Scoring R-CNN"
    td_tag_dict["Model Name"] = "Mask Scoring R-CNN"
    for i in range(len(td_tags)):
        if i == 0:
            td_tag_dict["Backbone"] = td_tags[i].text
        elif i == 1:
            td_tag_dict["Style"] = td_tags[i].text
        elif i == 2:
            td_tag_dict["Lr schd"] = td_tags[i].text
        elif i == 3:
            td_tag_dict["Memory (GB)"] = td_tags[i].text
        elif i == 4:
            td_tag_dict["Inference Time (fps)"] = td_tags[i].text
        elif i == 5:
            td_tag_dict["box AP"] = td_tags[i].text
        elif i == 6:
            td_tag_dict["mask AP"] = td_tags[i].text
        elif i == 7:
            td_tag_dict["Config"] = td_tags[i].find('a')['href']
        elif i == 8:
            td_tag_dict["Checkpoint_link"] = td_tags[i].find('a')['href']

    tr_tags_list.append(td_tag_dict)
# =================================================================================================# Hybrid Task Cascade (CVPR'2019)
url = li_tags_list[3]['link']
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')

# get all tr tags from the second table in the page, for each tr tag, get all td tags and append to a dictionary, append all dictionaries to a list
table = soup.find_all('table')[0]
tr_tags = table.find_all('tr')

for tr in tr_tags[1:]:
    td_tags = tr.find_all('td')
    td_tag_dict = {}
    td_tag_dict["id"] = model_id
    model_id += 1
    td_tag_dict["Model"] = "Hybrid Task Cascade"
    td_tag_dict["Model Name"] = "Hybrid Task Cascade"
    for i in range(len(td_tags)):
        if i == 0:
            td_tag_dict["Backbone"] = td_tags[i].text
        elif i == 1:
            td_tag_dict["Style"] = td_tags[i].text
        elif i == 2:
            td_tag_dict["Lr schd"] = td_tags[i].text
        elif i == 3:
            td_tag_dict["Memory (GB)"] = td_tags[i].text
        elif i == 4:
            td_tag_dict["Inference Time (fps)"] = td_tags[i].text
        elif i == 5:
            td_tag_dict["box AP"] = td_tags[i].text
        elif i == 6:
            td_tag_dict["mask AP"] = td_tags[i].text
        elif i == 7:
            td_tag_dict["Config"] = td_tags[i].find('a')['href']
        elif i == 8:
            td_tag_dict["Checkpoint_link"] = td_tags[i].find('a')['href']

    tr_tags_list.append(td_tag_dict)
# # =================================================================================================
# # YOLACT (ICCV'2019)
url = li_tags_list[4]['link']
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')

# get all tr tags from the second table in the page, for each tr tag, get all td tags and append to a dictionary, append all dictionaries to a list
table = soup.find_all('table')[0]
tr_tags = table.find_all('tr')

for tr in tr_tags[1:]:
    td_tags = tr.find_all('td')
    td_tag_dict = {}
    td_tag_dict["id"] = model_id
    model_id += 1
    td_tag_dict["Model"] = "YOLACT"
    td_tag_dict["Model Name"] = "YOLACT"
    for i in range(len(td_tags)):
        if i == 2:
            td_tag_dict["Backbone"] = td_tags[i].text
            td_tag_dict["Style"] = "-"
            td_tag_dict["Lr schd"] = "-"
            td_tag_dict["Memory (GB)"] = "-"
        elif i == 3:
            td_tag_dict["Inference Time (fps)"] = td_tags[i].text
        elif i == 4:
            td_tag_dict["box AP"] = "-"
            td_tag_dict["mask AP"] = td_tags[i].text
        elif i == 6:
            td_tag_dict["Config"] = td_tags[i].find('a')['href']
        elif i == 7:
            td_tag_dict["Checkpoint_link"] = td_tags[i].find('a')['href']

    tr_tags_list.append(td_tag_dict)
# remove the model at index 1tr_tags_list.pop(1)
# =================================================================================================

# InstaBoost (ICCV'2019) cancelled ❌ requires custom installation

# =================================================================================================

# SOLO (ECCV'2020)
url = li_tags_list[6]['link']
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')

# get all tr tags from the second table in the page, for each tr tag, get all td tags and append to a dictionary, append all dictionaries to a list
tables = soup.find_all('table')

for tableno, table in enumerate(tables):
    tr_tags = table.find_all('tr')
    for trno, tr in enumerate(tr_tags[1:]):
        td_tags = tr.find_all('td')
        td_tag_dict = {}
        td_tag_dict["id"] = model_id
        model_id += 1
        td_tag_dict["Model"] = "SOLO"
        if tableno == 0:
            td_tag_dict["Model Name"] = "SOLO"
        elif tableno == 1:
            td_tag_dict["Model Name"] = "Decoupled SOLO"
        elif tableno == 2:
            td_tag_dict["Model Name"] = "Decoupled Light SOLO"
        for i in range(len(td_tags)):
            if i == 0:
                td_tag_dict["Backbone"] = td_tags[i].text
            elif i == 1:
                td_tag_dict["Style"] = td_tags[i].text
            elif i == 3:
                td_tag_dict["Lr schd"] = td_tags[i].text
            elif i == 4:
                td_tag_dict["Memory (GB)"] = td_tags[i].text
            elif i == 5:
                td_tag_dict["Inference Time (fps)"] = td_tags[i].text
            elif i == 6:
                td_tag_dict["box AP"] = "-"
                td_tag_dict["mask AP"] = td_tags[i].text
            elif i == 7:
                td_tag_dict["Config"] = "https://github.com/open-mmlab/mmdetection/tree/master/"
                if trno == 0 and tableno == 0:
                    td_tag_dict["Config"] += "configs/solo/solo_r50_fpn_1x_coco.py"
                elif trno == 1 and tableno == 0:
                    td_tag_dict["Config"] += "configs/solo/solo_r50_fpn_3x_coco.py"
                elif trno == 0 and tableno == 1:
                    td_tag_dict["Config"] += "configs/solo/decoupled_solo_r50_fpn_1x_coco.py"
                elif trno == 1 and tableno == 1:
                    td_tag_dict["Config"] += "configs/solo/decoupled_solo_r50_fpn_3x_coco.py"
                elif trno == 0 and tableno == 2:
                    td_tag_dict["Config"] += "configs/solo/decoupled_solo_light_r50_fpn_3x_coco.py"
                td_tag_dict["Checkpoint_link"] = td_tags[i].find('a')['href']
        tr_tags_list.append(td_tag_dict)
# =================================================================================================
# PointRend (CVPR'2020) cancelled ❌ caffe only

# =================================================================================================

# DetectoRS (ArXiv'2020) cancelled ❌ complicated format

# =================================================================================================

# SOLOv2 (NeurIPS'2020)

url = li_tags_list[9]['link']
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')

# get all tr tags from the second table in the page, for each tr tag, get all td tags and append to a dictionary, append all dictionaries to a list
tables = soup.find_all('table')


for tableno, table in enumerate(tables):
    tr_tags = table.find_all('tr')
    for trno, tr in enumerate(tr_tags[1:]):
        td_tags = tr.find_all('td')
        td_tag_dict = {}
        td_tag_dict["id"] = model_id
        model_id += 1
        td_tag_dict["Model"] = "SOLOv2"
        if tableno == 0:
            td_tag_dict["Model Name"] = "SOLOv2"
        elif tableno == 1:
            td_tag_dict["Model Name"] = "Light SOLOv2"
        for i in range(len(td_tags)):
            if i == 0:
                td_tag_dict["Backbone"] = td_tags[i].text
            elif i == 1:
                td_tag_dict["Style"] = td_tags[i].text
            elif i == 3:
                td_tag_dict["Lr schd"] = td_tags[i].text
            elif i == 4:
                td_tag_dict["Memory (GB)"] = td_tags[i].text
                td_tag_dict["Inference Time (fps)"] = "-"
                td_tag_dict["box AP"] = "-"
            elif i == 5:
                td_tag_dict["mask AP"] = td_tags[i].text
            elif i == 6:
                td_tag_dict["Config"] = td_tags[i].find('a')['href']
            elif i == 7:
                td_tag_dict["Checkpoint_link"] = td_tags[i].find('a')['href']

        tr_tags_list.append(td_tag_dict)
# =================================================================================================

# SCNet (AAAI'2021)

url = li_tags_list[10]['link']
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')

# get all tr tags from the second table in the page, for each tr tag, get all td tags and append to a dictionary, append all dictionaries to a list
table = soup.find_all('table')[0]
tr_tags = table.find_all('tr')

for tr in tr_tags[1:]:
    td_tags = tr.find_all('td')
    td_tag_dict = {}
    td_tag_dict["id"] = model_id
    model_id += 1
    td_tag_dict["Model"] = "SCNet"
    td_tag_dict["Model Name"] = "SCNet"
    for i in range(len(td_tags)):
        if i == 0:
            td_tag_dict["Backbone"] = td_tags[i].text
        elif i == 1:
            td_tag_dict["Style"] = td_tags[i].text
        elif i == 2:
            td_tag_dict["Lr schd"] = td_tags[i].text
        elif i == 3:
            td_tag_dict["Memory (GB)"] = td_tags[i].text
        elif i == 4:
            td_tag_dict["Inference Time (fps)"] = td_tags[i].text
        elif i == 5:
            td_tag_dict["box AP"] = td_tags[i].text
        elif i == 6:
            td_tag_dict["mask AP"] = td_tags[i].text
        elif i == 9:
            td_tag_dict["Config"] = td_tags[i].find('a')['href']
        elif i == 10:
            td_tag_dict["Checkpoint_link"] = td_tags[i].find('a')['href']

    tr_tags_list.append(td_tag_dict)
# =================================================================================================

# QueryInst (ICCV'2021)

url = li_tags_list[11]['link']
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')

# get all tr tags from the second table in the page, for each tr tag, get all td tags and append to a dictionary, append all dictionaries to a list
table = soup.find_all('table')[0]
tr_tags = table.find_all('tr')

for tr in tr_tags[1:]:
    td_tags = tr.find_all('td')
    td_tag_dict = {}
    td_tag_dict["id"] = model_id
    model_id += 1
    td_tag_dict["Model"] = "QueryInst"
    td_tag_dict["Model Name"] = "QueryInst"
    for i in range(len(td_tags)):
        if i == 1:
            td_tag_dict["Backbone"] = td_tags[i].text
        elif i == 2:
            td_tag_dict["Style"] = td_tags[i].text
        elif i == 3:
            td_tag_dict["Lr schd"] = td_tags[i].text
            td_tag_dict["Memory (GB)"] = "-"
            td_tag_dict["Inference Time (fps)"] = "-"
        elif i == 7:
            td_tag_dict["box AP"] = td_tags[i].text
        elif i == 8:
            td_tag_dict["mask AP"] = td_tags[i].text
        elif i == 9:
            td_tag_dict["Config"] = td_tags[i].find('a')['href']
        elif i == 10:
            td_tag_dict["Checkpoint_link"] = td_tags[i].find('a')['href']

    tr_tags_list.append(td_tag_dict)
# =================================================================================================

# Mask2Former (ArXiv'2021)

url = li_tags_list[12]['link']
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')

# get all tr tags from the second table in the page, for each tr tag, get all td tags and append to a dictionary, append all dictionaries to a list
table = soup.find_all('table')[1]
tr_tags = table.find_all('tr')

for tr in tr_tags[1:]:
    td_tags = tr.find_all('td')
    td_tag_dict = {}
    td_tag_dict["id"] = model_id
    model_id += 1
    td_tag_dict["Model"] = "Mask2Former"
    td_tag_dict["Model Name"] = "Mask2Former"
    for i in range(len(td_tags)):
        if i == 0:
            td_tag_dict["Backbone"] = td_tags[i].text
        elif i == 1:
            td_tag_dict["Style"] = td_tags[i].text
        elif i == 3:
            td_tag_dict["Lr schd"] = td_tags[i].text
        elif i == 4:
            td_tag_dict["Memory (GB)"] = td_tags[i].text
        elif i == 5:
            td_tag_dict["Inference Time (fps)"] = td_tags[i].text
        elif i == 6:
            td_tag_dict["box AP"] = td_tags[i].text
        elif i == 7:
            td_tag_dict["mask AP"] = td_tags[i].text
        elif i == 8:
            td_tag_dict["Config"] = td_tags[i].find('a')['href']
        elif i == 9:
            td_tag_dict["Checkpoint_link"] = td_tags[i].find('a')['href']

    tr_tags_list.append(td_tag_dict)

#
# Save the list of dictionaries as a json file
tr_tags_list = [x for x in tr_tags_list if x["Style"] != "caffe"]

# reid all models
id_count = 5
corrupted_models = []
for i in tqdm(range(len(tr_tags_list))):
    tr_tags_list[i]["id"] = id_count
    id_count += 1
    # replace /open-mmlab/mmdetection/blob/master in config with /mmdetection/configs
    tr_tags_list[i]["Config"] = tr_tags_list[i]["Config"].replace(
        "https://github.com/open-mmlab/mmdetection/tree/master", "mmdetection")
    tr_tags_list[i]["Config"] = tr_tags_list[i]["Config"].replace(
        "https://github.com/open-mmlab/mmdetection/blob/master", "mmdetection")
    tr_tags_list[i]["Checkpoint"] = "mmdetection/checkpoints/" + tr_tags_list[i]["Checkpoint_link"].split(
        "/")[-1]
    tr_tags_list[i]["Checkpoint Size (MB)"] = round(int(requests.head(
        tr_tags_list[i]["Checkpoint_link"]).headers.get('Content-Length', 0)) / (1024 * 1024), 2)

    if tr_tags_list[i]["Checkpoint Size (MB)"] == 0:
        print("Checkpoint size not found for model: ", tr_tags_list[i]["id"])
        corrupted_models.append(i)
        id_count -= 1

    tr_tags_list[i].pop("Style", None)

# remove corrupted models
for i in sorted(corrupted_models, reverse=True):
    tr_tags_list.pop(i)


with open('models_json.json', 'w') as f:
    json.dump(tr_tags_list, f, indent=4)
