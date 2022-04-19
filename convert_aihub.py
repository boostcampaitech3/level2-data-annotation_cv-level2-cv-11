import os
import json
from collections import OrderedDict

file_data = dict(images=dict())
# file_data["images"] = dict()

#저장할 사진의 제목 저장
#저장할 사진을 하나의 파일에 모아야함.
#저장할 json들도 하나의 파일에 모아야함.
#--make_ufo.py
#--Outdata/ 사진파일
#--json/ json파일 
path = './images'
file_list = os.listdir(path)
cnt = -1
#저장할 사진의 json파일 열기
for name in file_list:
    #file_data["root"]["images"][name] = OrderedDict()
    temp = dict()
    j_name = name.replace('.JPG','.json')
    j_name = j_name.replace('.jpg','.json')
    j_name = j_name.replace('.jpeg','.json')
    j_name = j_name.replace('.JPEG','.json')
    cnt += 1
    address = './json/' + j_name
    with open(address,'r', encoding='utf-8') as f:
        json_data = json.load(f)
        temp["img_h"] = json_data["images"][0]["height"]
        temp["img_w"] = json_data["images"][0]["width"]
        temp["words"] = dict()
        
        i = 0
        for ann in json_data["annotations"]:
            temp["words"][i] = dict()
            if(len(ann["bbox"])!=4):
                continue
            
            x,y,w,h = ann["bbox"]
            if ann["bbox"][0] is None:
                continue 
                
            temp["words"][i] = {
                "transcription":ann["text"],
                "language": ["ko"]}
            
            if ann["text"] == "xxx":
                temp["words"][i]["illegibility"]=True
            else:
                temp["words"][i]["illegibility"]=False
           

            if json_data["metadata"][0]["wordorientation"] == "가로":
                temp["words"][i]["orientation"] = "Horizontal"
            elif json_data["metadata"][0]["wordorientation"] == "세로":
                temp["words"][i]["orientation"] = "Vertical"
            else:
                temp["words"][i]["orientation"] = "Irregular"

            point = [[x,y],
                     [x+w,y],
                     [x+w,y+h],
                     [x,y+h]]
            temp["words"][i]["points"] = point
            temp["words"][i]["word_tags"]=None
            i+=1
         
    file_data["images"][name] = temp
print(cnt)

#print(json.dumps(file_data,ensure_ascii=False,indent='\t'))
file_path = 'ufo/train.json'

with open(file_path,'w', encoding='utf-8-sig') as f:
    f.write(json.dumps(file_data, indent=4))
    # json.dump(file_data, f, indent=4)