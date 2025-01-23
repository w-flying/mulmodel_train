import json
import os

from zhipuai import ZhipuAI
import base64
import  time


def text_detection(img_path):
    with open(img_path, 'rb') as img_file:
        img_base = base64.b64encode(img_file.read()).decode('utf-8')
        client = ZhipuAI()  # 填写您自己的APIKey
        response = client.chat.completions.create(
            model="glm-4v-plus-0111",  # 填写需要调用的模型名称
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": img_base
                            }
                        },
                        {
                            "type": "text",
                            "text": '''请严格执行目标检测任务，图像中是否存在文字，若有则返回1，否则返回0，不要返回其他内容
          '''
                        }
                    ]
                }
            ]
        )
        return response.choices[0].message.content


img_path = "test.jpg"
img_files = os.listdir("images")
projects = []
for img_file in img_files:
    img_path = os.path.join("images", img_file)
    time.sleep(0.3)
    try:
        projects.append({img_file: text_detection(img_path)})
        if len(projects) % 100 == 0:
            print(len(projects))
    except:
        projects.append({img_file: 'error'})
with open('projects_glm.json', 'w', encoding='utf-8') as f:
    # 调用一次 json.dump 方法将 projects 写入文件
    json.dump(projects, f)


