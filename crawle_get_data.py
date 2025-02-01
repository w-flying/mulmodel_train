import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
import json
from selenium.webdriver.common.action_chains import ActionChains
from streamlit import video
from selenium.webdriver.chrome.options import Options
import pandas as pd
def load_image(url):
    response = requests.get(url)
    img_name= url.split('/')[-1]
    if response.status_code == 200:
        # 将图片保存到本地文件
        with open('images/'+img_name, 'wb') as file:
            file.write(response.content)
        return img_name,1
    else:
        return img_name,0
def crawle(url,cookies):
    driver = webdriver.Chrome()
    driver.get('https://weibo.com')
    time.sleep(40)
    for cookie in cookies:
        driver.add_cookie(cookie)
    driver.get(url)
    # 等待页面加载完成
    time.sleep(5)
    prev_count = 0
    while True:
        # 查找元素，这里以查找所有的 <div> 元素为例，你可以根据实际情况修改元素的定位方式
        elements = driver.find_elements(By.CLASS_NAME, 'Waterfall-item')
        curr_count = len(elements)
        if curr_count == prev_count:
            break
        prev_count = curr_count
        # 滚动到页面底部
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        # 等待一段时间以便页面加载新元素
        time.sleep(5)
    video_elements = driver.find_elements(By.CLASS_NAME, 'Waterfall-item')
    videos_info=[]
    for video_element in video_elements:
        img_url= video_element.find_element(By.CLASS_NAME, 'woo-picture-img').get_attribute('src')
        witchnum_timelength= video_element.find_element(By.CLASS_NAME, 'woo-box-flex').find_elements(By.TAG_NAME, 'span')
        witchnum= witchnum_timelength[0].text
        timelength= witchnum_timelength[1].text
        title= video_element.find_element(By.CLASS_NAME, 'Card1_ccon_1gZ2p').find_element(By.TAG_NAME, 'div').text
        likes=video_element.find_element(By.CLASS_NAME,'Card1_cct3_3gtSF').find_elements(By.TAG_NAME, 'span')[-1].text
        img_name,img_load =load_image(img_url)
        print(title+'已处理')
        videos_info.append([title,witchnum,timelength,likes,img_name,img_url,img_load])
    return videos_info
# 打开并读取JSON文件
with open('cookie.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
print(data)
cookies=[]
for i in data:
    cookie={}
    cookie['name']=i['name']
    cookie['value']=i['value']
    cookie['domain']=i['domain']
    cookies.append(cookie)
urls=['https://weibo.com/u/3905458630?tabtype=newVideo',
      'https://weibo.com/u/2809775704?tabtype=newVideo',
      'https://weibo.com/1704709632?tabtype=newVideo',
      'https://weibo.com/u/6349791448?tabtype=newVideo',
      'https://weibo.com/u/3921865344?tabtype=newVideo'
]
result=[]
for url in urls:
    result+=crawle(url,cookies)
pd.DataFrame(result,columns=['title','witchnum','timelength','likes','img_name','img_url','img_load']).to_excel('weibo1.31.xlsx',index=False)