import torch
from PIL import Image
import cn_clip.clip as clip
from cn_clip.clip import load_from_name
import os
import pandas as pd
def calculate_image_text_similarities(image_paths, texts, model_name="ViT-B-16", download_root='./'):
    # 选择设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载预训练模型和预处理函数
    model, preprocess = load_from_name(model_name, device=device, download_root=download_root)
    model.eval()

    # 对图片进行预处理并提取特征
    image_features = []
    for path in image_paths:
        try:
            image = preprocess(Image.open(path)).unsqueeze(0).to(device)
            with torch.no_grad():
                feature = model.encode_image(image)
                feature /= feature.norm(dim=-1, keepdim=True)
                image_features.append(feature)
        except FileNotFoundError:
            print(f"未找到图片文件: {path}")
            image_features.append(None)

    # 对文本进行编码并提取特征
    text_features = []
    for text in texts:
        tokenized_text = clip.tokenize([text]).to(device)
        with torch.no_grad():
            feature = model.encode_text(tokenized_text)
            feature /= feature.norm(dim=-1, keepdim=True)
            text_features.append(feature)

    # 计算每个图片和对应主题的相似性
    similarities = []
    for img_feat, txt_feat in zip(image_features, text_features):
        if img_feat is not None:
            similarity = (img_feat @ txt_feat.T).item()
            similarities.append(similarity)
        else:
            similarities.append(None)
    results = pd.DataFrame({'image_path': image_paths, 'text': texts, 'similarity': similarities})


    return similarities, results
imgname=pd.read_excel('../weibo.xlsx')['img_name'].tolist()

#读取images文件夹中的所有图片
image_paths = [os.path.join('../images', f) for f in imgname]

# 对应主题文本列表
texts = pd.read_excel('../weibo.xlsx')['title'].tolist()

similarities, results = calculate_image_text_similarities(image_paths, texts)
results.to_excel('results.xlsx', index=False)

