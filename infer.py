import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
torch.__version__
from train import ConvNet
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

MODEL_PATH="./data/MNIST/models/infer.pth"
PIC_PATH="/data/9-3.jpg"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多

def preprocess_image(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    
    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 二值化图像（阈值处理）
    _, binary = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)
    # Image.fromarray(binary).save('binary_image.jpg')

    # 创建一个结构元素（核）
    kernel = np.ones((5, 5), np.uint8)

    # 使用OpenCV进行膨胀操作
    dilated_image = cv2.dilate(binary, kernel, iterations=13)
    # Image.fromarray(dilated_image).save('binary_dilated_image.jpg')
    
    # 查找轮廓
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 找到包含手写字的最小矩形
    x, y, w, h = cv2.boundingRect(contours[0])
    s = w * h
    for cnt in contours:
        x_, y_, w_, h_ = cv2.boundingRect(cnt)
        # x = min(x, x_)
        # y = min(y, y_)
        # w = max(w, x_ + w_ - x)
        # h = max(h, y_ + h_ - y)
        s_ = w_ * h_
        if s_ > s :
            s = s_
            x = x_
            y = y_
            w = w_
            h = h_
    
    # 裁剪图像
    cropped = dilated_image[y:y+h, x:x+w]
    # Image.fromarray(cropped).save('binary_cropped_image.jpg')
    
    const_height = 20
    # 调整图像尺寸为22xdw
    dw = round(float(const_height) / h * w)
    # print("dw = ", dw)
    resized = cv2.resize(cropped, (dw, const_height), interpolation=cv2.INTER_AREA)
    
    # 创建28x28的空白图像
    canvas = np.zeros((28, 28), dtype=np.uint8)
    
    # 将调整大小后的图像居中放置到28x28图像中
    start_x = (28 - dw) // 2
    start_y = (28 - const_height) // 2 + 1
    canvas[start_y:start_y + const_height, start_x:start_x + dw] = resized
    # 保存二值化后的图像
    # Image.fromarray(canvas).save('binary_resized_image.jpg')

    # 归一化
    normalized = canvas / 255.0
    
    return normalized

# 加载模型
model = ConvNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# 加载并预处理图像
image_np = preprocess_image(PIC_PATH)

transform = transforms.Compose([
    # transforms.Resize((28, 28)),
    transforms.ToTensor(),               # 转换为张量
    transforms.Normalize((0.5,), (0.5,)) # 归一化
])

# image = transform(image).unsqueeze(0)  # 增加批次维度

image = Image.fromarray(image_np)

# 预处理图像并添加批次维度
processed_image = transform(image).unsqueeze(0).to(DEVICE)

# # 可视化预处理后的图像
# # 将张量转换为可以显示的格式
# image_to_show = processed_image.cpu().squeeze().numpy()
# image_to_show = image_to_show * 0.5 + 0.5  # 反归一化
# image_to_show = (image_to_show * 255).astype(np.uint8)  # 转换为8位整数

# # 将 NumPy 数组转换为 PIL 图像
# processed_pil_image = Image.fromarray(image_to_show)

# # 保存处理后的图像
# processed_pil_image.save("processed_image.jpg")

# 使用模型进行推理
with torch.no_grad():
    output = model(processed_image.to(DEVICE))
    probabilities = nn.functional.softmax(output, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

# 输出推理结果及其可信度
predicted_digit = predicted.item()
confidence_score = confidence.item()
print(f'Predicted Digit: {predicted_digit} with confidence: {confidence_score:.4f}')