import os
import torch
from PIL import Image
from MyLeNet import MyNet
from torchvision import transforms

datalist = "D:\\PythonProject\\LeNet\\images"

# 对输入图片的处理
trans = transforms.Compose([
    # transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

# 如果有显卡，可以转到GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 调用模型
model = MyNet().to(device)
model.load_state_dict(torch.load("D:\\PythonProject\\LeNet\\checkpoints\\best_model.pt"))

# 获取结果
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

if __name__ == '__main__':

    # 图像的读写和预处理
    img_list = os.listdir(datalist)
    for img_name in img_list:
        img_path = os.path.join(datalist, img_name)
        img = Image.open(img_path).convert('L')
        img = img.resize((28, 28), Image.ANTIALIAS)
        threshold = 50
        for i in range(28):
            for j in range(28):
                val = img.getpixel((i, j))
                img.putpixel((i, j), 255 - val)
                if img.getpixel((i, j)) < threshold:
                    img.putpixel((i, j), 0)
                else:
                    img.putpixel((i, j), 255)

        # img.show()
        img = trans(img)
        print(f'name: {img_path}')
        # print(img.size())
        img = torch.unsqueeze(img, dim=0)
        # print(img.size())
        img = img.to(device)
        with torch.no_grad():
            output = torch.squeeze(model(img), dim=0)
            # print(output)
            predict = classes[torch.argmax(output)]
            print(f'Predicted: {predict}')
            print("\n")
