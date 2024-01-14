import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn

from model import resnet34


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((256, 256)),
         # transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])                # 要和训练标准化参数一样

    # load image
    img_path = r'D:\DEEPLEARNING\Deeplearning\Bijie_landslide_dataset\devide_data\df035.png'
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    # model = resnet34()
    # in_channel = model.fc.in_features
    # model.fc = nn.Linear(in_channel, 2)             # 对应model中的fc全连接层，5为花分类的类别数
    # model.to(device)                                # 若不用迁移学习则注销上述几行代码
    model = resnet34(num_classes=2).to(device)
    # load model weights
    weights_path = r"D:\DEEPLEARNING\Deeplearning\Bijie_landslide_dataset/resNet34.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    with torch.no_grad():                   # 不对损失梯度进行跟踪
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()          # 将图片输入到模型当中 压缩batch维度
        predict = torch.softmax(output, dim=0)               # softmax处理得到概率分布
        predict_cla = torch.argmax(predict).numpy()            # argmax寻找最大值对应的索引

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())              # 打印类别信息 概率
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
