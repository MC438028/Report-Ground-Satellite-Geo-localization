import argparse
import numpy as np
import os
from torchvision import transforms
from image_folder import CustomData160k_sat, CustomData160k_drone
import matplotlib.pyplot as plt

#######################################################################
# 解析命令行参数
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--query_index', default=0, type=int, help='test_image_index')
parser.add_argument('--test_dir', default='./data/test', type=str, help='./test_data')
parser.add_argument('--query_name', default='query_street_name.txt', type=str, help='load query image')
opts = parser.parse_args()

# 定义数据变换
data_transforms = transforms.Compose([
    transforms.RandomResizedCrop((256, 256), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载数据集
gallery_name = 'gallery_satellite'
query_name = 'query_street'
data_dir = opts.test_dir
image_datasets = {
    'gallery_satellite': CustomData160k_sat(os.path.join(data_dir, 'workshop_gallery_satellite'), data_transforms),
    'query_street': CustomData160k_drone(os.path.join(data_dir, 'workshop_query_street'), data_transforms, query_name=opts.query_name)
}

#####################################################################
# 显示图像
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.1)  # 暂停一会儿，以便更新绘图

######################################################################
# 读取结果文件
results_rank10 = np.genfromtxt('answer.txt', delimiter='\t', dtype=str)

# 获取查询图像和画廊图像的路径
query_path, _ = image_datasets[query_name].imgs[opts.query_index]
gallery_paths = [path for path, _ in image_datasets[gallery_name].imgs]

# 获取查询图像对应的排名前10的画廊图像索引
rank10_indices = []
for label in results_rank10[opts.query_index]:
    for i, (_, img_label) in enumerate(image_datasets[gallery_name].imgs):
        if img_label == label:
            rank10_indices.append(i)
            break

########################################################################
# 可视化排名结果
print(query_path)
print('Top 10 images are as follow:')
# 修改保存文件夹名称
save_folder = 'image_show_160k/%02d' % opts.query_index
if not os.path.isdir(save_folder):
    os.makedirs(save_folder, exist_ok=True)
os.system('cp %s %s/query.jpg' % (query_path, save_folder))

try:  # 可视化排名结果
    # 需要图形用户界面
    fig = plt.figure(figsize=(16, 4))
    ax = plt.subplot(1, 11, 1)
    ax.axis('off')
    imshow(query_path, 'query')
    for i in range(10):
        ax = plt.subplot(1, 11, i + 2)
        ax.axis('off')
        img_path = gallery_paths[rank10_indices[i]]
        label = results_rank10[opts.query_index][i]
        print(label)
        imshow(img_path)
        os.system('cp %s %s/s%02d.jpg' % (img_path, save_folder, i))
        # 假设查询图像的标签可以从路径中提取，这里简单处理
        query_label = os.path.splitext(os.path.basename(query_path))[0]
        if label == query_label:
            ax.set_title('%d' % (i + 1), color='green')
        else:
            ax.set_title('%d' % (i + 1), color='red')
        print(img_path)
    # plt.pause(100)  # 暂停一会儿，以便更新绘图
except RuntimeError:
    for i in range(10):
        img_path = gallery_paths[rank10_indices[i]]
        print(img_path)
    print('If you want to see the visualization of the ranking result, graphical user interface is needed.')

fig.savefig("show.png")