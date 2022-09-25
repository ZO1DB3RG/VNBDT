import os

from vnbdt import *
import cv2
from nbdt.model import SoftNBDT, HardNBDT
from nbdt.models import ResNet18, wrn28_10_cifar10, wrn28_10_cifar100, wrn28_10  # use wrn28_10 for TinyImagenet200
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# 定义获取图的函数

if __name__ == '__main__':

    cam_method = 'gradcam'
    """
    可供选择
       "gradcam": GradCAM,
       "scorecam": ScoreCAM,
       "gradcam++": GradCAMPlusPlus,
       "ablationcam": AblationCAM,
       "xgradcam": XGradCAM,
       "eigencam": EigenCAM,
       "eigengradcam": EigenGradCAM,
       "layercam": LayerCAM,
       "fullgrad": FullGrad,
       'efccam': EFC_CAM
    """

    dataset = 'Imagenet10'
    """
    预先定义、可用的数据集：
        CIFAR10,
        Fashion10(时装数据集10子类),
        Emo(PARA美学数据的情感分类数据集)，
        'FGVC', 'FGVC12','FGVC10': FGVC飞行器数据集的子类
        'Imagenet10': imagenet10子类
    """

    method = 'induced'
    """
    induced: 诱导树
    pro: 专家树 ---> 需要重新在nbdt/hierarcies/dataset 文件夹下重新定义一个固定的专家树结构json文件，详情邮件咨询：li_zhili0105@163.com
    random: 随机树
    """
    arch = 'ResNet50'
    """
    支持：ResNet50, vgg16, ResNet18, wrn28_10_cifar10, DFLCNN
    可以扩展: 于vnbdt.py文件的call_xxx_model()与get_layer()函数中增加对应的模型结构及所需映射的网络层
                于nbdt/graph.py中的MODEL_FC_KEYS增加对应的FC层名称
    """

    pth_path = '/home/lzl001/NBDT/neural-backed-decision-trees/checkpoint/ckpt-Imagenet10-ResNet50-lr0.01-SoftTreeSupLoss_induced.pth'
    #tar_path = '/home/lzl001/VNBDT/checkpoints/male_emo25_71.tar'
    #tar_path = '/home/lzl001/VNBDT/checkpoints/female_emo33_75.tar'
    #tar_path = '/home/lzl001/FGVC/DFL-CNN/weight/model_best.pth.tar'

    """
    支持pth, pkl, tar三种形式的模型文件
    对应加载函数：arch为模型框架（如ResNet50）; cls_num为模型分类的类别数量，对应叶节点数量
    pth: call_pth_model(arch, pth_path, cls_num = 10)
    pkl: call_pkl_model(arch, pkl_path, cls_num=10)
    tar: call_tar_model(arch, tar_path, cls_num)
    一个细粒度模型DFLCNN：calling_DNLCNN_from_tar(tar_path, cls_num=10)
    """

    merge_method = 'simple'
    """
    内节点热力图融合方法: 
        simple: Decision-probability-oriented fusion of multi-CAM;
        complex: Similarity-attention-oriented fusion of multi-CAM;
        None: without any weight
    """

    # 获取该数据集下所有类别的对应ID
    wnids = get_wnids_from_dataset(dataset)
    num_cls = len(wnids)

    # 获取网络
    net = call_pth_model(arch, pth_path, cls_num=10)

    #输入网络权重获取NBDT树结构，延续原本的代码操作
    G, path = get_tree(dataset=dataset, arch=arch, model=net, method=method)
    #也可以生成固定结构的专家树或者随机树，但要保证nbdt/hierarcies/dataset 文件夹下已存在对应的结构json文件
    #G, path = get_pro_tree(dataset=dataset, arch=arch, method=method)

    # 验证树及其节点对应，并返回根节点Node
    # 该过程打印的信息若不需要可以删除
    root = validate_tree(G, path, wnids)

    """
    由于visualization依托于HTML页面，因此在生成树结构和热力图以后对其保存路径要求比较严格   
    文件夹设置：
        1.samples/img_xxx: 保存需要解释的图像集（文件夹）img_xxx在sample中，
                           此处代码假设img_xxx下没有子文件夹，全是图像
                           如果测试图像不在项目文件夹下，就会出现html页面根节点不显示原图
            img_1.jpg/png ---> 为了更好地展示结果，这里的图片名称最好与原本的类别对应，比如 mastiff1.jpg
            img_2.jpg/png
            ......
        
        2.xx_output: 运行代码后保存CAM解释生成的图像，每一个图像都会生成一个文件夹，文件夹下保存链路所有节点的图像
                     且这个生成的文件夹名称直接体现CAM方法和融合方法
            img_1_[CAM_METHODS]_[MERGE_METHODS]：
                img_1_[CAM_METHODS]_[MERGE_METHODS]_1.jpg
                img_1_[CAM_METHODS]_[MERGE_METHODS]_2.jpg
                ......
            img_2_[CAM_METHODS]_[MERGE_METHODS]：
                img_2_[CAM_METHODS]_[MERGE_METHODS]_1.jpg
                img_2_[CAM_METHODS]_[MERGE_METHODS]_2.jpg
                ......
    
        3.xx_html: 运行代码以后生成html文件，打开即可监视解释效果并互动
            xxx1.html
            xxx2.html
            ......
    """

    img_dir = 'samples/imagenet10'
    output_dir = 'img10_out'
    html_output = 'img10_html'

    #为html页面及其存储图片创建目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(html_output):
        os.makedirs(html_output)

    path_list = []

    if os.path.exists(img_dir):
        if os.listdir(img_dir) != 0:
            path_list = [os.path.join(img_dir, i) for i in os.listdir(img_dir)]
        else:
            print("No PIC for explain")

    """
    efccam的热力图要缩小到等高宽
    size = None ---> 用原图大小，更多细节
    size = (224, 224) ---> 实验大小，供选择
    """
    for img in path_list:
        if method != 'pro' and method != 'random':
            generate_html(G, root, arch, dataset, cam_method, img, net, wnids, num_cls,
                          output_dir, html_output, (448,448), 's', merge_method)
            # generate_html(G, root, arch, dataset, cam_method, img, net, wnids, num_cls,
            #               output_dir, html_output, (448,448),'w', 'w')
            # generate_html(G, root, arch, dataset, cam_method, img, net, wnids, num_cls,
            #               output_dir, html_output, (448,448),'c', 'complex')
        else:
            generate_pro_html(G, root,method, path, arch, dataset, cam_method, img, net, wnids, num_cls,
                          output_dir, html_output, (448,448),'s1', merge_method)
