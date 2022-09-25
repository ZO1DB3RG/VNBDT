import os

from vnbdt import *
import cv2
from nbdt.model import SoftNBDT, HardNBDT
from nbdt.models import ResNet18, wrn28_10_cifar10, wrn28_10_cifar100, wrn28_10  # use wrn28_10 for TinyImagenet200
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# 定义获取图的函数

if __name__ == '__main__':

    cam_method = 'gradcam'
    #dataset = 'FGVC12'
    dataset = 'Imagenet10'
    #dataset = 'Fashion10'
    #dataset = 'Emo'
    #dataset = 'FGVC'

    method = 'induced'
    arch = 'ResNet50'
    #arch = 'DFLCNN'
    # pth_path = './pretrain_model/ckpt-DIYdataset-ResNet18-lr0.01-SoftTreeSupLoss.pth'
    #tar_path = 'pretrain_model/model_best_12.pth.tar'
    pth_path = '/home/lzl001/NBDT/neural-backed-decision-trees/checkpoint/ckpt-Imagenet10-ResNet50-lr0.01-SoftTreeSupLoss_induced.pth'
    #pth_path = '/home/mist/checkpoints/ckpt-Imagenet10-vgg16-lr0.01-SoftTreeSupLoss.pth'
    #pkl_path = '/home/lzl001/CNN_train/resnet50_pretrained3.pkl'
    #pth_path = '/home/lzl001/NBDT/neural-backed-decision-trees/checkpoint/ckpt-Imagenet10-vgg19_bn-lr0.01-SoftTreeSupLoss.pth'
    #pth_path = '/home/lzl001/VNBDT/pretrain_model/resnet18_cifar10_9499.pth'

    #tar_path = '/data/LZL/model/bd_resnet50_96.pth.tar'
    #tar_path ='/home/lzl001/FGVC/DFL-CNN/weight_fashion10/model_best.pth.tar'
    #tar_path = '/home/lzl001/FGVC/DFL-CNN/weight_fashion10/epoch_0009_top1_66_checkpoint.pth.tar'

    #tar_path = '/home/lzl001/VNBDT/checkpoints/male_emo25_71.tar'
    #tar_path = '/home/lzl001/VNBDT/checkpoints/female_emo33_75.tar'
    #tar_path = '/home/lzl001/FGVC/DFL-CNN/weight/model_best.pth.tar'

    # 获取该数据集下所有类别的对应ID
    wnids = get_wnids_from_dataset(dataset)
    num_cls = len(wnids)

    # 获取网络
    #net = calling_DNLCNN_from_tar(tar_path, cls_num=10)
    #net = call_tar_model(arch, tar_path, 7)
    #net = call_pkl_model(arch, pkl_path, cls_num=10)
    #net = calling_resnet_modified(pth_path)
    net = call_pth_model(arch, pth_path, cls_num=10)

    #net = wrn28_10_cifar10()
    # model = SoftNBDT(
    #     pretrained=False,
    #     dataset=dataset,
    #     arch=arch,
    #     model=net,
    #     classes=wnids
    # )
    # 生成树并读取
    #G, path = get_pro_tree(dataset=dataset, arch=arch, method=method)
    G, path = get_tree(dataset=dataset, arch=arch, model=net, method=method)
    # 验证树及其节点对应，并返回根节点Node
    root = validate_tree(G, path, wnids)

    """
    此处图像、文件夹的路径应该保证最后HTML页面的图像路径与生成路径一致
    因此需要两个路径，
        path_img1 ----> 原图对于本程序的相对路径
        path_img2 ----> 原图对于HTML的相对路径
        output_dir1 ----> Saliency map要保存的相对路径
        type_name ----> 保存前缀，方便寻找
        
    文件夹设置：
        test_html: 存放HTML和原图、生成图
            image: 存放图像
                origin_image: 存放原图
                out_image: 存放生产图
            xxx1.html & xxx2.html ......
    """

    path_list = os.listdir('false_label')
    path_list = [os.path.join('false_label',i) for i in path_list]

    # path_img1 = '/home/lzl001/VNBDT/emo_img/iaa_pub22492_.jpg'
    # output_dir = 'female_out'
    # html_output = 'female_html'

    # output_dir = 'final_out'
    # html_output = 'final_html'

    #path_list = ['/home/lzl001/VNBDT/fashion10_val/KD.jpg']

    output_dir = 'false_out'
    html_output = 'false_html'
    #
    # output_dir = 'male_out'
    # html_output = 'male_html'
    # output_dir = 'FGVC30_out'
    # html_output = 'FGVC30_html'

    """
    size = None ---> 用原图大小，更多细节
    size = (224, 224) ---> 实验大小，供选择
    """

    # path_list = ['ori_image/0048340.jpg']
    # efccam的热力图要缩小到等高宽
    for img in path_list:
        if method != 'pro' and method != 'random':
            generate_html(G, root, arch, dataset, cam_method, img, net, wnids, num_cls,
                          output_dir, html_output, (448,448),'s', 'simple')
            # generate_html(G, root, arch, dataset, cam_method, img, net, wnids, num_cls,
            #               output_dir, html_output, (448,448),'w', 'w')
            generate_html(G, root, arch, dataset, cam_method, img, net, wnids, num_cls,
                          output_dir, html_output, (448,448),'c', 'complex')
            # generate_html(G, root, arch, dataset, cam_method, img, net, wnids, num_cls,
            #               output_dir, html_output, None, 's1', 'simple')
            # generate_html(G, root, arch, dataset, cam_method, img, net, wnids, num_cls,
            #               output_dir, html_output, None, 'w1', 'w')
            # generate_html(G, root, arch, dataset, cam_method, img, net, wnids, num_cls,
            #               output_dir, html_output, None, 'c1', 'complex')
        else:
            generate_pro_html(G, root,method, path, arch, dataset, cam_method, img, net, wnids, num_cls,
                          output_dir, html_output, (448,448),'s1', 'simple')
