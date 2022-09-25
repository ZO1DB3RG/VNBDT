from vnbdt import *
import cv2
from nbdt.model import SoftNBDT, HardNBDT

# 定义获取图的函数

if __name__ == '__main__':
    cam_method = 'gradcam'
    dataset = 'Imagenet10'
    method = 'induced'
    arch = 'ResNet50'
    # pth_path = './pretrain_model/ckpt-DIYdataset-ResNet18-lr0.01-SoftTreeSupLoss.pth'
    #tar_path = './pretrain_model/model_best_12.pth.tar'
    pth_path = '/home/lzl001/NBDT/neural-backed-decision-trees/checkpoint/ckpt-Imagenet10-ResNet50-lr0.01-SoftTreeSupLoss_induced.pth'

    # 获取该数据集下所有类别的对应ID
    wnids = get_wnids_from_dataset(dataset)
    num_cls = len(wnids)

    # 获取网络
    #net = calling_DNLCNN_from_tar(tar_path, cls_num=12)
    net = call_pth_model(arch, pth_path, cls_num=10)

    # 生成树并读取
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

    path_img1 = '/data/LZL/imagenet-10/val/Tibetan_mastiff/n0210855100000057.jpg'
    output_dir = 'masked_img'
    path_img2 = os.path.join('..', path_img1)
    type_name = os.path.split(path_img1)[1].split('.')[0] + '_' + dataset + '_' + arch + '_' + cam_method + '-0.6'
    img_name = os.path.split(path_img1)[1].split('.')[0]

    # 图像读取和预处理，读取的图像img1用来合成CAM，im用来获取x
    # img1 = cv2.imread(path_img1, 1)[:, :, ::-1]
    # img1 = np.float32(img1) / 255
    # im = load_image_from_path(path_img1)
    # x = preprocess_img(im, [0.4948052, 0.48568845, 0.44682974],
    #                        [0.24580306, 0.24236229, 0.2603115])

    img1 = Image.open(path_img1)
    img2 = cv2.imread(path_img1, 1)
    img3 = np.float32(img2) / 255
    transform1 = get_transform()
    img_tensor = transform1(img1)
    x = img_tensor.unsqueeze(0)

    # 获取树模型，并装载预训练权重，最后前向推导树，获得决策链路
    # pretrained保持false，否则会重新调取随机的权重，导致CAM不一致

    model = SoftNBDT(
        pretrained=False,
        dataset=dataset,
        arch=arch,
        model=net,
        classes=wnids
    )
    # 模块化后不需要.eval()

    decisions, leaf_to_prob, node_to_prob,predicted = forword_tree(x, model, wnids, dataset)

    # 获取一个包含所有叶节点对应cam图的字典，此处还未resize，再进行融合

    target_layers = None
    if arch == "ResNet50":
        target_layers = [model.model.layer4[-1]]
    elif arch == 'DFLCNN':
        target_layers = [model.model.conv5]
    elif arch == 'vgg16':
        target_layers = [model.model.features[-1]]
    elif arch == 'ResNet18':
        target_layers = [model.model.layer4[-1]]
    elif arch == 'wrn28_10_cifar10':
        target_layers = [model.model.features[-3][-1].body.conv2.conv]
    # 模型哪一个layer
    # cam_dict = get_all_leaf_cam_from_method(x, model.model, leaf_to_prob,
    #                                         num_cls, cam_method, target_layers,
    #                                         aug_smooth=False,
    #                                         eigen_smooth=False)

    if cam_method == 'efccam':
        cam_dict = get_all_leaf_cam_efc(x.cuda(), net.cuda(), leaf_to_prob, num_cls,
                                        cam_method, target_layers, path_img1, img_name)
    else:
        cam_dict = get_all_leaf_cam_from_method(x.cuda(), model.model.cuda(), leaf_to_prob,
                                                num_cls, cam_method, target_layers,
                                                aug_smooth=True,
                                                eigen_smooth=True)
    generate_cam_mask(decisions[0], img2, img_name, cam_dict, output_dir)

