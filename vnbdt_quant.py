import os.path
import time

import argparse

import tqdm

from nbdt.utils import DATASET_TO_NUM_CLASSES
from vnbdt import *
from vnbdt_metric import *
import cv2
from nbdt.model import SoftNBDT
import numpy as np
import shutil
import gc

parser = argparse.ArgumentParser()
parser.add_argument('--cam', default='gradcam', type=str, help='class activate mapping methods')
parser.add_argument('--dataset', default='Imagenet10', type=str, help='dataset name')
parser.add_argument('--arch', default='ResNet50', type=str, help='name of the architecture')
parser.add_argument('--method', default='induced', type=str, help='tree type, others are pro or random')
parser.add_argument('--pth_path', default='/home/lzl001/NBDT/neural-backed-decision-trees/checkpoint/ckpt-Imagenet10-ResNet50-lr0.01-SoftTreeSupLoss_induced.pth', type=str, help='class activate mapping methods')
parser.add_argument('--merge', default='simple', type=str, help='the way to merge the cam')

parser.add_argument('--img_dir', default="/data/LZL/imagenet-10/test/Ibizan_hound", type=str, help='image folder waiting infered and explained')
parser.add_argument('--output_dir', default="/data/LZL/imagenet-10/test/Ibizan_hound_out", type=str, help='Store CAM')
parser.add_argument('--html_output', default="/data/LZL/imagenet-10/test/Ibizan_hound_html", type=str, help='Store html, should be the same father with output_dir')

parser.add_argument('--mask_threshold', default=0.9, type=float, help='')

parser.add_argument('--name', default='', type=str, help='something you want your file name to be')
parser.add_argument('--device', default='0', type=str, help='device')

if __name__ == "__main__":
    # base_path = '/home/lzl001/CNN_train/resnet50_pretrained3.pkl'
    # nbdt_ft_path = '/home/lzl001/NBDT/neural-backed-decision-trees/checkpoint/ckpt-Imagenet10-ResNet50-lr0.01-SoftTreeSupLoss_induced.pth'
    # pro_pth = '/home/lzl001/NBDT/neural-backed-decision-trees/checkpoint/ckpt-Imagenet10-ResNet50-lr0.01-SoftTreeSupLoss_pro.pth'
    # random_pth = '/home/lzl001/NBDT/neural-backed-decision-trees/checkpoint/ckpt-Imagenet10-ResNet50-lr0.01-SoftTreeSupLoss_random.pth'
    # # arch = 'vgg19_bn'
    # # base_path = '/home/lzl001/CNN_train/model_vgg19bn_pretrained.pkl'
    # # nbdt_ft_path = '/home/lzl001/NBDT/neural-backed-decision-trees/checkpoint/ckpt-Imagenet10-vgg19_bn-lr0.01-SoftTreeSupLoss.pth'
    # # tar_path = './pretrain_model/model_best_12.pth.tar'
    # # exp_img_source_path = '/home/mist/imagenet-10/val'

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # 获取该数据集下所有类别的对应ID
    wnids = get_wnids_from_dataset(args.dataset)
    num_cls = len(wnids)

    # 获取网络
    if os.path.splitext(args.pth_path)[-1].find('pth') == 1:
        net = call_pth_model(args.arch, args.pth_path, cls_num=DATASET_TO_NUM_CLASSES[args.dataset])
    else:
        net = call_pkl_model(args.arch, args.pth_path, cls_num=DATASET_TO_NUM_CLASSES[args.dataset])

    # 输入网络权重获取NBDT树结构，延续原本的代码操作
    if args.method == 'induced':
        G, path = get_tree(dataset=args.dataset, arch=args.arch, model=net, method=args.method)
    # 也可以生成固定结构的专家树或者随机树，但要保证nbdt/hierarcies/dataset 文件夹下已存在对应的结构json文件
    else:
        G, path = get_pro_tree(dataset=args.dataset, arch=args.arch, method=args.method)

    # 验证树及其节点对应，并返回根节点Node
    # 该过程打印的信息若不需要可以删除
    root = validate_tree(G, path, wnids)
    model = SoftNBDT(
        pretrained=False,
        dataset=args.dataset,
        arch=args.arch,
        model=net,
        classes=wnids
    ).eval()
    # 为html页面及其存储图片创建目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.html_output):
        os.makedirs(args.html_output)

    path_list = []

    if os.path.isdir(args.img_dir):
        if os.path.exists(args.img_dir):
            if os.listdir(args.img_dir) != 0:
                path_list = [os.path.join(args.img_dir, i) for i in os.listdir(args.img_dir)]
            else:
                print("No PIC for explain")

        ori_cls = os.path.split(args.img_dir)[-1]

        mask_dict = {}
        pred = None

        for img in tqdm.tqdm(path_list):
            img2 = cv2.imread(img, 1)

            path_img2 = os.path.join('..', os.path.join(img.split('/')[-2], img.split('/')[-1]))  # 该路径用于插入root图像
            type_name = os.path.split(img)[1].split('.')[0].split('_')[-1] \
                        + '_' + args.cam + '-' + args.merge
            img_name = os.path.split(img)[1].split('.')[0]

            decisions, leaf_to_prob, node_to_prob, predicted, cls, decision_to_wnid, cam_dict = get_nbdt_inference(args.arch, args.dataset, img, net, model, wnids,
                                                                                                                    (448,448),
                                                                                                                    args.cam,num_cls,img_name)

            if ori_cls == cls:
                file_name = 'T-' + type_name
                pred = 1
            else:
                file_name = 'F-' + type_name
                pred = 0

            if not os.path.exists(os.path.join(args.output_dir, file_name)):
                os.makedirs(os.path.join(args.output_dir, file_name))
            output_dir = os.path.join(args.output_dir, file_name)

            if args.merge == 'complex':
                complex_w = compute_complex_weight(cam_dict, predicted)
                mask_dict = mask_and_inference(decisions[0], img2, cam_dict, model, wnids, args, complex_w)
            else:
                mask_dict = mask_and_inference(decisions[0], img2, cam_dict, model, wnids, args)

            mask_dict['no_mask'] = [node_to_prob, decision_to_wnid]

            change_prob = compute_prob_change(mask_dict)
            change_iou = compute_iou_change(mask_dict)
            plot_metric(change_prob, change_iou, output_dir)


            if args.method != 'pro' and args.method != 'random':
                generate_html(G, root, args.arch, args.dataset, args.cam, img, net, wnids, num_cls,
                              args.output_dir, args.html_output, (448, 448), args.name, args.merge, ori_cls)

            else:
                generate_pro_html(G, root, args.method, path, args.arch, args.dataset, args.cam, img, net, wnids, num_cls,
                                  args.output_dir, args.html_output, (448, 448), args.name, args.merge, ori_cls)

    else:
        ori_cls = os.path.split(args.img_dir)[-2]
        mask_dict = {}
        pred = None
        img = args.img_dir
        img2 = cv2.imread(img, 1)

        path_img2 = os.path.join('..', os.path.join(img.split('/')[-2], img.split('/')[-1]))  # 该路径用于插入root图像
        type_name = os.path.split(img)[1].split('.')[0].split('_')[-1] \
                    + '_' + args.cam + '-' + args.merge
        img_name = os.path.split(img)[1].split('.')[0]

        decisions, leaf_to_prob, node_to_prob, predicted, cls, decision_to_wnid, cam_dict = get_nbdt_inference(
            args.arch, args.dataset, img, net, model, wnids,
            (448, 448),
            args.cam, num_cls, img_name)

        if ori_cls == cls:
            file_name = 'T-' + type_name
            pred = 1
        else:
            file_name = 'F-' + type_name
            pred = 0

        if not os.path.exists(os.path.join(args.output_dir, file_name)):
            os.makedirs(os.path.join(args.output_dir, file_name))
        output_dir = os.path.join(args.output_dir, file_name)

        if args.merge == 'complex':
            complex_w = compute_complex_weight(cam_dict, predicted)
            mask_dict = mask_and_inference(decisions[0], img2, cam_dict, model, wnids, args, complex_w)
        else:
            mask_dict = mask_and_inference(decisions[0], img2, cam_dict, model, wnids, args)

        mask_dict['no_mask'] = [node_to_prob, decision_to_wnid]

        change_prob = compute_prob_change(mask_dict)
        change_iou = compute_iou_change(mask_dict)
        plot_metric(change_prob, change_iou, output_dir)

        if args.method != 'pro' and args.method != 'random':
            generate_html(G, root, args.arch, args.dataset, args.cam, img, net, wnids, num_cls,
                          args.output_dir, args.html_output, (448, 448), args.name, args.merge, ori_cls)

        else:
            generate_pro_html(G, root, args.method, path, args.arch, args.dataset, args.cam, img, net, wnids,
                              num_cls,
                              args.output_dir, args.html_output, (448, 448), args.name, args.merge, ori_cls)
