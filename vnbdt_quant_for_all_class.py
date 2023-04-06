import os.path
import time

import argparse

from nbdt.utils import DATASET_TO_NUM_CLASSES
from vnbdt import *
from vnbdt_metric import *
import cv2
from nbdt.model import SoftNBDT
import numpy as np
import shutil
import gc
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--cam', default='gradcam', type=str, help='class activate mapping methods')
parser.add_argument('--dataset', default='Imagenet10', type=str, help='dataset name')
parser.add_argument('--arch', default='ResNet50', type=str, help='name of the architecture')
parser.add_argument('--method', default='induced', type=str, help='tree type, others are pro or random')
parser.add_argument('--pth_path', default='/home/lzl001/NBDT/neural-backed-decision-trees/checkpoint/ckpt-Imagenet10-ResNet50-lr0.01-SoftTreeSupLoss_induced.pth', type=str, help='class activate mapping methods')
parser.add_argument('--merge', default='simple', type=str, help='the way to merge the cam')

parser.add_argument('--img_dir', default="/data/LZL/imagenet-10/test", type=str, help='image folder waiting infered and explained')
parser.add_argument('--mask_threshold', default=0.9, type=float, help='')

parser.add_argument('--device', default='0', type=str, help='device')

if __name__ == "__main__":
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

    metric_class = {}
    for CLASS in DATASET_TO_CLASSES[args.dataset]:
        path_list = []
        afoc_list = []
        iou_list = []
        cls_dir = os.path.join(args.img_dir, CLASS)
        if os.path.exists(cls_dir):
            if os.listdir(cls_dir) != 0:
                path_list = [os.path.join(cls_dir, i) for i in os.listdir(cls_dir)][:50]
            else:
                Colors.red("No PIC of cls {} for explain".format(CLASS))

        for img in tqdm(path_list):
            gc.collect()
            torch.cuda.empty_cache()
            mask_dict = {}
            img2 = cv2.imread(img, 1)
            img_name = os.path.split(img)[1].split('.')[0]

            decisions, leaf_to_prob, node_to_prob, predicted, cls, decision_to_wnid, cam_dict = get_nbdt_inference(args.arch, args.dataset, img, net, model, wnids,
                                                                                                                   (448, 448), args.cam, num_cls, img_name)
            flag = 1 if CLASS == cls else 0

            mask_dict = mask_and_inference(decisions[0], img2, cam_dict, model, wnids, args)
            mask_dict['no_mask'] = [node_to_prob, decision_to_wnid]

            change_prob = compute_prob_change(mask_dict)
            change_iou = compute_iou_change(mask_dict)
            afoc_list.append(change_prob)
            iou_list.append(change_iou)
        if len(afoc_list) != 0:
            mean_acof = np.mean(np.array(afoc_list), axis=0).tolist()
            mean_iou = np.mean(np.array(iou_list), axis=0).tolist()
            metric_class[CLASS] = [mean_acof, mean_iou]
            Colors.green('metric calculation of {} is finished'.format(CLASS))
        else:
            Colors.red('no pic, check again!')
    plot_metric_all_class(metric_class, args.img_dir)
