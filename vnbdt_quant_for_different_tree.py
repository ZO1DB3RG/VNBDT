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
parser.add_argument('--induced_pth_path', default='/home/lzl001/NBDT/neural-backed-decision-trees/checkpoint/ckpt-Imagenet10-ResNet50-lr0.01-SoftTreeSupLoss_induced.pth', type=str, help='class activate mapping methods')
parser.add_argument('--pro_pth_path', default='/home/lzl001/NBDT/neural-backed-decision-trees/checkpoint/ckpt-Imagenet10-ResNet50-lr0.01-SoftTreeSupLoss_induced.pth', type=str, help='class activate mapping methods')
parser.add_argument('--random_pth_path', default='/home/lzl001/NBDT/neural-backed-decision-trees/checkpoint/ckpt-Imagenet10-ResNet50-lr0.01-SoftTreeSupLoss_induced.pth', type=str, help='class activate mapping methods')
parser.add_argument('--merge', default='simple', type=str, help='the way to merge the cam')

parser.add_argument('--img_dir', default="/data/LZL/imagenet-10/test", type=str, help='image folder waiting infered and explained')
parser.add_argument('--plot_name', default="metric_3_tree_same", type=str, help='name of the final plot')

parser.add_argument('--mask_threshold', default=0.9, type=float, help='')

parser.add_argument('--device', default='3', type=str, help='device')

if __name__ == "__main__":
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # 获取该数据集下所有类别的对应ID
    wnids = get_wnids_from_dataset(args.dataset)
    num_cls = len(wnids)

    # 获取网络
    if os.path.splitext(args.induced_pth_path)[-1].find('pth') == 1:
        induced_net = call_pth_model(args.arch, args.induced_pth_path, cls_num=DATASET_TO_NUM_CLASSES[args.dataset])
    else:
        induced_net = call_pkl_model(args.arch, args.induced_pth_path, cls_num=DATASET_TO_NUM_CLASSES[args.dataset])

    if os.path.splitext(args.pro_pth_path)[-1].find('pth') == 1:
        pro_net = call_pth_model(args.arch, args.pro_pth_path, cls_num=DATASET_TO_NUM_CLASSES[args.dataset])
    else:
        pro_net = call_pkl_model(args.arch, args.pro_pth_path, cls_num=DATASET_TO_NUM_CLASSES[args.dataset])

    if os.path.splitext(args.random_pth_path)[-1].find('pth') == 1:
        random_net = call_pth_model(args.arch, args.random_pth_path, cls_num=DATASET_TO_NUM_CLASSES[args.dataset])
    else:
        random_net = call_pkl_model(args.arch, args.random_pth_path, cls_num=DATASET_TO_NUM_CLASSES[args.dataset])

    net_dict = {'induced': induced_net, 'pro':pro_net, 'random':random_net}
    model_dict = {}
    for method, net in net_dict.items():
        if method == 'induced':
            G, path = get_tree(dataset=args.dataset, arch=args.arch, model=net, method='induced')
            root = validate_tree(G, path, wnids)
            model = SoftNBDT(
                pretrained=False,
                dataset=args.dataset,
                arch=args.arch,
                model=net,
                classes=wnids
            ).eval()
            model_dict[method] = model
        else:
            G, path = get_pro_tree(dataset=args.dataset, arch=args.arch, method=method)
            root = validate_tree(G, path, wnids)
            model = SoftNBDT(
                pretrained=False,
                dataset=args.dataset,
                path_graph=path,
                model=net,
                classes=wnids
            ).cuda()
            model_dict[method] = model

    # 为html页面及其存储图片创建目录
    path_list = []
    for CLASS in DATASET_TO_CLASSES[args.dataset]:
        cls_dir = os.path.join(args.img_dir, CLASS)
        if os.path.exists(cls_dir):
            if os.listdir(cls_dir) != 0:
                path_list += [os.path.join(cls_dir, i) for i in os.listdir(cls_dir)]
        if len(path_list) == 0:
            Colors.red("No PIC of cls {} for explain".format(CLASS))
    metric_tree = {}

    for method, net in net_dict.items():
        model = model_dict[method]
        afoc_list = []
        iou_list = []
        for img in tqdm(path_list):
            gc.collect()
            torch.cuda.empty_cache()
            mask_dict = {}
            img2 = cv2.imread(img, 1)
            img_name = os.path.split(img)[1].split('.')[0]

            decisions, leaf_to_prob, node_to_prob, predicted, cls, decision_to_wnid, cam_dict = get_nbdt_inference(args.arch, args.dataset, img, net, model, wnids,
                                                                                                                   (448, 448), args.cam, num_cls, img_name)
            if args.merge == 'complex':
                complex_w = compute_complex_weight(cam_dict, predicted)
                mask_dict = mask_and_inference(decisions[0], img2, cam_dict, model, wnids, args, complex_w)
            else:
                mask_dict = mask_and_inference(decisions[0], img2, cam_dict, model, wnids, args)
            mask_dict['no_mask'] = [node_to_prob, decision_to_wnid]

            change_prob = compute_prob_change(mask_dict)
            change_iou = compute_iou_change(mask_dict)
            afoc_list.append(change_prob)
            iou_list.append(change_iou)
        if len(afoc_list) != 0:
            mean_acof = np.mean(np.array(afoc_list), axis=0).tolist()
            mean_iou = np.mean(np.array(iou_list), axis=0).tolist()
            metric_tree[method] = [mean_acof, mean_iou]
            Colors.green('metric calculation of {} is finished'.format(method))
        else:
            Colors.red('no pic, check again!')
    plot_metric_all_tree(metric_tree, args.img_dir, args.plot_name)