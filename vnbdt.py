import cv2
import numpy as np
import numpy.ma as ma
from PIL import Image
from nbdt.model import SoftNBDT, HardNBDT

from DFLCNN.DFL import DFL_VGG16
from nbdt.graph import generate_fname, get_wnids_from_dataset, \
    get_graph_path_from_args, \
    read_graph, get_leaves, \
    get_roots, synset_to_wnid, wnid_to_name, get_root
from nbdt.utils import DATASET_TO_CLASSES, load_image_from_path, maybe_install_wordnet
from collections import defaultdict
import json
from nbdt.utils import Colors, fwd
import os
import torch
import collections
import torchvision.models as models
import torch.nn as nn
from nbdt.hierarchy import generate_hierarchy
from pathlib import Path
from torchvision import transforms
import gc

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad, \
    EFC_CAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image, preprocess_image_resize
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import scale_cam_image

NAME_TO_METHODS = {"gradcam": GradCAM,
                   "scorecam": ScoreCAM,
                   "gradcam++": GradCAMPlusPlus,
                   "ablationcam": AblationCAM,
                   "xgradcam": XGradCAM,
                   "eigencam": EigenCAM,
                   "eigengradcam": EigenGradCAM,
                   "layercam": LayerCAM,
                   "fullgrad": FullGrad,
                   'efccam': EFC_CAM}


def get_seen_wnids(wnid_set, nodes):
    leaves_seen = set()
    for leaf in nodes:
        if leaf in wnid_set:
            wnid_set.remove(leaf)
        if leaf in leaves_seen:
            pass
        leaves_seen.add(leaf)
    return leaves_seen


def match_wnid_leaves(wnids, G, tree_name):
    wnid_set = set()
    for wnid in wnids:
        wnid_set.add(wnid.strip())

    leaves_seen = get_seen_wnids(wnid_set, get_leaves(G))
    return leaves_seen, wnid_set


def match_wnid_nodes(wnids, G, tree_name):
    wnid_set = {wnid.strip() for wnid in wnids}
    leaves_seen = get_seen_wnids(wnid_set, G.nodes)

    return leaves_seen, wnid_set


def print_stats(leaves_seen, wnid_set, tree_name, node_type):
    print(f"[{tree_name}] \t {node_type}: {len(leaves_seen)} \t WNIDs missing from {node_type}: {len(wnid_set)}")
    if len(wnid_set):
        Colors.red(f"==> Warning: WNIDs in wnid.txt are missing from {tree_name} {node_type}")


def build_graph(G):
    return {
        'nodes': [{
            'name': wnid,
            'label': G.nodes[wnid].get('label', ''),
            'id': wnid
        } for wnid in G.nodes],
        'links': [{
            'source': u,
            'target': v
        } for u, v in G.edges]
    }


def get_class_image_from_dataset(dataset, candidate):
    """Returns image for given class `candidate`. Image is PIL."""
    if isinstance(candidate, int):
        candidate = dataset.classes[candidate]
    for sample, label in dataset:
        intersection = compare_wnids(dataset.classes[label], candidate)
        if label == candidate or intersection:
            return sample
    raise UserWarning(f'No samples with label {candidate} found.')


def compare_wnids(label1, label2):
    from nltk.corpus import wordnet as wn  # entire script should not depend on wordnet
    synsets1 = wn.synsets(label1, pos=wn.NOUN)
    synsets2 = wn.synsets(label2, pos=wn.NOUN)
    wnids1 = set(map(synset_to_wnid, synsets1))
    wnids2 = set(map(synset_to_wnid, synsets2))
    return wnids1.intersection(wnids2)


def generate_vis(path_template, data, fname, zoom=2, straight_lines=True,
                 show_sublabels=False, height=750, dark=False, margin_top=20,
                 above_dy=325, y_node_sep=170, hide=[], _print=False, out_dir='.',
                 scale=1, colormap='colormap_annotated.png', below_dy=475, root_y='null',
                 width=1000, margin_left=250):
    with open(path_template, encoding='utf-8') as f:
        html = f.read() \
            .replace(
            "CONFIG_MARGIN_LEFT",
            str(margin_left)) \
            .replace(
            "CONFIG_VIS_WIDTH",
            str(width)) \
            .replace(
            "CONFIG_SCALE",
            str(scale)) \
            .replace(
            "CONFIG_PRINT",
            str(_print).lower()) \
            .replace(
            "CONFIG_HIDE",
            str(hide)) \
            .replace(
            "CONFIG_Y_NODE_SEP",
            str(y_node_sep)) \
            .replace(
            "CONFIG_ABOVE_DY",
            str(above_dy)) \
            .replace(
            "CONFIG_BELOW_DY",
            str(below_dy)) \
            .replace(
            "CONFIG_TREE_DATA",
            json.dumps([data])) \
            .replace(
            "CONFIG_ZOOM",
            str(zoom)) \
            .replace(
            "CONFIG_STRAIGHT_LINES",
            str(straight_lines).lower()) \
            .replace(
            "CONFIG_SHOW_SUBLABELS",
            str(show_sublabels).lower()) \
            .replace(
            "CONFIG_TITLE",
            fname) \
            .replace(
            "CONFIG_VIS_HEIGHT",
            str(height)) \
            .replace(
            "CONFIG_BG_COLOR",
            "#111111" if dark else "#FFFFFF") \
            .replace(
            "CONFIG_TEXT_COLOR",
            '#FFFFFF' if dark else '#000000') \
            .replace(
            "CONFIG_TEXT_RECT_COLOR",
            "rgba(17,17,17,0.8)" if dark else "rgba(255,255,255,1)") \
            .replace(
            "CONFIG_MARGIN_TOP",
            str(margin_top)) \
            .replace(
            "CONFIG_ROOT_Y",
            str(root_y)) \
            .replace(
            "CONFIG_COLORMAP",
            f'''<img src="{colormap}" style="
        position: absolute;
        top: 600px;
        left: 50px;
        height: 250px;
        border: 4px solid #ccc;
        z-index: -1">''' if isinstance(colormap, str) else ''
        )

    os.makedirs(out_dir, exist_ok=True)
    path_html = f'{fname}.html'
    with open(path_html, 'w', encoding='utf-8') as f:
        f.write(html)

    # Colors.green('==> Wrote HTML to {}'.format(path_html))


def get_color_info(G, color, color_leaves, color_path_to=None, color_nodes=()):
    """Mapping from node to color information."""
    nodes = {}
    leaves = list(get_leaves(G))
    if color_leaves:
        for leaf in leaves:
            nodes[leaf] = {'color': color, 'highlighted': True}

    for (id, node) in G.nodes.items():  # 判断如果在color_node有该标签或id的node就为他着相应的顔色，
        # 需要用到该步骤给我们的决策链路进行着色
        if node.get('label', '') in color_nodes or id in color_nodes:
            nodes[id] = {'color': color, 'highlighted': True}

    root = get_root(G)
    target = None
    for leaf in leaves:
        node = G.nodes[leaf]
        if node.get('label', '') == color_path_to or leaf == color_path_to:
            target = leaf
            break

    if target is not None:
        for node in G.nodes:
            nodes[node] = {'color': '#cccccc', 'color_incident_edge': True, 'highlighted': False}

        while target != root:
            nodes[target] = {'color': color, 'color_incident_edge': True, 'highlighted': True}
            view = G.pred[target]
            target = list(view.keys())[0]
        nodes[root] = {'color': color, 'highlighted': True}
    return nodes


def generate_vis_fname(vis_color_path_to=None, vis_out_fname=None, **kwargs):
    fname = vis_out_fname
    if fname is None:
        fname = generate_fname(**kwargs).replace('graph-', f'{kwargs["dataset"]}-', 1)
    if vis_color_path_to is not None:
        fname += '-' + vis_color_path_to
    return fname


def generate_node_conf(node_conf):
    node_to_conf = defaultdict(lambda: {})
    if not node_conf:
        return node_to_conf

    for node, key, value in node_conf:
        if value.isdigit():
            value = int(value)
        node_to_conf[node][key] = value
    return node_to_conf


def set_dot_notation(node, key, value):
    """
    >>> a = {}
    >>> set_dot_notation(a, 'above.href', 'hello')
    >>> a['above']['href']
    'hello'
    """
    curr = last = node
    key_part = key
    if '.' in key:
        for key_part in key.split('.'):
            last = curr
            curr[key_part] = node.get(key_part, {})
            curr = curr[key_part]
    last[key_part] = value


def build_tree(G, root,
               parent='null',
               color_info=(),
               force_labels_left=(),
               include_leaf_images=False,
               dataset=None,
               image_resize_factor=1,
               include_fake_sublabels=False,
               include_fake_labels=False,
               node_to_conf={},
               wnids=[]):
    """
    :param color_info dict[str, dict]: mapping from node labels or IDs to color
                                       information. This is by default just a
                                       key called 'color'
    """
    children = [
        build_tree(G, child, root,
                   color_info=color_info,
                   force_labels_left=force_labels_left,
                   include_leaf_images=include_leaf_images,
                   dataset=dataset,
                   image_resize_factor=image_resize_factor,
                   include_fake_sublabels=include_fake_sublabels,
                   include_fake_labels=include_fake_labels,
                   node_to_conf=node_to_conf,
                   wnids=wnids)
        for child in G.succ[root]]
    _node = G.nodes[root]
    label = _node.get('label', '')
    sublabel = ''

    if root.startswith('f') and label.startswith('(') and not include_fake_labels:
        label = ''

    if root.startswith(
            'f') and not include_fake_sublabels:  # WARNING: hacky, ignores fake wnids -- this will have to be changed lol
        sublabel = ''

    node = {
        'sublabel': sublabel,
        'label': label,
        'parent': parent,
        'children': children,
        'alt': _node.get('alt', ', '.join(map(wnid_to_name, get_leaves(G, root=root)))),
        'id': root
    }

    if label in color_info:
        node.update(color_info[label])

    if root in color_info:
        node.update(color_info[root])

    if label in force_labels_left:
        node['force_text_on_left'] = True

    is_leaf = len(children) == 0

    # 在此处插入图片
    if is_leaf and node['label'] == '':
        node['label'] = DATASET_TO_CLASSES[dataset][wnids.index(node['id'])]

    for key, value in node_to_conf[root].items():
        set_dot_notation(node, key, value)
    return node


###########################################key func ###########################################

def get_tree(dataset='CIFAR10', arch='ResNet18',
             model=None, method='Induced'):
    """
    输入模型后获取树结构并保存，然后再获取结构路径并读取
    返回G图和其路径
    """
    generate_hierarchy(dataset=dataset, arch=arch, model=model, method=method)
    # 根据保存的树结构json文件读取树
    path = get_graph_path_from_args(dataset, method, seed=0, branching_factor=2, extra=0,
                                    no_prune=False, fname='', path='', multi_path=False,
                                    induced_linkage='ward', induced_affinity='euclidean',
                                    checkpoint=None, arch=arch)
    print('==> Reading from {}'.format(path))
    G = read_graph(path)
    return G, path

def get_random_tree(dataset='CIFAR10', arch='ResNet18',
                    model=None, method='random',new=True):
    seed = 0
    if new:
        generate_hierarchy(dataset=dataset, arch=arch, model=model, method=method, seed=seed)
    # 根据保存的树结构json文件读取树
    path = get_graph_path_from_args(dataset, method, seed=seed, branching_factor=2, extra=0,
                                    no_prune=False, fname='', path='', multi_path=False,
                                    induced_linkage='ward', induced_affinity='euclidean',
                                    checkpoint=None, arch=arch)
    print('==> Reading from {}'.format(path))
    G = read_graph(path)
    return G, path

def get_pro_tree(dataset='CIFAR10', arch='ResNet18',method='pro'):
    """
    请根据自己想要构建的结构修改json文件
    """
    seed = 0
    # 根据保存的树结构json文件读取树
    path = get_graph_path_from_args(dataset, method, seed=0, branching_factor=2, extra=0,
                                    no_prune=False, fname='', path='', multi_path=False,
                                    induced_linkage='ward', induced_affinity='euclidean',
                                    checkpoint=None, arch=arch)
    print('==> Reading from {}'.format(path))
    G = read_graph(path)
    return G, path

def validate_tree(G, path, wnids):
    """
    验证树是否匹配
    output:根节点Node类
    """
    G_name = Path(path).stem

    leaves_seen, wnid_set1 = match_wnid_leaves(wnids, G, G_name)  # 从G图中找到叶节点
    print_stats(leaves_seen, wnid_set1, G_name, 'leaves')
    leaves_seen, wnid_set2 = match_wnid_nodes(wnids, G, G_name)  # 从G图中找到中间节点
    print_stats(leaves_seen, wnid_set2, G_name, 'nodes')  # 此处的leaves_seen表示所有节点的wnid

    # 获得根节点
    num_roots = len(list(get_roots(G)))
    root = next(get_roots(G))
    if num_roots == 1:
        Colors.green('Found just 1 root.')
    else:
        Colors.red(f'Found {num_roots} roots. Should be only 1.')
    if len(wnid_set1) == len(wnid_set2) == 0 and num_roots == 1:
        Colors.green("==> All checks pass!")
    else:
        Colors.red('==> Test failed')

    return root


def preprocess_img(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return transform(img)[None]


def scale_keep_ar_min_fixed(img, fixed_min):
    ow, oh = img.size

    if ow < oh:

        nw = fixed_min

        nh = nw * oh // ow

    else:

        nh = fixed_min

        nw = nh * ow // oh
    return img.resize((nw, nh), Image.BICUBIC)


def get_transform():
    transform_list = []

    transform_list.append(transforms.Lambda(lambda img: scale_keep_ar_min_fixed(img, 448)))

    # transform_list.append(transforms.RandomHorizontalFlip(p=0.3))

    transform_list.append(transforms.CenterCrop((448, 448)))

    transform_list.append(transforms.ToTensor())

    transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

    return transforms.Compose(transform_list)

def get_transform_cifar10():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    return transform_test


def transform_onlysize():
    transform_list = []
    transform_list.append(transforms.Resize(448))
    transform_list.append(transforms.CenterCrop((448, 448)))
    transform_list.append(transforms.Pad((42, 42)))
    return transforms.Compose(transform_list)


def forword_tree(x, model, wnids, dataset):
    """
    前向获取decision
    input:input_tensor, TreeModel, 叶节点ID集, 数据集名称
    output:决策路径，叶节点对应的概率, 预测结果
    """
    outputs, decisions, leaf_to_prob, node_to_prob = model.forward_with_decisions(x)
    leaf_to_prob = {wnids.index(k): v for k, v in leaf_to_prob.items()}
    _, predicted = outputs.max(1)
    cls = DATASET_TO_CLASSES[dataset][predicted[0]]
    # print('Prediction:', cls, '// Decisions:',
    #       ', '.join(['{} ({:.2f}%)'.format(info['name'], info['prob'] * 100) for
    #                  info in decisions[0]
    #                  ][1:]))

    return decisions, leaf_to_prob, node_to_prob, predicted[0].item(), cls

def forword_node_logit(x, model):
    """
    前向获取decision
    input:input_tensor, TreeModel, 叶节点ID集, 数据集名称
    output:决策路径，叶节点对应的概率, 预测结果
    """
    wnid_to_logits = model.forward_with_logit(x)

    return wnid_to_logits

def forword_tree_no(x, model, wnids, dataset):
    """
    不以树结构进行inference，得到依据原模型的一条路径
    input:input_tensor, TreeModel, 叶节点ID集, 数据集名称
    output:决策路径，叶节点对应的概率, 预测结果
    """
    outputs, decisions, leaf_to_prob, node_to_prob = model.forward_with_decisions_no_tree(x)
    leaf_to_prob = {wnids.index(k): v for k, v in leaf_to_prob.items()}
    _, predicted = outputs.max(1)
    cls = DATASET_TO_CLASSES[dataset][predicted[0]]
    # print('Prediction:', cls, '// Decisions:',
    #       ', '.join(['{} ({:.2f}%)'.format(info['name'], info['prob'] * 100) for
    #                  info in decisions[0]
    #                  ][1:]))

    return decisions, leaf_to_prob, node_to_prob, predicted[0].item()


#######################以上与NBDT原代码五大差异可以不看################################


#######################以下为链路节点热力图生成融合代码################################

def get_all_leaf_cam(x, model, leaf_to_prob, num_cls):
    # 模块化应该修改此处
    # 用一个字典装下所有叶节点的cam图，记得修改model.py中的forward_with_decisions函数使其返回一字典---叶节点：logit值
    """
    Input: 图像，树模型，叶节点到概率字典【保存了output值】，类别数量
    Output: 一个包含所有叶节点对应cam图的字典，此处还未resize; 特征图
    """

    def backward_hook(module, grad_in, grad_out):
        grad_block.append(grad_out[0].detach())

    def farward_hook(module, input, output):
        fmap_block.append(output)

    cam_dict = dict()
    for leaf in range(num_cls):
        fmap_block = list()
        grad_block = list()
        # model.layer4[-1].conv2.register_forward_hook(farward_hook)
        # model.layer4[-1].conv2.register_backward_hook(backward_hook)
        model.conv6.register_forward_hook(farward_hook)
        model.conv6.register_backward_hook(backward_hook)

        x1, x2, x3, index = model(x)
        output = x1 + x2 + 0.1 * x3
        model.zero_grad()
        class_loss = output[0, leaf]
        class_loss.backward()
        grads_val = grad_block[0].cpu().data.numpy().squeeze()
        fmap = fmap_block[0].cpu().data.numpy().squeeze()
        cam = np.zeros(fmap.shape[1:], dtype=np.float32)  # 4
        grads = grads_val.reshape([grads_val.shape[0], -1])  # 5
        weights = np.mean(grads, axis=1)  # 6
        for i, w in enumerate(weights):
            cam += w * fmap[i, :, :]
        cam = np.maximum(cam, 0)
        cam_dict[leaf] = (leaf_to_prob[leaf], cam)
    return cam_dict

#只对预测类别生成显著图
def get_cam_from_method(x, model, predicted,
                        method, target_layers,
                        aug_smooth=True,
                        eigen_smooth=True):
    cam_method = NAME_TO_METHODS[method]
    device = torch.cuda.is_available()
    with cam_method(model=model, target_layers=target_layers, use_cuda=device) as cam:
        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=x,
                            targets=[ClassifierOutputTarget(predicted)],
                            aug_smooth=aug_smooth,
                            eigen_smooth=eigen_smooth)
        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]
    return grayscale_cam

def get_all_leaf_cam_from_method(x, model, leaf_to_prob,
                                 num_cls, method, target_layers,
                                 aug_smooth=True,
                                 eigen_smooth=True):
    # 模块化应该修改此处，同类型方法
    # 用一个字典装下所有叶节点的cam图，
    """
    Input: 图像，树模型，叶节点到概率字典【保存了output值】，类别数量
    Output: 一个包含所有叶节点对应cam图的字典，此处还未resize; 特征图
    """
    cam_dict = dict()
    device = torch.cuda.is_available()
    for leaf in range(num_cls):
        targets = [ClassifierOutputTarget(leaf)]
        # type_name = type_name + '_leaf#' + str(leaf)
        cam_method = NAME_TO_METHODS[method]
        with cam_method(model=model,
                        target_layers=target_layers,
                        use_cuda=device) as cam:
            cam.batch_size = 4
            grayscale_cam = cam(input_tensor=x,
                                targets=targets,
                                aug_smooth=aug_smooth,
                                eigen_smooth=eigen_smooth)
            # Here grayscale_cam has only one image in the batch
            grayscale_cam = grayscale_cam[0, :]
        cam_dict[leaf] = (leaf_to_prob[leaf], grayscale_cam)
        gc.collect()
        torch.cuda.empty_cache()
        del cam
    return cam_dict

def get_cam_efc(x, model,predicted,
                target_layers,img_path, type_name):
    cam_method = NAME_TO_METHODS['efccam']
    device = torch.cuda.is_available()
    with cam_method(model=model, target_layers=target_layers, use_cuda=device) as cam:
        cam.batch_size = 16
        grayscale_cam = cam.compute_sal(x, img_path, type_name + '-leaf#' + str(predicted),
                                        targets=[ClassifierOutputTarget(predicted)])
    return grayscale_cam

def get_all_leaf_cam_efc(x, model, leaf_to_prob,
                         num_cls, method, target_layers,
                         img_path, type_name):
    # 模块化应该修改此处，同类型方法
    # 用一个字典装下所有叶节点的cam图，
    """
    Input: 图像，树模型，叶节点到概率字典【保存了output值】，类别数量
    Output: 一个包含所有叶节点对应cam图的字典，此处还未resize; 特征图
    """
    cam_dict = dict()
    device = torch.cuda.is_available()
    for leaf in range(num_cls):
        # type_name = type_name + '_leaf#' + str(leaf)
        cam_method = NAME_TO_METHODS[method]
        with cam_method(model=model, target_layers=target_layers, use_cuda=device) as cam:
            cam.batch_size = 16
            grayscale_cam = cam.compute_sal(x, img_path, type_name + '-leaf#' + str(leaf),
                                            targets=[ClassifierOutputTarget(leaf)])
        cam_dict[leaf] = (leaf_to_prob[leaf], grayscale_cam)
        print("==============================")
        print("cam of leaf: {} has been saved".format(leaf))
        print("==============================")
        del cam
    return cam_dict


def fuse_leaf_cam(decisions, img, type_name,
                  cam_dict, output, wnids, predicted):
    """
    将叶节点cam根据决策链路融合并保存
    Input: 决策，图像，命名，cam字典，特征图，输出路径*2，叶节点ID集，预测类别index
    Output: 两个字典：决策链路上的每一个点ID对应其CAM图路径、概率
    """

    H, W, _ = img.shape
    decicion_num = len(decisions)
    image_dict = {'': ''}
    for ind in range(decicion_num - 1):
        # 获取该节点子节点概率和叶节点列表
        child_node = decisions[ind]['node'].new_to_old_classes
        child_prob = decisions[ind]['child_prob']
        name = type_name + '_cam_' + str(ind) + '.jpg'
        cam = np.zeros(cam_dict[0][1].shape, dtype=np.float32)
        for i in range(2):
            w = child_prob[i]
            for node in child_node[i]:
                cam += w.detach().numpy() * cam_dict[node][1]
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        cam = cv2.resize(cam, (W, H))

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        cam_img = 0.3 * heatmap + 0.7 * img

        if not os.path.exists(os.path.join(output, type_name)):
            os.makedirs(os.path.join(output, type_name))
        path_cam_img = os.path.join(output, name)
        cv2.imwrite(path_cam_img, cam_img)
        path_cam_img = os.path.join('..', path_cam_img)

        Colors.green(name + ' has been generated')
        if decisions[ind + 1]['node'] != None:
            image_dict[decisions[ind + 1]['node'].wnid] = path_cam_img
        else:
            image_dict[wnids[predicted]] = path_cam_img

    decisions_path_label = ['']
    decisions_prob = {}

    for i in range(len(decisions)):
        if decisions[i]['node'] != None:
            decisions_path_label.append(decisions[i]['node'].wnid)
            decisions_prob[decisions[i]['node'].wnid] = "{:.2%}".format(decisions[i]['prob'])
        else:
            decisions_path_label.append(wnids[predicted])
            decisions_prob[wnids[predicted]] = "{:.2%}".format(decisions[i]['prob'])

    return image_dict, decisions_path_label, decisions_prob

def fuse_leaf_cam_with_simple_w(decisions, img, type_name,
                                cam_dict, output, wnids, predicted):
    """
    将叶节点cam根据决策链路融合并保存，升级版
    Input: 决策，图像，命名，cam字典，特征图，输出路径*2，叶节点ID集，预测类别index
    Output: 两个字典：决策链路上的每一个点ID对应其CAM图路径、概率
    """
    H, W, _ = img.shape
    decicion_num = len(decisions)
    image_dict = {'': ''}
    for ind in range(decicion_num - 1):
        # 获取该节点子节点概率和叶节点列表
        child_node = decisions[ind]['node'].new_to_old_classes
        child_prob = decisions[ind]['child_prob']
        name = type_name + '_cam_' + str(ind) + '.jpg'
        cam = np.zeros(cam_dict[0][1].shape, dtype=np.float32)
        for i in range(2):
            w = child_prob[i]
            for node in child_node[i]:
                cam += w.detach().cpu().numpy() * cam_dict[node][1]

        scaled = scale_cam_image([cam], (W, H))[0, :]  # resize

        cam_image = show_cam_on_image(img, scaled, use_rgb=True)

        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        if not os.path.exists(os.path.join(output, type_name)):
            os.makedirs(os.path.join(output, type_name))
        path_cam_img = os.path.join(os.path.join(output, type_name), name)
        cv2.imwrite(path_cam_img, cam_image)
        path_cam_img = os.path.join('..', path_cam_img.split('/')[-3], path_cam_img.split('/')[-2], path_cam_img.split('/')[-1])

        # Colors.green(name + ' has been generated')
        if decisions[ind + 1]['node'] != None:
            image_dict[decisions[ind + 1]['node'].wnid] = path_cam_img
        else:
            image_dict[wnids[predicted]] = path_cam_img

    decisions_path_label = ['']
    decisions_prob = {}

    for i in range(len(decisions)):
        if decisions[i]['node'] != None:
            decisions_path_label.append(decisions[i]['node'].wnid)
            decisions_prob[decisions[i]['node'].wnid] = "{:.2%}".format(decisions[i]['prob'])
        else:
            decisions_path_label.append(wnids[predicted])
            decisions_prob[wnids[predicted]] = "{:.2%}".format(decisions[i]['prob'])

    return image_dict, decisions_path_label, decisions_prob

def compute_complex_weight(cam_dict:dict, predicted):
    """
    输入：cam字典(概率值，cam)、预测类别
    输出：cam权重对应向量
    """
    target = cam_dict[predicted][1]
    w_dict = {}
    for cls, cam in cam_dict.items():
        cam = cam[1]
        w_dict[cls] = np.sum(np.abs(target - cam)) / (target.shape[0] * target.shape[1])

    w_dict = list(w_dict.values())
    #w_dict = 1.0 - ((w_dict - min(w_dict)) / (max(w_dict) - min(w_dict)))

    return w_dict


def fuse_leaf_cam_with_complex_w(decisions, img, type_name,complex_w,
                                 cam_dict, output, wnids, predicted):
    """
    将叶节点cam根据决策链路融合并保存，升级版
    Input: 决策，图像，命名， 权重，cam字典，特征图，输出路径*2，叶节点ID集，预测类别index
    Output: 两个字典：决策链路上的每一个点ID对应其CAM图路径、概率
    """
    H, W, _ = img.shape
    decicion_num = len(decisions)
    image_dict = {'': ''}
    for ind in range(decicion_num - 1):
        # 获取该节点子节点概率和叶节点列表
        w_dict = {}
        child_node = decisions[ind]['node'].new_to_old_classes
        name = type_name + '_cam_' + str(ind) + '.jpg'
        cam = np.zeros(cam_dict[0][1].shape, dtype=np.float32)

        # 归一化
        for i in range(2):
            for node in child_node[i]:
                w_dict[node] = complex_w[node]

        ma = np.max(list(w_dict.values()))
        for k in list(w_dict.keys()):
            w_dict[k] = 1 - (w_dict[k] / ma)

        for i in range(2):
            for node in child_node[i]:
                cam += w_dict[node] * cam_dict[node][1]

        scaled = scale_cam_image([cam], (W, H))[0, :]  # resize

        cam_image = show_cam_on_image(img, scaled, use_rgb=True)

        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        if not os.path.exists(os.path.join(output, type_name)):
            os.makedirs(os.path.join(output, type_name))
        path_cam_img = os.path.join(os.path.join(output, type_name), name)
        cv2.imwrite(path_cam_img, cam_image)
        path_cam_img = os.path.join('..', path_cam_img.split('/')[-3], path_cam_img.split('/')[-2], path_cam_img.split('/')[-1])

        # Colors.green(name + ' has been generated')
        if decisions[ind + 1]['node'] != None:
            image_dict[decisions[ind + 1]['node'].wnid] = path_cam_img
        else:
            image_dict[wnids[predicted]] = path_cam_img

    decisions_path_label = ['']
    decisions_prob = {}

    for i in range(len(decisions)):
        if decisions[i]['node'] != None:
            decisions_path_label.append(decisions[i]['node'].wnid)
            decisions_prob[decisions[i]['node'].wnid] = "{:.2%}".format(decisions[i]['prob'])
        else:
            decisions_path_label.append(wnids[predicted])
            decisions_prob[wnids[predicted]] = "{:.2%}".format(decisions[i]['prob'])

    return image_dict, decisions_path_label, decisions_prob

def fuse_leaf_cam_without_w(decisions, img, type_name,
                              cam_dict, output, wnids, predicted):
    """
    将叶节点cam根据决策链路融合并保存，升级版
    Input: 决策，图像，命名，cam字典，特征图，输出路径*2，叶节点ID集，预测类别index
    Output: 两个字典：决策链路上的每一个点ID对应其CAM图路径、概率
    """
    H, W, _ = img.shape
    decicion_num = len(decisions)
    image_dict = {'': ''}
    for ind in range(decicion_num - 1):
        # 获取该节点子节点概率和叶节点列表
        child_node = decisions[ind]['node'].new_to_old_classes
        child_prob = decisions[ind]['child_prob']
        name = type_name + '_cam_' + str(ind) + '.jpg'
        cam = np.zeros(cam_dict[0][1].shape, dtype=np.float32)
        for i in range(2):
            w = child_prob[i]
            for node in child_node[i]:
                cam += cam_dict[node][1]

        scaled = scale_cam_image([cam], (W, H))[0, :]  # resize

        cam_image = show_cam_on_image(img, scaled, use_rgb=True)

        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        if not os.path.exists(os.path.join(output, type_name)):
            os.makedirs(os.path.join(output, type_name))
        path_cam_img = os.path.join(os.path.join(output, type_name), name)
        cv2.imwrite(path_cam_img, cam_image)
        path_cam_img = os.path.join('..', path_cam_img.split('/')[-3],path_cam_img.split('/')[-2], path_cam_img.split('/')[-1])

        # Colors.green(name + ' has been generated')
        if decisions[ind + 1]['node'] != None:
            image_dict[decisions[ind + 1]['node'].wnid] = path_cam_img
        else:
            image_dict[wnids[predicted]] = path_cam_img

    decisions_path_label = ['']
    decisions_prob = {}

    for i in range(len(decisions)):
        if decisions[i]['node'] != None:
            decisions_path_label.append(decisions[i]['node'].wnid)
            decisions_prob[decisions[i]['node'].wnid] = "{:.2%}".format(decisions[i]['prob'])
        else:
            decisions_path_label.append(wnids[predicted])
            decisions_prob[wnids[predicted]] = "{:.2%}".format(decisions[i]['prob'])

    return image_dict, decisions_path_label, decisions_prob

def generate_all_leaf_cam(img, type_name, cam_dict, dataset, output):
    H, W, _ = img.shape
    for leaf, cam in cam_dict.items():
        name = type_name + '_leaf_' + DATASET_TO_CLASSES[dataset][leaf] + '.jpg'
        #scaled = scale_cam_image([cam[1]], (W, H))[0, :]  # resize

        cam_image = show_cam_on_image(img, cam[1], use_rgb=True)

        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        if not os.path.exists(os.path.join(output, type_name)):
            os.makedirs(os.path.join(output, type_name))
        path_cam_img = os.path.join(os.path.join(output, type_name), name)
        cv2.imwrite(path_cam_img, cam_image)
        print('cam of cls: {} generated'.format(DATASET_TO_CLASSES[dataset][leaf]))


#######################以下为量化测试用的代码################################

def generate_cam_mask(decisions, img, type_name, cam_dict, output):
    H, W, _ = img.shape
    decicion_num = len(decisions)
    for ind in range(decicion_num - 1):
        # 获取该节点子节点概率和叶节点列表
        child_node = decisions[ind]['node'].new_to_old_classes
        child_prob = decisions[ind]['child_prob']
        name = type_name + '_masked_leaf_' + str(ind) + '.jpg'
        cam = np.zeros(cam_dict[0][1].shape, dtype=np.float32)
        for i in range(2):
            w = child_prob[i]
            for node in child_node[i]:
                cam += w.detach().cpu().numpy() * cam_dict[node][1]

        scaled = scale_cam_image([cam], (W, H))[0, :]  # resize
        #mean = np.percentile(scaled, 80)

        #可以在此处修改遮盖方式xxxxx
        img_to_mask = img.copy()
        img_to_mask[scaled > 0.7] = 0

        if not os.path.exists(os.path.join(output, type_name)):
            os.makedirs(os.path.join(output, type_name))
        path_cam_img = os.path.join(os.path.join(output, type_name), name)
        cv2.imwrite(path_cam_img, img_to_mask)

        Colors.green(name + ' has been generated')

def generate_cam_mask_one_sample(img, type_name, cam, output):
    H, W, _ = img.shape
    if not os.path.exists(os.path.join(output, type_name)):
        os.makedirs(os.path.join(output, type_name))
    scaled = scale_cam_image([cam], (W, H))[0, :]  # resize
    img2keep = img.copy()
    img2keep[scaled <= 0.3] = 0  # 保留重要区域
    scaled_ = ma.array(scaled, mask=scaled > 0.3)

    max, min = scaled_.max(), scaled_.min()
    scaled_ = ((scaled_ - min) / (max - min))
    remove_pixel = scaled_.compressed()
    remove_pixel = remove_pixel[remove_pixel > 0]
    # plt.hist(remove_pixel,bins = 100)
    # plt.show()
    remove_pixel = [np.percentile(remove_pixel, x * 10) for x in range(1, 10)]
    scaled_ = scaled_.filled(0)

    for i, odd in enumerate(remove_pixel):
        name = type_name + '_masked_' + str(int(i + 1)) + '.jpg'
        # 可以在此处修改遮盖方式xxxxx

        img2remove = img.copy()

        img2remove[scaled_ <= odd] = 0

        img2remove = img2keep + img2remove

        path_cam_img = os.path.join(os.path.join(output, type_name), name)
        cv2.imwrite(path_cam_img, img2remove)
    Colors.cyan("mask of img {} has been generated".format(type_name))


def get_layer(arch, model):
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

    return target_layers

def generate_html(G, root, arch, dataset, cam_method, path_img, net, wnids, num_cls,
                  output_dir, html_output, size, name, weight, ori_cls: None):
    """
    :param G: tree structure
    :param root: root node
    :param arch: (str) model architecture
    :param dataset: (str) dataset name
    :param cam_method: (str) explainable methods of CAM
    :param path_img: (str) image relative path
    :param net: (model) pytorch model
    :param wnids: (list) leaf node id
    :param num_cls: number of leaves
    :param output_dir: (str) directory of output explanation CAM
    :param html_output: (str) directory of output html
    :param size: (set) output image size your explanation desires
    :param name: (str) a file identifier
    :param weight: methods of get inner nodes: (complex & simple & None)
    :return: get explanation
    """

    path_img2 = os.path.join('..', os.path.join(path_img.split('/')[-2], path_img.split('/')[-1])) #该路径用于插入root图像
    type_name = os.path.split(path_img)[1].split('.')[0].split('_')[-1] \
                    + '_' + cam_method + '-' + weight

    img_name = os.path.split(path_img)[1].split('.')[0]

    # 图像读取和预处理，读取的图像img1用来合成CAM，im用来获取x
    # img1 = cv2.imread(path_img1, 1)[:, :, ::-1]
    # img1 = np.float32(img1) / 255
    # im = load_image_from_path(path_img1)
    # x = preprocess_img(im, [0.4948052, 0.48568845, 0.44682974],
    #                        [0.24580306, 0.24236229, 0.2603115])

    if dataset != 'FGVC12':
        if size != None:
            img2 = cv2.resize(cv2.imread(path_img, 1), size)[:, :, ::-1]
        else:
            img2 = cv2.imread(path_img, 1)[:, :, ::-1]
    else:
        img2 = cv2.resize(cv2.imread(path_img, 1), (448, 448))[:, :, ::-1]

    img2 = np.float32(img2) / 255
    x = preprocess_image(img2,
                         mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])

    # 获取树模型，并装载预训练权重，最后前向推导树，获得决策链路
    # pretrained保持false，否则会重新调取随机的权重，导致CAM不一致

    model = SoftNBDT(
        pretrained=False,
        dataset=dataset,
        arch=arch,
        model=net,
        classes=wnids
    ).cuda()

    decisions, leaf_to_prob, node_to_prob, predicted, cls = forword_tree(x.cuda(), model, wnids, dataset)
    if ori_cls == cls:
        type_name = 'T-' + type_name
    else:
        type_name = 'F-' + type_name
    # decision_to_wnid = get_decision_wnid(decisions[0])
    # record_node_prob(node_to_prob, decision_to_wnid, './experiment/mask_leaf_record.txt', img_name)

    # 获取一个包含所有叶节点对应cam图的字典，此处还未resize，再进行融合
    # 模型哪一个layer
    target_layers = get_layer(arch, model)


    if cam_method == 'efccam':
        cam_dict = get_all_leaf_cam_efc(x.cuda(), net.cuda(), leaf_to_prob, num_cls,
                                        cam_method, target_layers, path_img, img_name)
    else:
        cam_dict = get_all_leaf_cam_from_method(x.cuda(), net.cuda(), leaf_to_prob,
                                                num_cls, cam_method, target_layers,
                                                aug_smooth=False,
                                                eigen_smooth=False)

    # if len(cam_dict) != 0:
    #     print("============================= successfully get all leaf CAM =============================")
    # else:
    #     print("============================= fail to get the leaf CAM, nothing here =============================")

    # 生成所有叶节点热力图
    #generate_all_leaf_cam(img2, type_name, cam_dict, dataset, output_dir)


    #不同添加权重的方法:
    if weight == 'simple':
        image_dict, decisions_path_label, decisions_prob = fuse_leaf_cam_with_simple_w(decisions[0],
                                                                                       img2, type_name, cam_dict,
                                                                                       output_dir, wnids, predicted)
    elif weight == 'complex':
        complex_w = compute_complex_weight(cam_dict, predicted)
        image_dict, decisions_path_label, decisions_prob = fuse_leaf_cam_with_complex_w(decisions[0],img2, type_name,
                                                                                        complex_w, cam_dict,
                                                                                       output_dir, wnids, predicted)
    else:
        image_dict, decisions_path_label, decisions_prob = fuse_leaf_cam_without_w(decisions[0],
                                                                                     img2, type_name, cam_dict,
                                                                                     output_dir, wnids, predicted)
    # image_dict, decisions_path_label, \
    # decisions_prob = fuse_leaf_cam(decisions[0], img2,
    #                                            type_name, cam_dict,
    #                                            output_dir, wnids, predicted)

    """
    接下来基本都是HTML生成相关，相关参数可以args传入，设置一个默认值
    """
    color = '#cccccc'
    vis_no_color_leaves = False  # 是否要为叶节点着色
    vis_color_path_to = None  # ？？？
    vis_color_nodes = tuple(decisions_path_label)
    vis_node_conf = {}
    vis_force_labels_left = {}
    vis_leaf_images = False
    vis_image_resize_factor = 1
    vis_fake_sublabels = False  # 可以自己操作节点名称
    vis_zoom = 1.5  # 控制大小？
    vis_curved = True  # 是否要弯曲直线？
    vis_sublabels = True  # 可以用来增加概率？
    vis_height = 800
    vis_width = 1200
    vis_dark = False
    vis_margin_top = 20
    vis_margin_left = 250
    vis_hide = []
    vis_above_dy = 325
    vis_below_dy = 475
    vis_scale = 1
    vis_root_y = 'null'
    vis_colormap = os.path.join('..', os.path.split(output_dir)[-1], type_name, 'metrics.png')

    color_info = get_color_info(
        G,
        color,
        color_leaves=not vis_no_color_leaves,
        color_path_to=vis_color_path_to,
        color_nodes=vis_color_nodes or ())

    node_to_conf = generate_node_conf(vis_node_conf)

    # 在构建数的过程中已经将所有的需要插入的图片完成了插入，并以href的形式出现
    tree = build_tree(G, root,
                      color_info=color_info,
                      force_labels_left=vis_force_labels_left or [],
                      dataset=dataset,
                      include_leaf_images=vis_leaf_images,
                      image_resize_factor=vis_image_resize_factor,
                      include_fake_sublabels=vis_fake_sublabels,
                      node_to_conf=node_to_conf,
                      wnids=wnids)
    # 为root插入图像
    tree = change_root(tree, decisions[0], image_dict, root, path_img2, dataset)

    if len(decisions_path_label) != len(image_dict):
        Colors.red('failed to insert because of unmatcheed number of wnidset and image path')
        exit(1)

    # 为中间节点、叶节点插入图像
    if decisions[0][-1]['name'] == '(generated)':
        insert_image_for_no_name(tree['children'], decisions_path_label, image_dict, decisions_prob,
                                 vis_image_resize_factor)
    else:
        insert_image(tree['children'], decisions_path_label, image_dict, decisions_prob,
                     vis_image_resize_factor)

    fname = os.path.join(html_output, type_name + '_cam_tree')

    parent = Path(fwd()).parent

    generate_vis(
        str(parent / 'nbdt/templates/tree-template-insert.html'), tree, fname,
        zoom=vis_zoom,
        straight_lines=not vis_curved,
        show_sublabels=vis_sublabels,
        height=vis_height,
        width=vis_width,
        dark=vis_dark,
        margin_top=vis_margin_top,
        margin_left=vis_margin_left,
        hide=vis_hide or [],
        above_dy=vis_above_dy,
        below_dy=vis_below_dy,
        scale=vis_scale,
        root_y=vis_root_y,
        colormap=vis_colormap)

def generate_pro_html(G, root, method, path, arch, dataset, cam_method, path_img, net, wnids, num_cls,
                      output_dir, html_output, size, name, weight, ori_cls):
    path_img2 = os.path.join('..', os.path.join(path_img.split('/')[-2], path_img.split('/')[-1]))  # 该路径用于插入root图像
    type_name = os.path.split(path_img)[1].split('.')[0].split('_')[-1] \
                 + '_' + cam_method + '-' + weight
    img_name = os.path.split(path_img)[1].split('.')[0]

    # 图像读取和预处理，读取的图像img1用来合成CAM，im用来获取x
    # img1 = cv2.imread(path_img1, 1)[:, :, ::-1]
    # img1 = np.float32(img1) / 255
    # im = load_image_from_path(path_img1)
    # x = preprocess_img(im, [0.4948052, 0.48568845, 0.44682974],
    #                        [0.24580306, 0.24236229, 0.2603115])

    if dataset != 'FGVC12':
        if size != None:
            img2 = cv2.resize(cv2.imread(path_img, 1), size)[:, :, ::-1]
        else:
            img2 = cv2.imread(path_img, 1)[:, :, ::-1]
    else:
        img2 = cv2.resize(cv2.imread(path_img, 1), (448, 448))[:, :, ::-1]

    img2 = np.float32(img2) / 255
    x = preprocess_image(img2,
                         mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])

    # 获取树模型，并装载预训练权重，最后前向推导树，获得决策链路
    # pretrained保持false，否则会重新调取随机的权重，导致CAM不一致

    model = SoftNBDT(
        pretrained=False,
        dataset=dataset,
        path_graph=path,
        model=net,
        classes=wnids
    ).cuda()
    decisions, leaf_to_prob, node_to_prob, predicted, cls = forword_tree(x.cuda(), model, wnids, dataset)

    if ori_cls == cls:
        type_name = 'T-' + type_name
    else:
        type_name = 'F-' + type_name
    # decision_to_wnid = get_decision_wnid(decisions[0])
    # record_node_prob(node_to_prob, decision_to_wnid, './experiment/mask_leaf_record.txt', img_name)

    # 获取一个包含所有叶节点对应cam图的字典，此处还未resize，再进行融合
    # 模型哪一个layer

    target_layers = get_layer(arch, model)


    if cam_method == 'efccam':
        cam_dict = get_all_leaf_cam_efc(x.cuda(), net.cuda(), leaf_to_prob, num_cls,
                                        cam_method, target_layers, path_img, img_name)
    else:
        cam_dict = get_all_leaf_cam_from_method(x.cuda(), net.cuda(), leaf_to_prob,
                                                num_cls, cam_method, target_layers,
                                                aug_smooth=False,
                                                eigen_smooth=False)

    # 生成所有叶节点热力图
    # generate_all_leaf_cam(img2, type_name, cam_dict, dataset, output_dir)
    complex_w = compute_complex_weight(cam_dict, predicted)

    #不同添加权重的方法:
    if weight == 'simple':
        image_dict, decisions_path_label, decisions_prob = fuse_leaf_cam_with_simple_w(decisions[0],
                                                                                       img2, type_name, cam_dict,
                                                                                       output_dir, wnids, predicted)
    elif weight == 'complex':
        image_dict, decisions_path_label, decisions_prob = fuse_leaf_cam_with_complex_w(decisions[0],img2, type_name,
                                                                                        complex_w, cam_dict,
                                                                                       output_dir, wnids, predicted)
    else:
        image_dict, decisions_path_label, decisions_prob = fuse_leaf_cam_without_w(decisions[0],
                                                                                     img2, type_name, cam_dict,
                                                                                     output_dir, wnids, predicted)
    # image_dict, decisions_path_label, \
    # decisions_prob = fuse_leaf_cam(decisions[0], img2,
    #                                            type_name, cam_dict,
    #                                            output_dir, wnids, predicted)

    """
    接下来基本都是HTML生成相关，相关参数可以args传入，设置一个默认值
    """
    color = '#cccccc'
    vis_no_color_leaves = False  # 是否要为叶节点着色
    vis_color_path_to = None  # ？？？
    vis_color_nodes = tuple(decisions_path_label)
    vis_node_conf = {}
    vis_force_labels_left = {}
    vis_leaf_images = False
    vis_image_resize_factor = 1
    vis_fake_sublabels = False  # 可以自己操作节点名称
    vis_zoom = 1.5  # 控制大小？
    vis_curved = True  # 是否要弯曲直线？
    vis_sublabels = True  # 可以用来增加概率？
    vis_height = 800
    vis_width = 1200
    vis_dark = False
    vis_margin_top = 20
    vis_margin_left = 250
    vis_hide = []
    vis_above_dy = 325
    vis_below_dy = 475
    vis_scale = 1
    vis_root_y = 'null'
    vis_colormap = os.path.join('..', os.path.split(output_dir)[-1], type_name, 'metrics.png')

    color_info = get_color_info(
        G,
        color,
        color_leaves=not vis_no_color_leaves,
        color_path_to=vis_color_path_to,
        color_nodes=vis_color_nodes or ())

    node_to_conf = generate_node_conf(vis_node_conf)

    # 在构建数的过程中已经将所有的需要插入的图片完成了插入，并以href的形式出现
    tree = build_tree(G, root,
                      color_info=color_info,
                      force_labels_left=vis_force_labels_left or [],
                      dataset=dataset,
                      include_leaf_images=vis_leaf_images,
                      image_resize_factor=vis_image_resize_factor,
                      include_fake_sublabels=vis_fake_sublabels,
                      node_to_conf=node_to_conf,
                      wnids=wnids)
    # 为root插入图像
    tree = change_root(tree, decisions[0], image_dict, root, path_img2, dataset)

    if len(decisions_path_label) != len(image_dict):
        Colors.red('failed to insert because of unmatcheed number of wnidset and image path')
        exit(0)

    # 为中间节点、叶节点插入图像
    if decisions[0][-1]['name'] == '(generated)':
        insert_image_for_no_name(tree['children'], decisions_path_label, image_dict, decisions_prob,
                                 vis_image_resize_factor)
    else:
        insert_image(tree['children'], decisions_path_label, image_dict, decisions_prob,
                     vis_image_resize_factor)

    fname = os.path.join(html_output, type_name + '_cam_tree')

    parent = Path(fwd()).parent

    generate_vis(
        str(parent / 'nbdt/templates/tree-template-insert.html'), tree, fname,
        zoom=vis_zoom,
        straight_lines=not vis_curved,
        show_sublabels=vis_sublabels,
        height=vis_height,
        width=vis_width,
        dark=vis_dark,
        margin_top=vis_margin_top,
        margin_left=vis_margin_left,
        hide=vis_hide or [],
        above_dy=vis_above_dy,
        below_dy=vis_below_dy,
        scale=vis_scale,
        root_y=vis_root_y,
        colormap=vis_colormap)


############################################################

def change_root(tree, decisions, image_dict, root, path_img2, dataset):
    """
    为root节点插入图像
    根据dataset调整位置
    """
    if decisions[-1]['name'] == '(generated)':
        image_dict[root] = path_img2
        if tree['id'] == root:
            tree['color'] = '#FF0000'  # 增加判断正确与否的颜色判断
            tree['highlighted'] = True
            if dataset != 'Emo':
                tree['image'] = {'href': image_dict[root],
                                 'width': 128 * 1.5,
                                 'heitht': 128 * 1.5}
            else:
                tree['image'] = {'href': image_dict[root],
                                 'width': 128 * 1.5,
                                 'heitht': 128 * 1.5,
                                 'x': -150}
            tree['sublabel'] = 1
            tree['label'] = ''
    else:
        image_dict['root'] = path_img2
        if tree['id'] == root:
            tree['color'] = '#FF0000'
            tree['highlighted'] = True
            tree['image'] = {'href': image_dict['root'],
                             'width': 128 * 1.5,
                             'heitht': 128 * 1.5}
            tree['sublabel'] = 1
            tree['label'] = os.path.split(path_img2)[1].split('.')[0]

    return tree

#讲dict的node对应节点概率变化记录到txt文件中
def record_node_prob(node_to_prob:dict, decision_to_wnid:list, file_path, img_name):
    with open(file_path, 'a') as f:
        f.write(img_name + '--->')
        for key, val in node_to_prob.items():
            f.write(key)
            f.write('=')
            f.write("%.4f" % val.detach().item())
            f.write('|')
        f.write("--".join(decision_to_wnid))
        f.write('\n')

#获取决策链路的wnid
def get_decision_wnid(decisions):
    decision_to_wnid = []
    for i in range(len(decisions) - 1):
        decision_to_wnid.append(decisions[i]['node'].wnid)
    leaf = decisions[-2]['node'].children[decisions[-2]['child_prob'].argmax()]
    decision_to_wnid.append(leaf)
    return decision_to_wnid
#######################################获取权重 ############################################

def calling_torchvision_model_pth(pth_path, cls_num):
    return None

def call_pth_model(arch:str, path:str, cls_num:int, device = 'cuda'):
    if device == 'cuda':
        state_dict = torch.load(path)['net']
    else:
        state_dict = torch.load(path,map_location=torch.device('cpu'))['net']
    state = collections.OrderedDict(
        [(k.replace('module.', ''), v) for k, v in state_dict.items()])
    if arch == 'ResNet50':
        model = models.resnet50(pretrained = False)
        model.fc = torch.nn.Linear(in_features=2048, out_features=cls_num, bias=True)
        model.load_state_dict(state)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained = False)
        model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                               torch.nn.ReLU(),
                                               torch.nn.Dropout(p=0.5),
                                               torch.nn.Linear(4096, 4096),
                                               torch.nn.ReLU(),
                                               torch.nn.Dropout(p=0.5),
                                               torch.nn.Linear(4096, cls_num))
        model.load_state_dict(state)
    elif arch == 'ResNet18':
        resnet = models.resnet50(pretrained=False)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        resnet.fc = nn.Linear(in_features=512, out_features=cls_num, bias=True)
        resnet.load_state_dict(state)
    return model


def call_tar_model(arch:str, path:str, cls_num:int):
    state_dict = torch.load(path)['state_dict']
    # state = collections.OrderedDict(
    #     [(k.replace('module.', ''), v) for k, v in state_dict.items()])
    if arch == 'ResNet50':
        model = models.resnet50(pretrained = False)
        model.fc = torch.nn.Linear(in_features=2048, out_features=cls_num, bias=True)
        model.load_state_dict(state_dict)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained = False)
        model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                               torch.nn.ReLU(),
                                               torch.nn.Dropout(p=0.5),
                                               torch.nn.Linear(4096, 4096),
                                               torch.nn.ReLU(),
                                               torch.nn.Dropout(p=0.5),
                                               torch.nn.Linear(4096, cls_num))
        model.load_state_dict(state_dict)
    elif arch == 'ResNet18':
        resnet = models.resnet50(pretrained=False)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        resnet.fc = nn.Linear(in_features=512, out_features=cls_num, bias=True)
        resnet.load_state_dict(state_dict)
    return model

def calling_DNLCNN_from_tar(path, cls_num):
    model = DFL_VGG16(k=10, nclass=cls_num)
    if torch.cuda.is_available():
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location='cpu')

    ckpt_new = collections.OrderedDict()
    for layer in checkpoint['state_dict'].keys():
        ckpt_new[layer[7:]] = checkpoint['state_dict'][layer]
    model.load_state_dict(ckpt_new)

    return model

def call_pkl_model(arch:str,path:str, cls_num:int, device = 'cuda'):
    if device == 'cuda':
        state = torch.load(path)
    else:
        state = torch.load(path,map_location=torch.device('cpu'))
    state = collections.OrderedDict(
        [(k.replace('module.', ''), v) for k, v in state.items()])
    if arch == 'ResNet50':
        model = models.resnet50(pretrained = False)
        model.fc = torch.nn.Linear(in_features=2048, out_features=cls_num, bias=True)
        model.load_state_dict(state)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained = False)
        model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                               torch.nn.ReLU(),
                                               torch.nn.Dropout(p=0.5),
                                               torch.nn.Linear(4096, 4096),
                                               torch.nn.ReLU(),
                                               torch.nn.Dropout(p=0.5),
                                               torch.nn.Linear(4096, cls_num))

        model.load_state_dict(state)
    return model

def insert_image(tree_node, label_list, image_dict, decisions_prob,
                 vis_image_resize_factor=1):
    if len(tree_node) != 0:
        for node in tree_node:
            if len(node['children']) == 0 and node['label'] in label_list:
                node['sublabel'] = decisions_prob[node['label']]
                node['color'] = '#1E90FF'
                node['highlighted'] = True
                node['image'] = {'href': image_dict[node['label']],
                                 'width': 64 * vis_image_resize_factor,
                                 'heitht': 64 * vis_image_resize_factor}
            elif len(node['children']) != 0 and node['label'] in label_list:
                node['sublabel'] = decisions_prob[node['label']]
                node['color'] = '#1E90FF'
                node['highlighted'] = True
                node['image'] = {'href': image_dict[node['label']],
                                 'width': 64 * vis_image_resize_factor,
                                 'heitht': 64 * vis_image_resize_factor}
                insert_image(node['children'], label_list, image_dict, decisions_prob)
            elif len(node['children']) != 0 and node['label'] not in label_list:
                insert_image(node['children'], label_list, image_dict, decisions_prob)
            else:
                pass


def insert_image_for_no_name(tree_node, label_list, image_dict, decisions_prob,
                             vis_image_resize_factor=1):
    if len(tree_node) != 0:
        for node in tree_node:
            if len(node['children']) == 0 and node['id'] in label_list:
                node['sublabel'] = decisions_prob[node['id']]
                node['color'] = '#1E90FF'
                node['highlighted'] = True
                node['image'] = {'href': image_dict[node['id']],
                                 'width': 128 * vis_image_resize_factor,
                                 'heitht': 128 * vis_image_resize_factor}
            elif len(node['children']) != 0 and node['id'] in label_list:
                node['sublabel'] = decisions_prob[node['id']]
                node['color'] = '#1E90FF'
                node['highlighted'] = True
                node['image'] = {'href': image_dict[node['id']],
                                 'width': 128 * vis_image_resize_factor,
                                 'heitht': 128 * vis_image_resize_factor}
                insert_image_for_no_name(node['children'], label_list, image_dict, decisions_prob)
            elif len(node['children']) != 0 and node['id'] not in label_list:
                insert_image_for_no_name(node['children'], label_list, image_dict, decisions_prob)
            else:
                pass


def generate_cam_mask_with_simple_w(decisions, img, type_name, cam_dict, output):
    H, W, _ = img.shape
    decicion_num = len(decisions)
    for ind in range(decicion_num - 1):
        if not os.path.exists(os.path.join(os.path.join(output, type_name), 'node_'+str(ind + 1))):
            os.makedirs(os.path.join(os.path.join(output, type_name), 'node_'+str(ind + 1)))

        # 获取该节点子节点概率和叶节点列表
        child_node = decisions[ind]['node'].new_to_old_classes
        child_prob = decisions[ind]['child_prob']

        cam = np.zeros(cam_dict[0][1].shape, dtype=np.float32)
        for i in range(2):
            w = child_prob[i]
            for node in child_node[i]:
                cam += w.detach().cpu().numpy() * cam_dict[node][1]

        scaled = scale_cam_image([cam], (W, H))[0, :]  # resize

        img2keep = img.copy()
        img2keep[scaled <= 0.3] = 0  # 保留重要区域
        scaled_ = ma.array(scaled, mask=scaled > 0.3)

        max,min = scaled_.max(), scaled_.min()
        scaled_ = ((scaled_ - min) / (max - min))
        remove_pixel = scaled_.compressed()
        remove_pixel = remove_pixel[remove_pixel>0]
        # plt.hist(remove_pixel,bins = 100)
        # plt.show()
        remove_pixel = [np.percentile(remove_pixel, x * 10) for x in range(1,10)]
        scaled_ = scaled_.filled(0)


        for i, odd in enumerate(remove_pixel):
            name = type_name + '_masked_leaf_' + str(ind) + '_' + str(int(i+1)) + '.jpg'
            #可以在此处修改遮盖方式xxxxx

            img2remove = img.copy()

            img2remove[scaled_ <= odd] = 0

            img2remove = img2keep + img2remove

            path_cam_img = os.path.join(os.path.join(os.path.join(output, type_name), 'node_'+str(ind + 1)), name)
            cv2.imwrite(path_cam_img, img2remove)

            #Colors.green(name + ' has been generated')
        Colors.cyan("decision {} of img {} has been generated".format(str(ind), type_name))

def generate_cam_mask_with_complex_w(decisions, img, type_name, cam_dict,
                                     output, complex_w, predicted):
    H, W, _ = img.shape
    decicion_num = len(decisions)
    for ind in range(decicion_num - 1):
        if not os.path.exists(os.path.join(os.path.join(output, type_name), 'node_'+str(ind + 1))):
            os.makedirs(os.path.join(os.path.join(output, type_name), 'node_'+str(ind + 1)))

        # 获取该节点子节点概率和叶节点列表
        child_node = decisions[ind]['node'].new_to_old_classes
        w_dict = {}
        cam = np.zeros(cam_dict[0][1].shape, dtype=np.float32)

        for i in range(2):
            for node in child_node[i]:
                w_dict[node] = complex_w[node]

        m = np.max(list(w_dict.values()))
        for k in list(w_dict.keys()):
            w_dict[k] = 1 - (w_dict[k] / m)

        for i in range(2):
            for node in child_node[i]:
                cam += w_dict[node] * cam_dict[node][1]

        scaled = scale_cam_image([cam], (W, H))[0, :]  # resize
        img2keep = img.copy()
        img2keep[scaled <= 0.3] = 0  # 保留重要区域
        scaled_ = ma.array(scaled, mask=scaled > 0.3)

        max, min = scaled_.max(), scaled_.min()
        scaled_ = ((scaled_ - min) / (max - min))
        remove_pixel = scaled_.compressed()
        remove_pixel = remove_pixel[remove_pixel>0]

        remove_pixel = [np.percentile(remove_pixel, x * 10) for x in range(1,10)]
        scaled_ = scaled_.filled(0)


        for i, odd in enumerate(remove_pixel):
            name = type_name + '_masked_leaf_' + str(ind) + '_' + str(int(i+1)) + '.jpg'
            #可以在此处修改遮盖方式xxxxx

            img2remove = img.copy()

            img2remove[scaled_ <= odd] = 0

            img2remove = img2keep + img2remove

            path_cam_img = os.path.join(os.path.join(os.path.join(output, type_name), 'node_'+str(ind + 1)), name)
            cv2.imwrite(path_cam_img, img2remove)

            #Colors.green(name + ' has been generated')
        Colors.cyan("decision {} of img {} has been generated".format(str(ind), type_name))
