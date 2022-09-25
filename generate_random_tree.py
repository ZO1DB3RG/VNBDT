from vnbdt import *
import cv2
from nbdt.model import SoftNBDT, HardNBDT

# 定义获取图的函数

if __name__ == '__main__':

    cam_method = 'gradcam'
    dataset = 'Imagenet10'
    method = 'pro'
    arch = 'ResNet50'

    pth_path = '/home/lzl001/NBDT/neural-backed-decision-trees/checkpoint/ckpt-Imagenet10-ResNet50-lr0.01-SoftTreeSupLoss_pro.pth'
    #pth_path = '/home/lzl001/NBDT/neural-backed-decision-trees/checkpoint/ckpt-Imagenet10-ResNet50-lr0.01-SoftTreeSupLoss_random.pth'


    # 获取该数据集下所有类别的对应ID
    wnids = get_wnids_from_dataset(dataset)
    num_cls = len(wnids)

    # 获取网络
    # net = calling_VGG_from_tar(tar_path, cls_num=12)
    #net = call_pkl_model(arch, pkl_path, cls_num=10)
    net = call_pth_model(arch, pth_path, cls_num=10)

    # 生成树并读取
    G, path = get_pro_tree(dataset=dataset, arch=arch, method=method)
    # 验证树及其节点对应，并返回根节点Node
    root = validate_tree(G, path, wnids)

    path_img1 = 'imagenet10/mastiff.jpg'
    output_dir = '10-pro-random'
    html_output = '10-pro-random-html'

    path_img2 = os.path.join('..', path_img1)
    type_name = os.path.split(path_img1)[1].split('.')[0] + '_' + dataset + '_' + method + '_' + arch + '_' + cam_method + '-comp'
    img_name = os.path.split(path_img1)[1].split('.')[0]

    img1 = Image.open(path_img1)
    img2 = cv2.imread(path_img1, 1)
    img2 = np.float32(img2) / 255
    x = preprocess_image(img2,
                         mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])

    #生成random或pro树的decision路径:

    model = SoftNBDT(
        pretrained=False,
        dataset=dataset,
        path_graph=path,
        model=net,
        classes=wnids
    )

    decisions, leaf_to_prob, node_to_prob, predicted = forword_tree(x, model, wnids, dataset)

    decision_to_wnid = get_decision_wnid(decisions[0])
    # record_node_prob(node_to_prob, decision_to_wnid, './experiment/mask_leaf_record.txt', img_name)

    # 获取一个包含所有叶节点对应cam图的字典，此处还未resize，再进行融合

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

    if cam_method == 'efccam':
        cam_dict = get_all_leaf_cam_efc(x.cuda(), net.cuda(), leaf_to_prob, num_cls,
                                        cam_method, target_layers, path_img1, img_name)
    else:
        cam_dict = get_all_leaf_cam_from_method(x.cuda(), model.model.cuda(), leaf_to_prob,
                                                num_cls, cam_method, target_layers,
                                                aug_smooth=False,
                                                eigen_smooth=False)
    # cam_dict = get_all_leaf_cam(x, model.model, leaf_to_prob,num_cls)
    # generate_all_leaf_cam(img2, type_name, cam_dict, dataset, output_dir)
    complex_w = compute_complex_weight(cam_dict, predicted)

    weight = 'complex'

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
    vis_colormap = 'colormap_annotated.png'

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
    tree = change_root(tree, decisions[0], image_dict, root, path_img2)

    if len(decisions_path_label) == len(image_dict):
        Colors.green('trying to insert images into decision nodes')
    else:
        Colors.red('failed to insert because of unmatcheed number of wnidset and image path')

    # 为中间节点、叶节点插入图像
    if decisions[0][-1]['name'] == '(generated)':
        insert_image_for_no_name(tree['children'], decisions_path_label, image_dict, decisions_prob,
                                 vis_image_resize_factor)
    else:
        insert_image(tree['children'], decisions_path_label, image_dict, decisions_prob,
                     vis_image_resize_factor)



    fname = os.path.join(html_output, type_name + '_tree')

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