import os.path

from vnbdt import *
import cv2
from nbdt.model import SoftNBDT, HardNBDT
import argparse
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#定义获取图的函数

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Insert Saliency map for NBDT")
    parser.add_argument(
        "--cam_method", default='gradcam', help="Method of Saliency map"
    )
    parser.add_argument(
        "--dataset", default='CIFAR10'
    )
    parser.add_argument(
        "--method", default='induced', help="Method of constructing tree"
    )
    parser.add_argument(
        "--arch", default='ResNet18', help="Name of CNN"
    )
    parser.add_argument(
        "--model_path", default='', help="Path of model, which is needed to reload the model"
    )
    parser.add_argument(
        "--img_path", default='./test_html/image/origin_image/F22.jpg'
    )
    parser.add_argument(
        "--output_path", default='./test_html/image/plane_out_image'
    )
    parser.add_argument(
        "--html_output", default='./test_html'
    )
    parser.add_argument(
        "--aug_smooth", action="store_true", default=False, help="Test time augmentation: increases the run time by x6."
    )
    parser.add_argument(
        "--eigen_smooth", action="store_true", default=False, help="First principle component of activations*weights"
    )
    parser.add_argument(
        "--pretrained", action="store_true", default=False, help="Tree model pretrained or not, better False"
    )

    args = parser.parse_args()

    #获取该数据集下所有类别的对应ID
    wnids = get_wnids_from_dataset(args.dataset)
    num_cls = len(wnids)

    #获取网络
    net = calling_resnet_modified_pth(args.model_path)
    #生成树并读取
    G, path = get_tree(dataset=args.dataset, arch=args.arch,
                        model=net, method=args.method)
    #验证树及其节点对应，并返回根节点Node
    root = validate_tree(G, path, wnids)

    path_img = args.img_path
    output_dir = args.output_path

    path_img2 = os.path.join('..',path_img)
    type_name = os.path.split(path_img)[1].split('.')[0] + '_' + args.cam_method + '_' + args.arch + '_' + 'test-1'


    #图像读取和预处理，读取的图像img1用来合成CAM，im用来获取x
    img1 = cv2.imread(path_img, 1)[:, :, ::-1]
    img1 = np.float32(img1) / 255
    im = load_image_from_path(path_img)
    x = preprocess_img(im, [0.4948052, 0.48568845, 0.44682974],
                           [0.24580306, 0.24236229, 0.2603115])

    #获取树模型，并装载预训练权重，最后前向推导树，获得决策链路
    #pretrained保持false，否则会重新调取随机的权重，导致CAM不一致

    model = SoftNBDT(
        pretrained=args.pretrained,
        dataset=args.dataset,
        arch=args.arch,
        model=net,
    )
    #模块化后不需要.eval()

    decisions, leaf_to_prob, predicted = forword_tree(x, model, wnids, args.dataset)

    #获取一个包含所有叶节点对应cam图的字典，此处还未resize，再进行融合

    target_layers = [model.model.layer4] #模型哪一个layer
    x = x.cuda()
    cam_dict = get_all_leaf_cam_from_method(x, model.model, leaf_to_prob,
                                            num_cls, args.cam_method, target_layers,
                                            aug_smooth=args.aug_smooth,
                                            eigen_smooth=args.eigen_smooth)
    image_dict, decisions_path_label, \
    decisions_prob = fuse_leaf_cam_from_method(decisions[0],img1,type_name,
                                               cam_dict,output_dir, wnids, predicted)
                                               
    
    
    
    del model, net, x

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
                      dataset=args.dataset,
                      include_leaf_images=vis_leaf_images,
                      image_resize_factor=vis_image_resize_factor,
                      include_fake_sublabels=vis_fake_sublabels,
                      node_to_conf=node_to_conf,
                      wnids=wnids)
    #为root插入图像
    tree = change_root(tree, decisions[0], image_dict, root, path_img2)

    if len(decisions_path_label) == len(image_dict):
        Colors.green('trying to insert images into decision nodes')
    else:
        Colors.red('failed to insert because of unmatcheed number of wnidset and image path')


    #为中间节点、叶节点插入图像
    if decisions[0][-1]['name'] == '(generated)':
        insert_image_for_no_name(tree['children'], decisions_path_label, image_dict, decisions_prob,
                                 vis_image_resize_factor)
    else:
        insert_image(tree['children'], decisions_path_label, image_dict, decisions_prob,
                     vis_image_resize_factor)

    fname = os.path.join(args.html_output, type_name + '_cam_tree')
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
        
    torch.cuda.empty_cache()