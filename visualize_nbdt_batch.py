from vnbdt import *
import cv2
from nbdt.model import SoftNBDT, HardNBDT
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# 定义获取图的函数

if __name__ == '__main__':

    cam_method = 'efccam'
    dataset = 'Imagenet10'
    method = 'induced'
    arch = 'vgg19_bn'
    # pth_path = './pretrain_model/ckpt-DIYdataset-ResNet18-lr0.01-SoftTreeSupLoss.pth'
    # tar_path = '../FGVC/DFL-CNN/weight/model_best.pth.tar'
    #pkl_path = '/home/lzl001/CNN_train/model_vgg19bn_pretrained.pkl'
    #pkl_path = '/home/lzl001/CNN_train/model_resnet50_pretrained.pkl'

    #pth_path = '/home/lzl001/NBDT/neural-backed-decision-trees/checkpoint/ckpt-Imagenet10-ResNet50-lr0.01-SoftTreeSupLoss.pth'
    pth_path = '/home/lzl001/NBDT/neural-backed-decision-trees/checkpoint/ckpt-Imagenet10-vgg19_bn-lr0.01-SoftTreeSupLoss.pth'
    # 获取该数据集下所有类别的对应ID
    wnids = get_wnids_from_dataset(dataset)
    num_cls = len(wnids)

    # 获取网络
    #net = calling_VGG_from_tar(tar_path, cls_num=10)
    #net = call_pkl_model(arch, pkl_path, cls_num=10)
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
    ori_path = 'imagenet10'

    output_dir = '10-vgg-nbdt'
    html_output = '10-vgg-nbdt-html'

    # output_dir = './IN10-vgg-FT'
    # html_output = './IN10-vgg-FT-html'

    for img in os.listdir(ori_path):
        Colors.cyan("{} has started".format(img))
        path_img1 = os.path.join(ori_path, img)
        path_img2 = os.path.join('..', path_img1)
        type_name = os.path.split(path_img1)[1].split('.')[0] + '_' + dataset + '_' + arch + '_' + cam_method + '-2'
        img_name = os.path.split(path_img1)[1].split('.')[0]

        # 图像读取和预处理，读取的图像img1用来合成CAM，im用来获取x
        # img1 = cv2.imread(path_img1, 1)[:, :, ::-1]
        # img1 = np.float32(img1) / 255
        # im = load_image_from_path(path_img1)
        # x = preprocess_img(im, [0.4948052, 0.48568845, 0.44682974],
        #                        [0.24580306, 0.24236229, 0.2603115])

        img1 = Image.open(path_img1)
        img2 = cv2.imread(path_img1, 1)
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
        )
        # 模块化后不需要.eval()

        decisions, leaf_to_prob, node_to_prob, predicted = forword_tree(x.cuda(), model.cuda(), wnids, dataset)

        decision_to_wnid = get_decision_wnid(decisions[0])
        #record_node_prob(node_to_prob, decision_to_wnid, './experiment/mask_leaf_record.txt', img_name)

        # 获取一个包含所有叶节点对应cam图的字典，此处还未resize，再进行融合

        if arch == "ResNet50":
            target_layers = [net.layer4[-1]]
        elif arch == 'DFLCNN':
            target_layers = [net.conv5]
        elif arch == 'vgg19_bn':
            target_layers = [net.features[-1]]
        elif arch == 'ResNet18':
            target_layers = [net.layer4[1].bn2]
        elif arch == 'wrn28_10_cifar10':
            target_layers = [net.features[-3][-1].body.conv2.conv]
        # 模型哪一个layer

        if cam_method == 'efccam':
            cam_dict = get_all_leaf_cam_efc(x.cuda(), net.cuda(), leaf_to_prob, num_cls,
                                            cam_method, target_layers, path_img1, img_name)
        else:
            cam_dict = get_all_leaf_cam_from_method(x.cuda(), model.model.cuda(), leaf_to_prob,
                                                    num_cls, cam_method, target_layers,
                                                    aug_smooth=True,
                                                    eigen_smooth=True)
        # cam_dict = get_all_leaf_cam(x, model.model, leaf_to_prob,num_cls)
        image_dict, decisions_path_label, decisions_prob = fuse_leaf_cam_from_method(decisions[0],
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
