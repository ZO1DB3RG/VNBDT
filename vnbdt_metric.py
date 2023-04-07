import matplotlib.pyplot as plt
from vnbdt import *

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


def get_nbdt_inference(arch, dataset, path_img, net, model, wnids, size, cam, num_cls, img_name):

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


    decisions, leaf_to_prob, node_to_prob, predicted, cls = forword_tree(x.cuda(), model.cuda(), wnids, dataset)

    decision_to_wnid = get_decision_wnid(decisions[0])

    target_layers = get_layer(arch, model)

    if cam == 'efccam':
        cam_dict = get_all_leaf_cam_efc(x.cuda(), net.cuda(), leaf_to_prob, num_cls,
                                        cam, target_layers, path_img, img_name)
    else:
        cam_dict = get_all_leaf_cam_from_method(x.cuda(), net.cuda(), leaf_to_prob,
                                                num_cls, cam, target_layers,
                                                aug_smooth=False,
                                                eigen_smooth=False)


    return decisions, leaf_to_prob, node_to_prob, predicted, cls, decision_to_wnid, cam_dict


def mask_and_inference(decisions, img, cam_dict, model, wnids, args, complex_w = None):
    H, W, _ = img.shape
    decicion_num = len(decisions)
    node_dict = {}
    for ind in range(decicion_num - 1):
        nodes = 'node' + str(ind + 1)
        # 获取该节点子节点概率和叶节点列表
        child_node = decisions[ind]['node'].new_to_old_classes
        child_prob = decisions[ind]['child_prob']

        cam = np.zeros(cam_dict[0][1].shape, dtype=np.float32)

        if complex_w == None:
            for i in range(2):
                w = child_prob[i]
                for node in child_node[i]:
                    cam += w.detach().cpu().numpy() * cam_dict[node][1]
        else:
            w_dict = {}
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
        img2keep[scaled <= args.mask_threshold] = 0  # 保留重要区域
        scaled_ = ma.array(scaled, mask=scaled > args.mask_threshold)
        max, min = scaled_.max(), scaled_.min()
        scaled_ = ((scaled_ - min) / (max - min))
        remove_pixel = scaled_.compressed()
        remove_pixel = remove_pixel[remove_pixel > 0]
        # plt.hist(remove_pixel,bins = 100)
        # plt.show()
        remove_pixel = [np.percentile(remove_pixel, x * 10) for x in range(1, 10)]
        scaled_ = scaled_.filled(0)

        masked_inference = []
        for i, odd in enumerate(remove_pixel):

            # 可以在此处修改遮盖方式xxxxx
            img2remove = img.copy()
            img2remove[scaled_ <= odd] = 0
            img2remove = img2keep + img2remove
            # path_cam_img = os.path.join(output, 'node_' + str(ind + 1), name)
            # cv2.imwrite(path_cam_img, img2remove)

            img2remove = Image.fromarray(img2remove.astype('uint8')).convert('RGB')
            # img1 = Image.open(path_cam_img)
            # # img2 = cv2.imread(os.path.join(node_dir, masked), 1)
            transform1 = get_transform()
            # img_tensor = transform1(img1)
            # x1 = img_tensor.unsqueeze(0)
            x = transform1(img2remove).unsqueeze(0)
            d, leaf_to_prob, n2p, predicted, cls = forword_tree(x.cuda(), model.cuda(), wnids, args.dataset)
            d2w = get_decision_wnid(d[0])
            masked_inference.append([n2p, d2w])
        node_dict[nodes] = masked_inference
    #Colors.cyan("mask inference of {} has been generated".format(type_name))
    return node_dict

def compute_prob_change(mask_dict: dict):
    per_change = []
    for per in range(9):
        abs_change_per = 0
        for i in range(1, len(mask_dict)):
            name = 'node' + str(i)
            abs_change = 0
            for node in mask_dict['no_mask'][0].keys():
                abs_change += np.abs(float(mask_dict['no_mask'][0][node]) - float(mask_dict[name][per][0][node]))
            abs_change = abs_change / len(mask_dict['no_mask'][0].keys()) * (i / (len(mask_dict) - 1))
        abs_change_per += abs_change
        per_change.append(abs_change_per)
    return per_change


def compute_iou_change(mask_dict: dict):
    per_change = []
    for per in range(9):
        iou_change_per = 0
        for i in range(1, len(mask_dict)):
            name = 'node' + str(i)
            iou_change_per += (len(set(mask_dict['no_mask'][1]) & set(mask_dict[name][per][1]))) / (len(set(mask_dict['no_mask'][1]) | set(mask_dict[name][per][1])))
        per_change.append(iou_change_per / (len(mask_dict) - 1))
    return per_change

def weighted_AVG(change_list: list):
    return np.sum(np.array(change_list) * np.arange(0.1, 1, 0.1)) / len(change_list)


def plot_metric_all_class(metric_class, output):
    plt.figure(figsize=(14, 6))
    x = np.array(np.linspace(0.1, 0.9, 9))
    colors = [plt.cm.Paired(i) for i in range(len(metric_class))]

    plt.subplot(1, 2, 1)
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框
    for i, cls in enumerate(metric_class.keys()):
        all_prob = weighted_AVG(metric_class[cls][0])
        plt.plot(x, metric_class[cls][0], marker='o', markersize=10, color=colors[i], label=cls + ": %.4f" % all_prob, linewidth=1.5, alpha=0.9)
    group_labels = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']  # x轴刻度的标识
    plt.xticks(x, group_labels, fontsize=12, fontweight='bold')  # 默认字体大小为10
    plt.yticks(fontsize=12, fontweight='bold')
    # plt.title("example", fontsize=12, fontweight='bold')  # 默认字体大小为12
    plt.xlabel("Pixels removed", fontsize=13, fontweight='bold')
    plt.ylabel("afc", fontsize=12, fontweight='bold')
    plt.xlim(0, 1)  # 设置x轴的范围

    # plt.legend()          #显示各曲线的图例
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=10, fontweight='bold')
    plt.subplot(1, 2, 2)
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框
    for i, cls in enumerate(metric_class.keys()):
        all_iou = weighted_AVG(metric_class[cls][1])
        plt.plot(x, metric_class[cls][1], marker='^', markersize=10, color=colors[i], label=cls + ": %.4f" % all_iou, linewidth=1.5, alpha=0.9)
    group_labels = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']  # x轴刻度的标识
    plt.xticks(x, group_labels, fontsize=12, fontweight='bold')  # 默认字体大小为10
    plt.yticks(fontsize=12, fontweight='bold')
    # plt.title("example", fontsize=12, fontweight='bold')  # 默认字体大小为12
    plt.xlabel("Pixels removed", fontsize=12, fontweight='bold')
    plt.ylabel("iou", fontsize=12, fontweight='bold')

    plt.xlim(0, 1)  # 设置x轴的范围

    # plt.legend()          #显示各曲线的图例
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=10, fontweight='bold')
    plt.savefig(os.path.join(output, 'class_metrics.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output, 'class_metrics.svg'), format='svg', bbox_inches='tight')
    Colors.green("successfully generate the plot of the metric")

def plot_metric_all_tree(metric_class, output, name):
    plt.figure(figsize=(18, 6))
    x = np.array(np.linspace(0.1, 0.9, 9))
    colors = ['red', 'green', 'blue']
    marker = ['o', 'x', '^']

    plt.subplot(1, 2, 1)
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框
    for i, cls in enumerate(metric_class.keys()):
        all_prob = weighted_AVG(metric_class[cls][0])
        plt.plot(x, metric_class[cls][0], marker=marker[i], markersize=10, color=colors[i], label=cls + ": %.4f" % all_prob, linewidth=1.5)
    group_labels = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']  # x轴刻度的标识
    plt.xticks(x, group_labels, fontsize=12, fontweight='bold')  # 默认字体大小为10
    plt.yticks(fontsize=12, fontweight='bold')
    # plt.title("example", fontsize=12, fontweight='bold')  # 默认字体大小为12
    plt.xlabel("Pixels removed", fontsize=13, fontweight='bold')
    plt.ylabel("afc", fontsize=12, fontweight='bold')
    plt.xlim(0, 1)  # 设置x轴的范围

    # plt.legend()          #显示各曲线的图例
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=10, fontweight='bold')
    plt.subplot(1, 2, 2)
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框
    for i, cls in enumerate(metric_class.keys()):
        all_iou = weighted_AVG(metric_class[cls][1])
        plt.plot(x, metric_class[cls][1], marker=marker[i], markersize=10, color=colors[i], label=cls + ": %.4f" % all_iou, linewidth=1.5)
    group_labels = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']  # x轴刻度的标识
    plt.xticks(x, group_labels, fontsize=12, fontweight='bold')  # 默认字体大小为10
    plt.yticks(fontsize=12, fontweight='bold')
    # plt.title("example", fontsize=12, fontweight='bold')  # 默认字体大小为12
    plt.xlabel("Pixels removed", fontsize=12, fontweight='bold')
    plt.ylabel("iou", fontsize=12, fontweight='bold')

    plt.xlim(0, 1)  # 设置x轴的范围

    # plt.legend()          #显示各曲线的图例
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=10, fontweight='bold')
    plt.savefig(os.path.join(output, name + '.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output, name + '.svg'), format='svg', bbox_inches='tight')
    Colors.green("successfully generate the plot of the metric")

def plot_metric(change_prob, change_iou, output):
    all_prob = weighted_AVG(change_prob)
    all_iou = weighted_AVG(change_iou)

    x = np.array(np.linspace(0.1, 0.9, 9))
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框

    plt.plot(x, change_prob, marker='o', markersize=10, color="red", label="afc = %.4f" % all_prob, linewidth=1.5)
    plt.plot(x, change_iou, marker='^', markersize=10, color="blue", label="iou = %.4f" % all_iou, linewidth=1.5)
    group_labels = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']  # x轴刻度的标识
    plt.xticks(x, group_labels, fontsize=12, fontweight='bold')  # 默认字体大小为10
    plt.yticks(fontsize=12, fontweight='bold')
    # plt.title("example", fontsize=12, fontweight='bold')  # 默认字体大小为12
    plt.xlabel("Pixels removed", fontsize=13, fontweight='bold')
    plt.ylabel("afoc / iou", fontsize=13, fontweight='bold')
    plt.xlim(0, 1)  # 设置x轴的范围

    # plt.legend()          #显示各曲线的图例
    plt.legend(loc=6, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=14, fontweight='bold')  # 设置图例字体的大小和粗细

    plt.savefig(os.path.join(output, 'metrics.png'), dpi=300)
    plt.clf()
    # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
    # Colors.green("successfully generate the plot of the metric to {}".format(os.path.join(output, 'metrics.png')))

def plot_metric_single_img(change_prob, change_iou, output):
    plt.figure(figsize=(8, 5))
    all_prob = weighted_AVG(change_prob)
    all_iou = weighted_AVG(change_iou)

    x = np.array(np.linspace(0.1, 0.9, 9))
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框

    plt.plot(x, change_prob, marker='o', markersize=10, color="red", label="afc = %.4f" % all_prob, linewidth=1.5)
    plt.plot(x, change_iou, marker='^', markersize=10, color="blue", label="iou = %.4f" % all_iou, linewidth=1.5)
    group_labels = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']  # x轴刻度的标识
    plt.xticks(x, group_labels, fontsize=12, fontweight='bold')  # 默认字体大小为10
    plt.yticks(fontsize=12, fontweight='bold')
    # plt.title("example", fontsize=12, fontweight='bold')  # 默认字体大小为12
    plt.xlabel("Pixels removed", fontsize=13, fontweight='bold')
    plt.ylabel("afoc / iou", fontsize=13, fontweight='bold')
    plt.xlim(0, 1)  # 设置x轴的范围

    # plt.legend()          #显示各曲线的图例
    plt.legend(loc=6, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=14, fontweight='bold')  # 设置图例字体的大小和粗细

    plt.savefig(os.path.join(output, 'metrics.png'), dpi=300)
###############################################################################

