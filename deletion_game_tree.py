import os.path
import time
from vnbdt_metric import *
from vnbdt import *
import cv2
from nbdt.model import SoftNBDT
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import shutil
import gc

if __name__ == "__main__":
    cam_methods = ['gradcam', 'gradcam++', 'scorecam', 'efccam']
    cam_method = cam_methods[0]

    tree_methods = ['induced']
    #dataset = 'Imagenet10'
    dataset = 'Fashion10'
    # method = 'induced'
    # arch = 'vgg16'
    # base_path = '/home/mist/checkpoints/vgg16_pretrained.pkl'
    # nbdt_ft_path = '/home/mist/checkpoints/ckpt-Imagenet10-vgg16-lr0.01-SoftTreeSupLoss.pth'

    #arch = 'ResNet50'
    arch = 'DFLCNN'
    # base_path = '/home/mist/checkpoints/resnet50_pretrained3.pkl'
    # nbdt_ft_path = '/home/mist/checkpoints/ckpt-Imagenet10-ResNet50-lr0.01-SoftTreeSupLoss.pth'

    base_path = '/home/lzl001/CNN_train/resnet50_pretrained3.pkl'
    nbdt_ft_path = '/home/lzl001/NBDT/neural-backed-decision-trees/checkpoint/ckpt-Imagenet10-ResNet50-lr0.01-SoftTreeSupLoss_induced.pth'
    pro_pth = '/home/lzl001/NBDT/neural-backed-decision-trees/checkpoint/ckpt-Imagenet10-ResNet50-lr0.01-SoftTreeSupLoss_pro.pth'
    random_pth = '/home/lzl001/NBDT/neural-backed-decision-trees/checkpoint/ckpt-Imagenet10-ResNet50-lr0.01-SoftTreeSupLoss_random.pth'
    # arch = 'vgg19_bn'
    # base_path = '/home/lzl001/CNN_train/model_vgg19bn_pretrained.pkl'
    # nbdt_ft_path = '/home/lzl001/NBDT/neural-backed-decision-trees/checkpoint/ckpt-Imagenet10-vgg19_bn-lr0.01-SoftTreeSupLoss.pth'
    #tar_path = './pretrain_model/model_best_12.pth.tar'
    #exp_img_source_path = '/home/mist/imagenet-10/val'
    tar_path ='/home/lzl001/FGVC/DFL-CNN/weight/epoch_0015_top1_68_checkpoint.pth.tar'

    exp_img_source_path = '/data/LZL/Fashion10/val'

    cls_list = os.listdir(exp_img_source_path)
    # cls_list_1 = ['Blouse','Cardigan','Dress','Jacket','Jeans','Romper','Shorts',
    #              'Sweater', 'Tee']

    # 获取该数据集下所有类别的对应ID
    wnids = get_wnids_from_dataset(dataset)
    num_cls = len(wnids)
    t = 0
    issue_img = []
    for method in tree_methods:
        Colors.bold(method + ' has started')
        if method == 'pro':#还可以优化
            net = call_pth_model(arch, pro_pth, cls_num=num_cls, device='cuda') #可改

            # 生成树并读取
            G, path = get_pro_tree(dataset=dataset, arch=arch, method=method)
            # 验证树及其节点对应，并返回根节点Node
            root = validate_tree(G, path, wnids)
            model = SoftNBDT(
                pretrained=False,
                dataset=dataset,
                path_graph=path,
                model=net,
                classes=wnids
            )
        elif method == 'random':#还可以优化
            net = call_pth_model(arch, random_pth, cls_num=num_cls,device='cuda') #可改

            # 生成树并读取
            G, path = get_pro_tree(dataset=dataset, arch=arch, method=method)
            # 验证树及其节点对应，并返回根节点Node
            root = validate_tree(G, path, wnids)
            model = SoftNBDT(
                pretrained=False,
                dataset=dataset,
                path_graph=path,
                model=net,
                classes=wnids
            )
        elif method == 'base':
            net = call_pkl_model(arch, base_path, cls_num=num_cls, device='cuda')
            G, path = get_tree(dataset=dataset, arch=arch, model=net, method='induced') #预测与树结构无关，但需要以induced为对比
            # 验证树及其节点对应，并返回根节点Node
            root = validate_tree(G, path, wnids)
            model = SoftNBDT(
                pretrained=False,
                dataset=dataset,
                arch=arch,
                model=net,
                classes=wnids
            )
        elif dataset == 'Fashion10' and method == 'induced':
            net = calling_DNLCNN_from_tar(tar_path, cls_num=10)
            G, path = get_tree(dataset=dataset, arch=arch, model=net, method='induced')
            # 验证树及其节点对应，并返回根节点Node
            root = validate_tree(G, path, wnids)
            model = SoftNBDT(
                pretrained=False,
                dataset=dataset,
                arch=arch,
                model=net,
                classes=wnids
            )
        else:
            net = call_pth_model(arch, nbdt_ft_path, cls_num=num_cls, device='cuda')
            G, path = get_tree(dataset=dataset, arch=arch, model=net, method='induced')
            # 验证树及其节点对应，并返回根节点Node
            root = validate_tree(G, path, wnids)
            model = SoftNBDT(
                pretrained=False,
                dataset=dataset,
                arch=arch,
                model=net,
                classes=wnids
            )

        Colors.red(method + ' has been successfully reloaded')

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

        output_dir = os.path.join('/data/LZL/Fashion-epoch/complex', method)
        t1 = time.time()
        # 对每一个数据类别文件夹进行处理
        # if method == 'induced':
        #     l = cls_list_1
        # else:
        #     l = cls_list
        # print('all cls:')
        # print(l)

        for cls in list(reversed(cls_list)):
            cls_out_dir = os.path.join(output_dir, cls)
            if not os.path.exists(cls_out_dir):
                os.makedirs(cls_out_dir)
            img_list = os.listdir(os.path.join(exp_img_source_path, cls))[-25:]
            for img in img_list:
                path_img = os.path.join(os.path.join(exp_img_source_path, cls), img)
                img_name = os.path.split(path_img)[1].split('.')[0]
                Colors.green("============= " + cls + " : " + img_name + " starts =============")
                masked_dir = os.path.join(cls_out_dir, img_name)

                if not os.path.exists(masked_dir):
                    os.makedirs(masked_dir)
                # 数据处理,生成遮掩
                img1 = Image.open(path_img)
                img2 = cv2.imread(path_img, 1)
                transform1 = get_transform()
                img_tensor = transform1(img1)
                x = img_tensor.unsqueeze(0)

                # 生成遮掩保存
                if method in ['pro','random','induced','base']:
                    gc.collect()
                    torch.cuda.empty_cache()
                    decisions, leaf_to_prob, node_to_prob, predicted = forword_tree(x.cuda(), model.cuda(), wnids,
                                                                                    dataset)
                    decision_to_wnid = get_decision_wnid(decisions[0])
                    record_node_prob(node_to_prob, decision_to_wnid,
                                     os.path.join(output_dir, method + '_mask_record.txt'), cls + '+' + img_name)
                    if cam_method != 'efccam':
                        cam_dict = get_all_leaf_cam_from_method(x, net, leaf_to_prob,
                                                                num_cls, cam_method, target_layers,
                                                                aug_smooth=False,
                                                                eigen_smooth=False)
                    else:
                        cam_dict = get_all_leaf_cam_efc(x.cuda(), net.cuda(), leaf_to_prob, num_cls,
                                                        cam_method, target_layers, '', '')

                    complex_w = compute_complex_weight(cam_dict, predicted)
                    #生成决策链路所有节点显著图的遮掩图
                    generate_cam_mask_with_simple_w(decisions[0], img2, img_name, cam_dict, cls_out_dir)
                    generate_cam_mask_with_complex_w(decisions[0], img2, img_name, cam_dict, cls_out_dir, complex_w, predicted)

                    del cam_dict

                    for node_ in range(1, len(decisions[0])):
                        node_dir = os.path.join(masked_dir, 'node_' + str(node_))
                        # 根据遮掩保存路径读取图像进行预测，写入预测数据
                        for masked in os.listdir(node_dir):
                            img1 = Image.open(os.path.join(node_dir, masked))
                            #img2 = cv2.imread(os.path.join(node_dir, masked), 1)
                            transform1 = get_transform()
                            img_tensor = transform1(img1)
                            x = img_tensor.unsqueeze(0)
                            # img2 = cv2.resize(cv2.imread(os.path.join(node_dir, masked), 1), (224, 224))[:, :, ::-1]
                            # img2 = np.float32(img2) / 255
                            # x = preprocess_image(img2,
                            #                      mean=[0.5, 0.5, 0.5],
                            #                      std=[0.5, 0.5, 0.5])
                            mask_name = os.path.split(masked)[1].split('.')[0]
                            gc.collect()
                            torch.cuda.empty_cache()
                            decisions, leaf_to_prob, n2p, predicted = forword_tree(x.cuda(), model.cuda(), wnids, dataset)
                            d2w = get_decision_wnid(decisions[0])
                            record_node_prob(n2p, d2w,
                                             os.path.join(output_dir, method + '_mask_record.txt'),
                                             cls + '+' + mask_name)
                            # Colors.cyan("== " + mask_name + " finished ==")

                    Colors.green("============== " + cls + " - " + img_name + " ends ==============")

                else:
                    decisions, leaf_to_prob, node_to_prob, predicted = forword_tree_no(x.cuda(), model.cuda(),
                                                                                       wnids,dataset)
                    decision_to_wnid = get_decision_wnid(decisions[0])
                    record_node_prob(node_to_prob, decision_to_wnid,
                                     os.path.join(output_dir, method + '_mask_record.txt'), cls + '+' + img_name)
                    if cam_method != 'efccam':
                        cam = get_cam_from_method(x.cuda(), model.model.cuda(), predicted,
                                                  cam_method, target_layers,False,False)
                    else:
                        cam = get_cam_efc(x.cuda(), model.cuda(),target_layers,'','')

                    generate_cam_mask_one_sample(img2, img_name, cam, cls_out_dir)

                    # 根据遮掩保存路径读取图像进行预测，写入预测数据
                    for masked in os.listdir(masked_dir):
                        img2 = cv2.imread(os.path.join(masked_dir, masked), 1)
                        img2 = np.float32(img2) / 255
                        x = preprocess_image(img2,
                                             mean=[0.5, 0.5, 0.5],
                                             std=[0.5, 0.5, 0.5])
                        mask_name = os.path.split(masked)[1].split('.')[0]
                        decisions, leaf_to_prob, n2p, predicted = forword_tree_no(x.cuda(), model.cuda(), wnids, dataset)
                        d2w = get_decision_wnid(decisions[0])
                        record_node_prob(n2p, d2w,
                                         os.path.join(output_dir, method + '_mask_record.txt'),
                                         cls + '+' + mask_name)
                    Colors.green("============== " + cls + " - " + img_name + " ends ==============")

                shutil.rmtree(masked_dir)

            Colors.cyan("============== " + cls + " ends ==============")

        t2 = time.time()
        t += (t2 - t1)
        Colors.red("{} cost time: {:.4f}".format(method, t2 - t1))
    for i in issue_img:
        print(i)
    Colors.bold(
        "total time: {}".format(t)
    )

