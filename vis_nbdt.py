import argparse
from vnbdt import *
from nbdt.utils import DATASET_TO_NUM_CLASSES

# 定义获取图的函数

parser = argparse.ArgumentParser()
parser.add_argument('--cam', default='gradcam', type=str, help='class activate mapping methods')
parser.add_argument('--dataset', default='Imagenet10', type=str, help='dataset name')
parser.add_argument('--arch', default='ResNet50', type=str, help='name of the architecture')
parser.add_argument('--method', default='induced', type=str, help='tree type, others are pro or random')
parser.add_argument('--pth_path', default='/home/lzl001/NBDT/neural-backed-decision-trees/checkpoint/ckpt-Imagenet10-ResNet50-lr0.01-SoftTreeSupLoss_induced.pth', type=str, help='class activate mapping methods')
parser.add_argument('--merge', default='simple', type=str, help='the way to merge the cam')

parser.add_argument('--img_dir', default="/data/LZL/imagenet-10/test/Ibizan_hound/n0209124400000864.jpg", type=str, help='image folder waiting infered and explained')
parser.add_argument('--output_dir', default='out', type=str, help='Store CAM')
parser.add_argument('--html_output', default='html', type=str, help='Store html, should be the same father with output_dir')

parser.add_argument('--name', default='', type=str, help='something you want your file name to be')
parser.add_argument('--device', default='3', type=str, help='device')

if __name__ == '__main__':

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # 获取该数据集下所有类别的对应ID
    wnids = get_wnids_from_dataset(args.dataset)
    num_cls = len(wnids)

    # 获取网络
    if os.path.splitext(args.pth_path)[-1].find('pth') == 1 and os.path.splitext(args.pth_path)[-1].find('tar') != 1:
        net = call_pth_model(args.arch, args.pth_path, cls_num=DATASET_TO_NUM_CLASSES[args.dataset])
    elif os.path.splitext(args.pth_path)[-1].find('pkl') == 1:
        net = call_pkl_model(args.arch, args.pth_path, cls_num=DATASET_TO_NUM_CLASSES[args.dataset])
    elif os.path.splitext(args.pth_path)[-1].find('tar') == 1:
        net = calling_DNLCNN_from_tar(args.pth_path, cls_num=DATASET_TO_NUM_CLASSES[args.dataset])
    #输入网络权重获取NBDT树结构，延续原本的代码操作
    if args.method == 'induced':
        G, path = get_tree(dataset=args.dataset, arch=args.arch, model=net, method=args.method)
    #也可以生成固定结构的专家树或者随机树，但要保证nbdt/hierarcies/dataset 文件夹下已存在对应的结构json文件
    else:
        G, path = get_pro_tree(dataset=args.dataset, arch=args.arch, method=args.method)

    # 验证树及其节点对应，并返回根节点Node
    # 该过程打印的信息若不需要可以删除
    root = validate_tree(G, path, wnids)

    """
    由于visualization依托于HTML页面，因此在生成树结构和热力图以后对其保存路径要求比较严格   
    文件夹设置：
        1.samples/img_xxx: 保存需要解释的图像集（文件夹）img_xxx在sample中，
                           此处代码假设img_xxx下没有子文件夹，全是图像
                           如果测试图像不在项目文件夹下，就会出现html页面根节点不显示原图
            img_1.jpg/png ---> 为了更好地展示结果，这里的图片名称最好与原本的类别对应，比如 mastiff1.jpg
            img_2.jpg/png
            ......
        
        2.xx_output: 运行代码后保存CAM解释生成的图像，每一个图像都会生成一个文件夹，文件夹下保存链路所有节点的图像
                     且这个生成的文件夹名称直接体现CAM方法和融合方法
            img_1_[CAM_METHODS]_[MERGE_METHODS]：
                img_1_[CAM_METHODS]_[MERGE_METHODS]_1.jpg
                img_1_[CAM_METHODS]_[MERGE_METHODS]_2.jpg
                ......
            img_2_[CAM_METHODS]_[MERGE_METHODS]：
                img_2_[CAM_METHODS]_[MERGE_METHODS]_1.jpg
                img_2_[CAM_METHODS]_[MERGE_METHODS]_2.jpg
                ......
    
        3.xx_html: 运行代码以后生成html文件，打开即可监视解释效果并互动
            xxx1.html
            xxx2.html
            ......
    """

    #为html页面及其存储图片创建目录
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
        """
        efccam的热力图要缩小到等高宽
        size = None ---> 用原图大小，更多细节
        size = (224, 224) ---> 实验大小，供选择
        """
        for img in path_list:
            if args.method != 'pro' and args.method != 'random':
                generate_html(G, root, args.arch, args.dataset, args.cam, img, net, wnids, num_cls,
                              args.output_dir, args.html_output, (448,448), 's', args.merge, ori_cls)
                # generate_html(G, root, arch, dataset, cam_method, img, net, wnids, num_cls,
                #               output_dir, html_output, (448,448),'w', 'w')
                # generate_html(G, root, arch, dataset, cam_method, img, net, wnids, num_cls,
                #               output_dir, html_output, (448,448),'c', 'complex')
            else:
                generate_pro_html(G, root, args.method, path, args.arch, args.dataset, args.cam, img, net, wnids, num_cls,
                                  args.output_dir, args.html_output, (448,448),'s1', args.merge, ori_cls)
    else:
        ori_cls = os.path.split(args.img_dir)[-2]
        if args.method != 'pro' and args.method != 'random':
            generate_html(G, root, args.arch, args.dataset, args.cam, args.img_dir, net, wnids, num_cls,
                          args.output_dir, args.html_output, (448,448), 's', args.merge, ori_cls)
            # generate_html(G, root, arch, dataset, cam_method, img, net, wnids, num_cls,
            #               output_dir, html_output, (448,448),'w', 'w')
            # generate_html(G, root, arch, dataset, cam_method, img, net, wnids, num_cls,
            #               output_dir, html_output, (448,448),'c', 'complex')
        else:
            generate_pro_html(G, root, args.method, path, args.arch, args.dataset, args.cam, args.img_dir, net, wnids, num_cls,
                              args.output_dir, args.html_output, (448,448),'s1', args.merge, ori_cls)