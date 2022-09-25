# VNBDT
## **An explainable method of visualized NBDT based on multi-class fusion**
***
<div align=center><img  src="https://github.com/ZO1DB3RG/VNBDT/blob/main/img/VNBDT.png"/></div>

***

#### 主代码请浏览 [**visualize_nbdt.py**](https://github.com/ZO1DB3RG/VNBDT/blob/main/visualize_nbdt.py)

#### 大部分函数在 [**vnbdt.py**](https://github.com/ZO1DB3RG/VNBDT/blob/main/vnbdt.py)

**可直接执行**：

    python visualize_nbdt.py
***
## **可选择项**：
### **网络结构**：
> 可以增加你自己的模型: 于[**vnbdt.py**](https://github.com/ZO1DB3RG/VNBDT/blob/main/vnbdt.py)文件的_call-xxx-model()_与_get-layer()_函数中增加对应的模型结构及所需映射的网络层，于[**nbdt/graph.py**](https://github.com/ZO1DB3RG/VNBDT/blob/main/nbdt/graph.py)中的***MODEL_FC_KEYS***增加对应的FC层名称
>> ResNet50, ResNet18

>> vgg16

>> wrn28_10_cifar10

>> 细粒度模型[DFLCNN](https://github.com/songdejia/DFL-CNN)
***

### **已定义完成的数据集**
>如果需要扩展自己想要的数据集，首先需要在[**nbdt/hierarchies**](https://github.com/ZO1DB3RG/VNBDT/tree/main/nbdt/hierarchies)中创建空文件夹（数据集名如Emo），在[**nbdt/wnids**](https://github.com/ZO1DB3RG/VNBDT/tree/main/nbdt/wnids)中创建
    对应名称的txt文件（如Emo.txt）并输入对应分类数量的ID，最后在[**nbdt/utils.py**](https://github.com/ZO1DB3RG/VNBDT/blob/main/nbdt/utils.py)中增加对应数据集的各类信息，才能保证
    代码能在新数据集上运行
>>CIFAR10,
>>时装数据集10子类 [Fashion10](https://github.com/switchablenorms/DeepFashion2)

>> PARA美学数据的情感分类数据集 [Emo](https://cv-datasets.institutecv.com/#/data-sets)

>> 'FGVC', 'FGVC12','FGVC10': FGVC飞行器数据集的子类

>> 'Imagenet10': imagenet10子类

***
### **可选择的类别激活映射方法：**
>>"gradcam": GradCAM,

>>"scorecam": ScoreCAM,

>>"gradcam++": GradCAMPlusPlus,

>> "ablationcam": AblationCAM,

>> "xgradcam": XGradCAM,

>>"eigencam": EigenCAM,

>>"eigengradcam": EigenGradCAM,

>>"layercam": LayerCAM,

>>"fullgrad": FullGrad,

>>'efccam': [EFC_CAM](https://ieeexplore.ieee.org/document/9405672/)

***
### **可选择的内部节点热力图生成方法：**
>> Decision-probability-oriented fusion of multi-CAM

>> Similarity-attention-oriented fusion of multi-CAM

>> Without any weight

### **可选择的树结构**
> 专家树需要重新在[**nbdt/hierarcies/dataset**](https://github.com/ZO1DB3RG/VNBDT/tree/main/nbdt/hierarchies/Emo) 文件夹下重新定义一个固定的专家树结构json文件，详情邮件咨询：li_zhili0105@163.com
>> induced: 诱导树

>> random: 随机树

>> pro: 专家树（结构固定）

***
**效果展示**
<div align=center><img width="500", height="400" src="https://github.com/ZO1DB3RG/VNBDT/blob/main/img/SAOVNBDT.png"/></div>
<div align=center><img width="300" height="300" src="https://github.com/ZO1DB3RG/VNBDT/blob/main/img/emo1.gif"/></div>

## 注意事项
***
由于visualization依托于HTML页面，因此在生成树结构和热力图以后对其保存路径有比较严格的文件夹设置要求：
    
    1.samples/img_xxx: 保存需要解释的图像集（文件夹）img_xxx在sample中，此处代码假设img_xxx下没有子文件夹，全是图像
                           如果测试图像不在项目文件夹下，就会出现html页面根节点不显示原图
            img_1.jpg/png ---> 为了更好地展示结果，这里的图片名称最好与原本的类别对应，比如 mastiff1.jpg
            img_2.jpg/png
            ......
     
    2.xx_output: 运行代码后保存CAM解释生成的图像，每一个图像都会生成一个文件夹，文件夹下保存链路所有节点的图像，这个生成的文件夹名称直接体现CAM方法和融合方法
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

***
依赖环境：在[requirements.txt](https://github.com/ZO1DB3RG/VNBDT/blob/main/requirements.txt) 中

    detail==0.2.2
    matplotlib==3.3.4
    networkx==2.5.1
    nltk==3.6.7
    numpy==1.19.5
    opencv_python==4.4.0.42
    Pillow==9.2.0
    pytorchcv==0.0.67
    scikit_learn==1.1.2
    torch==1.10.1
    torchvision==0.11.2
    tqdm==4.64.0
    ttach==0.0.3
