#

#可以按照下面这个array的格式替换为目标数据集的类别，不要忘记修改
array=("Arctic_fox" "Gordon_setter" "Ibizan_hound" "Saluki" "Tibetan_mastiff" "goose" "house_finch" "robin" "toucan" "white_wolf")
for i in "${array[@]}"
do
  python vnbdt_quant.py --dataset Imagenet10 --pth_path /home/lzl001/NBDT/neural-backed-decision-trees/checkpoint/ckpt-Imagenet10-ResNet50-lr0.01-SoftTreeSupLoss_induced.pth --merge simple --img_dir /data/LZL/imagenet-10/test/$i --output_dir /data/LZL/imagenet-10/test/{$i}_cam --html_output /data/LZL/imagenet-10/test/{$i}_html --device 2
  python vnbdt_quant.py --dataset Imagenet10 --pth_path /home/lzl001/NBDT/neural-backed-decision-trees/checkpoint/ckpt-Imagenet10-ResNet50-lr0.01-SoftTreeSupLoss_pro.pth --merge simple --img_dir /data/LZL/imagenet-10/test/$i --output_dir /data/LZL/imagenet-10/test/{$i}_cam_pro --html_output /data/LZL/imagenet-10/test/{$i}_html_pro --method pro --device 2
  python vnbdt_quant.py --dataset Imagenet10 --pth_path /home/lzl001/NBDT/neural-backed-decision-trees/checkpoint/ckpt-Imagenet10-ResNet50-lr0.01-SoftTreeSupLoss_random.pth --merge simple --img_dir /data/LZL/imagenet-10/test/$i --output_dir /data/LZL/imagenet-10/test/{$i}_cam_random --html_output /data/LZL/imagenet-10/test/{$i}_html_random --method random --device 2
done