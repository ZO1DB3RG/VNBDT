import os.path
import time

import numpy as np
from PIL import Image

from pytorch_grad_cam.base_cam import BaseCAM
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt

#hello there

def show_result(size, cam, img_path, epoch, type_name,target_size,save: bool):
    name = type_name + '_' +str(epoch)
    plt.figure(figsize=size)
    plt.title('iter:{}'.format(name))
    plt.axis('off')
    plt.imshow(Image.open(img_path).resize(target_size))
    plt.imshow(cam, cmap='jet', alpha=0.5)
    if save:
        plt.savefig(os.path.join('../CAM_img/DH-28', name+'.jpg'), bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()

class EFC_CAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        #后期需要考虑超参数的值
        super(
            EFC_CAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform)
        self.lr = 1e-4
        self.lmbda = -0.05
        self.beta = -250
        self.epoch = 15

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        return np.mean(grads, axis=(2, 3))

    def saliency(self, weights, activations, target_size):
        saliency_map = (weights * activations).sum(1, keepdim=True)
        # saliency_map = np.maximum(saliency_map, 0)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=target_size, mode='bilinear', align_corners=False)
        saliency_map = saliency_map / saliency_map.max()

        return saliency_map

    def calc_smoothness_loss(self, mask, power=2, border_penalty=0.1):
        ''' For a given image this loss should be more or less invariant to image resize when using power=2...
            let L be the length of a side
            EdgesLength ~ L
            EdgesSharpness ~ 1/L, easy to see if you imagine just a single vertical edge in the whole image'''
        x_loss = torch.sum((torch.abs(mask[:,:,1:,:] - mask[:,:,:-1,:]))**power)
        y_loss = torch.sum((torch.abs(mask[:,:,:,1:] - mask[:,:,:,:-1]))**power)
        if border_penalty>0:
            border = float(border_penalty)*torch.sum(mask[:,:,-1,:]**power + mask[:,:,0,:]**power + mask[:,:,:,-1]**power + mask[:,:,:,0]**power)
        else:
            border = 0.
        return (x_loss + y_loss + border) / float(power * mask.size(0))  # watch out, normalised by the batch size!

    def fowrad_acmp(self, img, retain_graph=True):
        logit = self.activations_and_grads(img)
        # 为DFLCNN做个判断：
        if type(logit) == tuple:
            logit = logit[0] + logit[1] + logit[2] * 0.1

        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        activations = torch.from_numpy(activations_list[0])
        return logit, activations.cuda()

    def compute_sal(self, input_tensor, img_path, type_name,targets):
        logit, cls, weights, weights_relu, activations = self.forward_seperate_part(input_tensor,
                                                                                    targets)
        target_size = self.get_target_width_height(input_tensor)
        predictions = F.softmax(logit, dim = 1)
        print("model prediction of category {}: {:.8f}".format(cls, predictions[0][cls]))
        trainable_weights = weights_relu

        if self.cuda == False:
            trainable_weights = Variable(trainable_weights, requires_grad=True)
        else:
            trainable_weights = Variable(trainable_weights.cuda(), requires_grad=True)
        alpha = 1 - torch.log(predictions[0][cls]).item()

        optimizer = torch.optim.Adam([trainable_weights], lr=self.lr)
        t1 = time.time()
        result_cam = self.saliency(weights_relu, activations, target_size).detach().cpu().numpy()

        for epoch in range(self.epoch):
            cam = self.saliency(trainable_weights, activations, target_size)
            # 展示每一次优化前的解释结果
            # if epoch == 0:
            #     show_result((5,5),cam.squeeze().detach().cpu().numpy(), img_path, epoch, type_name,
            #                 target_size, save=False)

            # show_result((5, 5), cam.squeeze().detach().cpu().numpy(), img_path, epoch, type_name,
            #             target_size, save=False)
            # 计算几个约束项
            # 计算全变分约束
            smooth_loss = self.calc_smoothness_loss(cam, border_penalty=0.1)

            cam_rp = cam.repeat(1, 3, 1, 1).cuda()  # ????
            cam_rp_inverse = torch.ones_like(cam_rp) - cam_rp  # 1 - M
            prod_img = input_tensor * cam_rp.cuda()
            prod_img_inverse = input_tensor * cam_rp_inverse.cuda()
            logit_prod, activations_prod = self.fowrad_acmp(prod_img)  # 获得原图logit和特征图
            logit_prod_inverse, activations_prod_inverse = self.fowrad_acmp(prod_img_inverse)  # 获得inverse的logit和特征图
            predictions_prod = F.softmax(logit_prod, dim=1)[0][cls]
            #防止过小的值

            # 计算总约束
            activations_prod = activations_prod.mean((2, 3))
            activations_prod_inverse = activations_prod_inverse.mean((2, 3))

            score = -alpha * logit_prod[:, cls].squeeze() \
                    + logit_prod_inverse[:, cls].squeeze() \
                    + smooth_loss \
                    + self.beta * (weights_relu * (activations_prod - activations_prod_inverse)).norm() \
                    + self.lmbda * (cam_rp - cam_rp_inverse).norm()

            if torch.any(torch.isnan(smooth_loss)):
                print("Nan happened, stop here")
                break

            # print('SSR:{:.4f}, SDR:{:.4f}, TV:{:.4f}/{:.4f}/{:.4f}, EFC:{:.4f}'.format(
            #     float(-alpha * logit_prod[:, cls].squeeze()),
            #     float(logit_prod_inverse[:, cls].squeeze()),
            #     float(smooth_loss + self.lmbda * (cam_rp - cam_rp_inverse).norm()),
            #     float(smooth_loss),
            #     float(self.lmbda * (cam_rp - cam_rp_inverse).norm()),
            #     float(self.beta * (weights_relu * (activations_prod - activations_prod_inverse)).norm())))

            # 显示优化过程信息，包括当前epcoh，prod_img的预测概率，以及总约束的值
            # print(epoch, predictions_prod.detach().cpu().numpy(), score.detach().cpu().numpy())
            # 保存已有cam

            result_cam = cam.squeeze().detach().cpu().numpy()

            # 执行优化
            optimizer.zero_grad()
            score.backward(retain_graph=True)
            optimizer.step()

        t2 = time.time()
        print('cost time:', t2 - t1)

        # show_result((10, 10), result_cam, img_path, 'final', type_name,
        #             target_size, save=False)

        return result_cam





