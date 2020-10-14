import torch
import torch.nn as nn
from . import functional as FC
from pandas._libs import reduction
from typing import *


class MSEOHEMLoss(nn.Module):
    def __init__(self):
        super(MSEOHEMLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="none")

    def avg_loss(self, predicted_image, target_image):
        positive_pixel = (target_image > 0)
        mse_loss = self.mse(predicted_image, target_image)

        positive_loss = torch.masked_select(mse_loss, positive_pixel)
        negative_loss = torch.masked_select(mse_loss, ~positive_pixel)
        sum_positive = int(positive_pixel.sum().data.cpu().item())

        k_factor = 3
        k = sum_positive * k_factor
        num_all = predicted_image.shape[1]
        if k + sum_positive > num_all:
            k = int(num_all - sum_positive)
        if k < 10:
            avg_loss = mse_loss.mean()
        else:
            negative_loss_topk, _ = torch.topk(negative_loss, k)
            avg_loss = positive_loss.mean() + negative_loss_topk.mean()
        return avg_loss

    def forward_once(self, predicted_images, target_images):
        loss = []
        batch_size = predicted_images.shape[0]
        for i in range(batch_size):
            predicted_image = predicted_images[i].view(1, -1)
            target_image = target_images[i].view(1, -1)
            avg_loss = self.avg_loss(predicted_image, target_image)
            loss.append(avg_loss)
        return torch.stack(loss, 0).mean()

    def forward(self, predicted_images, char_gt, aff_gt):
        batch_size, height, width, channels = predicted_images.shape
        predict = predicted_images.permute(3, 0, 1, 2)

        pred_char = predict[0].contiguous().view(
            [batch_size * height * width, 1])
        pred_aff = predict[1].contiguous().view(batch_size * height * width, 1)
        char_gt = char_gt.view([batch_size * height * width, 1])
        aff_gt = aff_gt.view([batch_size * height * width, 1])

        loss_char = self.forward_once(
            pred_char.cpu().float(), char_gt.cpu().float())
        loss_aff = self.forward_once(
            pred_aff.cpu().float(), aff_gt.cpu().float())
        loss = loss_char + loss_aff
        return loss


class Maploss(nn.Module):
    def __init__(self, use_gpu=True):

        super(Maploss, self).__init__()

    def single_image_loss(self, pre_loss, loss_label):
        batch_size = pre_loss.shape[0]
        sum_loss = torch.mean(pre_loss.view(-1)) * 0
        pre_loss = pre_loss.view(batch_size, -1)
        loss_label = loss_label.view(batch_size, -1)
        # internel = batch_size
        for i in range(batch_size):
            average_number: int = 0
            # loss = torch.mean(pre_loss.view(-1)) * 0
            positive_pixel: int = len(
                pre_loss[i][(loss_label[i] >= 0.1)])  # type: Any
            average_number += positive_pixel
            if positive_pixel != 0:
                posi_loss = torch.mean(pre_loss[i][(loss_label[i] >= 0.1)])
                sum_loss += posi_loss
                if len(pre_loss[i][(loss_label[i] < 0.1)]) < 3 * positive_pixel:
                    nega_loss = torch.mean(pre_loss[i][(loss_label[i] < 0.1)])
                    average_number += len(pre_loss[i][(loss_label[i] < 0.1)])
                else:
                    nega_loss = torch.mean(torch.topk(
                        pre_loss[i][(loss_label[i] < 0.1)], 3 * positive_pixel)[0])
                    average_number += 3 * positive_pixel
                sum_loss += nega_loss
            else:
                nega_loss = torch.mean(torch.topk(pre_loss[i], 500)[0])
                average_number += 500
                sum_loss += nega_loss
            # sum_loss += loss/average_number

        return sum_loss

    def forward(self, gh_label, gah_label, p_gh, p_gah, mask):
        gh_label = gh_label
        gah_label = gah_label
        p_gh = p_gh
        p_gah = p_gah
        loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)

        assert p_gh.size() == gh_label.size() and p_gah.size() == gah_label.size()
        loss1 = loss_fn(p_gh, gh_label)
        loss2 = loss_fn(p_gah, gah_label)
        loss_g = torch.mul(loss1, mask)
        loss_a = torch.mul(loss2, mask)

        char_loss = self.single_image_loss(loss_g, gh_label)
        affi_loss = self.single_image_loss(loss_a, gah_label)
        return char_loss / loss_g.shape[0] + affi_loss / loss_a.shape[0]


class MSE_OHEM_Loss(nn.Module):
    def __init__(self):
        super(MSE_OHEM_Loss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="none")

    def forward(self, output_imgs, target_imgs):
        loss_every_sample = []
        batch_size = output_imgs.size(0)
        for i in range(batch_size):
            output_img = output_imgs[i].view(1, -1)
            target_img = target_imgs[i].view(1, -1)
            positive_mask = (target_img > 0)
            sample_loss = self.mse_loss(output_img, target_img)

            positive_loss = torch.masked_select(sample_loss, positive_mask)
            negative_loss = torch.masked_select(sample_loss, ~positive_mask)
            num_positive = int(positive_mask.sum().data.cpu().item())

            k = num_positive * 3

            num_all = output_img.shape[1]
            if k + num_positive > num_all:
                k = int(num_all - num_positive)

            if k < 10:
                avg_sample_loss = sample_loss.mean()
            else:
                negative_loss_topk, _ = torch.topk(negative_loss, k)
                avg_sample_loss = positive_loss.mean() + negative_loss_topk.mean()
            loss_every_sample.append(avg_sample_loss)
        return torch.stack(loss_every_sample, 0).mean()


class OHEMLoss(nn.Module):
    def __init__(self, loss_fn=None, k_ratio=3):
        super(OHEMLoss, self).__init__()
        self.loss_fn = loss_fn
        if loss_fn is None:
            self.loss_fn = nn.MSELoss(reduction='none')
        self.k_ratio = k_ratio

    def forward_once(self, predicts: Any, targets: Any):
        average_loss = []
        batch_size = predicts.size(0)
        for i in range(batch_size):
            predict_flat = predicts[i].view(1, -1)
            target_flat = targets[i].view(1, -1)
            base_loss = self.loss_fn(predict_flat, target_flat.float())

            positive_loss, num_positive = FC.positive_mask_loss(
                base_loss, target_flat, thresh=0)
            negative_loss, num_negative = FC.negative_mask_loss(
                base_loss, target_flat, thresh=0)
            num_all = target_flat.size(0)

            k = num_positive * self.k_ratio
            if num_all < k + num_positive:
                k = int(num_all - num_positive)

            if k < 10:
                avg_loss = base_loss.mean()
            else:
                negative_loss_topk, _ = torch.topk(negative_loss, k)
                avg_loss = positive_loss.mean() + negative_loss_topk.mean()
            average_loss.append(avg_loss)
        return torch.stack(average_loss, 0).mean()

    def forward(self, predicts, region_label, affinity_label):
        region_score, affinity_score = predicts[:,
                                                :, :, 0], predicts[:, :, :, 1]
        region_loss = self.forward_once(region_score, region_label)
        affinity_loss = self.forward_once(affinity_score, affinity_label)
        combined_loss = region_loss + affinity_loss
        return combined_loss


# if __name__ == '__main__':
#     x = torch.FloatTensor([[1, 2], [3, 4]]).view(1, 1, 2, 2)
#     y = torch.FloatTensor([[1.1, 2.1], [3., 4.1]]).view(1, 1, 2, 2)
#     loss_fn = MSEOHEMLoss()
#     loss = loss_fn(y, x)
#     print(loss)
