from collections import namedtuple
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tracker.backbone import AlexNetV1ResBoF24
from utils.track_utils import pad_frame, extract_crops_x, extract_crops_z, image_to_tensor, resize_scoremap

TrackerConfig = namedtuple("TrackerConfig", ["scale_step", "scale_num", "scale_penalty", "scale_lr", "response_up",
                                             "context_amount", "max_scale_factor", "min_scale_factor",
                                             "window_influence", "z_lr", "template_size", "search_size"])
default_config = TrackerConfig(scale_step=1.05, scale_num=3, scale_penalty=0.98, scale_lr=0.64, window_influence=0.25,
                               z_lr=1e-3, max_scale_factor=5, min_scale_factor=0.2, context_amount=0.5, response_up=16,
                               template_size=127, search_size=255)


class DCHT(nn.Module):
    def __init__(self, config=default_config):
        super(DCHT, self).__init__()
        branch = AlexNetV1ResBoF24()
        self.model = SiameseNet(branch)
        self.tracker_config = config

        # tracker state
        self.scale_factors = self.tracker_config.scale_step ** (
                np.arange(self.tracker_config.scale_num) - self.tracker_config.scale_num // 2)
        self.scale_penalties = self.tracker_config.scale_penalty ** (
            np.abs((np.arange(self.tracker_config.scale_num) - self.tracker_config.scale_num // 2)))
        self.final_score_sz = None
        self.penalty = None
        self.center = None
        self.target_sz = None
        self.target_model = None
        self.avg_color = None
        self.max_x_sz = None
        self.min_x_sz = None
        self.z_sz = None
        self.x_sz = None

        z = torch.ones(1, 3, int(config.template_size), int(config.template_size))
        x = torch.ones(1, 3, int(config.search_size), int(config.search_size))
        self.output_size, self.total_stride = self._deduce_network_params(z, x)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_z, input_x):
        out, _, _ = self.model(input_z, input_x)
        return out

    def initialize(self, weights_path):
        state_dict = torch.load(weights_path)
        self.model.load_state_dict(state_dict)

    def get_params(self, initial_lr, weight_decay):
        params = []
        for name, param in self.model.named_parameters():
            lr = initial_lr
            if 'bof' in name:
                lr *= 2.0
            if '.0' in name:  # conv
                if 'weight' in name:
                    lr *= 1
                    weight_decay *= 1
                elif 'bias' in name:
                    lr *= 2
                    weight_decay *= 0
            elif '.1' in name or 'bn' in name:  # bn
                if 'weight' in name:
                    lr *= 2
                    weight_decay *= 0
                elif 'bias' in name:
                    lr *= 1
                    weight_decay *= 0
            elif 'linear' in name:
                if 'weight' in name:
                    lr *= 1
                    weight_decay *= 1
                elif 'bias' in name:
                    lr *= 1
                    weight_decay *= 0
            params.append({
                'params': param,
                'initial_lr': lr,
                'weight_decay': weight_decay,
                'name': name})
        return params

    def get_finetuning_params(self, finetune_lr, weight_decay):
        params = []
        for name, param in self.model.named_parameters():
            if 'bof' in name or 'join' in name or 'norm' in name:
                lr = finetune_lr
                params.append({
                    'params': param,
                    'initial_lr': lr,
                    'weight_decay': weight_decay,
                    'name': name})
        return params

    def _deduce_network_params(self, z, x):
        with torch.no_grad():
            self.model.eval()
            y, x, z = self.model(z, x)
        score_sz = y.size(-1)

        total_stride = 1
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.MaxPool2d)):
                stride = m.stride[0] if isinstance(
                    m.stride, tuple) else m.stride
                total_stride *= stride

        self.final_score_sz = self.tracker_config.response_up * score_sz  # + 1
        self.penalty = get_hann_window(self.final_score_sz)
        return score_sz, total_stride

    def init(self, im, init_rect):
        # im: cv2  RGB image (HWC)
        # init_rect : (cx, cy), (w, h)
        im_sz = im.shape

        self.center = np.asarray(init_rect[0])
        self.target_sz = np.asarray(init_rect[1])

        mean_dim = self.tracker_config.context_amount * (np.sum(self.target_sz))
        wc_z = self.target_sz[0] + mean_dim
        hc_z = self.target_sz[1] + mean_dim
        self.z_sz = np.sqrt(wc_z * hc_z)

        x_sz = float(self.tracker_config.search_size) / self.tracker_config.template_size * self.z_sz
        self.min_x_sz = self.tracker_config.min_scale_factor * x_sz
        self.max_x_sz = self.tracker_config.max_scale_factor * x_sz

        self.avg_color = np.mean(im, axis=(0, 1))

        scaled_search_area = np.asarray(x_sz) * self.scale_factors
        im_padded_x, npad_x = pad_frame(im, im_sz, self.center[0], self.center[1], scaled_search_area[2],
                                        self.avg_color)
        x_crops = extract_crops_x(im_padded_x, npad_x, self.center[0], self.center[1], scaled_search_area[0],
                                  scaled_search_area[1], scaled_search_area[2], self.tracker_config.search_size)

        size = (self.tracker_config.search_size - 1, self.tracker_config.search_size - 1)
        c = int(size[0] / 2)
        s = int(np.floor(self.tracker_config.template_size / 2))
        z_crop = x_crops[1, c - s: c + s + 1, c - s: c + s + 1]
        z_crops_tensor = image_to_tensor(z_crop).unsqueeze_(0).to(self.device)
        self.target_model = self.model.branch(z_crops_tensor)

        bbox_rect = np.concatenate([self.center - self.target_sz / 2, self.target_sz])
        bbox_rect[0] = max(0, bbox_rect[0])
        bbox_rect[1] = max(0, bbox_rect[1])
        bbox_rect[2] = max(1, min(im.shape[1] - bbox_rect[0], bbox_rect[2]))
        bbox_rect[3] = max(1, min(im.shape[0] - bbox_rect[1], bbox_rect[3]))
        return bbox_rect

    def update(self, im):
        im_sz = im.shape
        self.x_sz = float(self.tracker_config.search_size) / self.tracker_config.template_size * self.z_sz
        scaled_exemplar = np.asarray(self.z_sz) * self.scale_factors
        scaled_target = self.scale_factors[:, np.newaxis] * self.target_sz
        scaled_instance = np.asarray(self.x_sz) * self.scale_factors

        im_padded_x, npad_x = pad_frame(im, im_sz, self.center[0], self.center[1], scaled_instance[2],
                                        self.avg_color)
        x_crops = extract_crops_x(im_padded_x, npad_x, self.center[0], self.center[1], scaled_instance[0],
                                  scaled_instance[1], scaled_instance[2], self.tracker_config.search_size)
        x_crops = torch.stack([image_to_tensor(x_crop) for x_crop in x_crops]).to(self.device)

        with torch.no_grad():
            x_crops_feats = self.model.branch(x_crops)
            score, best_scale = self.best_score(self.model.calc_score(self.target_model, x_crops_feats))

        # update size
        self.x_sz = (1 - self.tracker_config.scale_lr) * self.x_sz + \
                    self.tracker_config.scale_lr * scaled_instance[best_scale]
        self.x_sz = np.clip(self.x_sz, self.min_x_sz, self.max_x_sz)

        self.center = self._locate_target(self.center, score, self.final_score_sz,
                                          self.tracker_config.search_size,
                                          self.tracker_config.response_up, self.x_sz)
        self.target_sz = (1 - self.tracker_config.scale_lr) * self.target_sz + \
                         self.tracker_config.scale_lr * scaled_target[best_scale]

        # update target model if needed
        if self.tracker_config.z_lr > 0:
            im_padded_z, npad_z = pad_frame(im, im_sz, self.center[0], self.center[1], self.z_sz, self.avg_color)
            z_crop = extract_crops_z(im_padded_z, npad_z, self.center[0], self.center[1], self.z_sz,
                                     self.tracker_config.template_size)
            z_crop = image_to_tensor(z_crop).unsqueeze_(0).to(self.device)
            with torch.no_grad():
                self.model.branch.eval()
                new_z = self.model.branch(z_crop)
            self.target_model = (1 - self.tracker_config.z_lr) * self.target_model + \
                                self.tracker_config.z_lr * new_z
        # update exemplar size
        self.z_sz = (1 - self.tracker_config.scale_lr) * self.z_sz + \
                    self.tracker_config.scale_lr * scaled_exemplar[best_scale]

        bbox_rect = np.concatenate([self.center - self.target_sz / 2, self.target_sz])
        bbox_rect[0] = max(0, bbox_rect[0])
        bbox_rect[1] = max(0, bbox_rect[1])
        bbox_rect[2] = max(1, min(im.shape[1] - bbox_rect[0], bbox_rect[2]))
        bbox_rect[3] = max(1, min(im.shape[0] - bbox_rect[1], bbox_rect[3]))

        return bbox_rect

    def _locate_target(self, center, score, final_score_sz,
                       search_sz, response_up, x_sz):
        pos = np.unravel_index(score.argmax(), score.shape)[::-1]
        half = (final_score_sz - 1) / 2

        disp_in_area = np.asarray(pos) - half
        disp_in_xcrop = disp_in_area * self.total_stride / response_up
        disp_in_frame = disp_in_xcrop * x_sz / search_sz

        center = center + disp_in_frame
        return center

    def best_score(self, scores):
        scores[:1] *= self.tracker_config.scale_penalty
        scores[2:] *= self.tracker_config.scale_penalty
        scale_id = scores.view(self.tracker_config.scale_num, -1).max(dim=1)[0].argmax()
        score = scores[scale_id].unsqueeze(0)
        score = resize_scoremap(score.cpu().numpy(), self.final_score_sz)
        score -= score.min()
        score /= max(1e-12, score.sum())
        score = (1 - self.tracker_config.window_influence) * score + self.tracker_config.window_influence * self.penalty
        return score, scale_id


class SiameseNet(nn.Module):
    def __init__(self, branch):
        super(SiameseNet, self).__init__()
        self.branch = branch
        self.join = BhattacharyyaCoeff(self.branch.output_dim)
        self.norm = Adjust2d()

    def forward(self, z, x):
        x = self.branch(x)
        z = self.branch(z)
        out = self.join(z, x)
        out = self.norm(out)
        return out, x, z

    def calc_score(self, z, x):
        with torch.no_grad():
            out = self.join(z.repeat(3, 1, 1, 1), x)
            out = self.norm(out)
        return out


class Adjust2d(nn.Module):
    def __init__(self):
        super(Adjust2d, self).__init__()
        self.type = 'bn'
        self.bn = nn.BatchNorm2d(1)
        self._initialize_weights()

    def forward(self, input):
        out = self.bn(input)
        return out

    def _initialize_weights(self):
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()


class BhattacharyyaCoeff(nn.Module):
    def __init__(self, input_dim=256):
        super(BhattacharyyaCoeff, self).__init__()
        self.weights = nn.Parameter(torch.ones(1, input_dim, 1, 1, 1))

    def forward(self, z, x):
        N = z.size(0)
        C = z.size(1)
        k = z.size(2)
        m = x.size(2)
        N_b = (m - k + 1) ** 2
        x_unf = F.unfold(x, k).view(N, C, k, k, N_b)

        coeff = torch.sum(torch.sqrt(x_unf * z.unsqueeze_(4).repeat(1, 1, 1, 1, N_b)) * self.weights,
                          dim=1, keepdim=True)
        mean_coeff = coeff.mean(dim=2).mean(dim=2).view(N, 1, (m - k + 1), (m - k + 1))
        return mean_coeff


def get_hann_window(size):
    hann_1d = torch.hann_window(size)
    window = torch.mm(hann_1d.view(-1, 1), hann_1d.view(1, -1))
    window = window / window.sum()
    return window.numpy()
