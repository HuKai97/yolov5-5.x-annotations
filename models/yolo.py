"""YOLOv5-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse            # 解析命令行参数模块
import logging             # 日志模块
import sys                 # sys系统模块 包含了与Python解释器和它的环境有关的函数
from copy import deepcopy  # 数据拷贝模块 深拷贝
from pathlib import Path   # Path将str转换为Path对象 使字符串路径易于操作的模块

FILE = Path(__file__).absolute()  # FILE = WindowsPath 'F:\yolo_v5\yolov5-U\modles\yolo.py'
# 将'F:/yolo_v5/yolov5-U'加入系统的环境变量  该脚本结束后失效
sys.path.append(FILE.parents[1].as_posix())  # add yolov5/ to path

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.plots import feature_visualization
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, \
                              scale_img, initialize_weights, select_device, copy_attr

# 导入thop包 用于计算FLOPs
try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None
# 初始化日志
logger = logging.getLogger(__name__)


def parse_model(d, ch):  # model_dict, input_channels(3)
    """用在上面Model模块中
    解析模型文件(字典形式)，并搭建网络结构
    这个函数其实主要做的就是: 更新当前层的args（参数）,计算c2（当前层的输出channel） =>
                          使用当前层的参数搭建当前层 =>
                          生成 layers + save
    :params d: model_dict 模型文件 字典形式 {dict:7}  yolov5s.yaml中的6个元素 + ch
    :params ch: 记录模型每一层的输出channel 初始ch=[3] 后面会删除
    :return nn.Sequential(*layers): 网络的每一层的层结构
    :return sorted(save): 把所有层结构中from不是-1的值记下 并排序 [4, 6, 10, 14, 17, 20, 23]
    """
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    # 读取d字典中的anchors和parameters(nc、depth_multiple、width_multiple)
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    # na: number of anchors 每一个predict head上的anchor数 = 3
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    # no: number of outputs 每一个predict head层的输出channel = anchors * (classes + 5) = 75(VOC)
    no = na * (nc + 5)

    # 开始搭建网络
    # layers: 保存每一层的层结构
    # save: 记录下所有层结构中from中不是-1的层结构序号
    # c2: 保存当前层的输出channel
    layers, save, c2 = [], [], ch[-1]
    # from(当前层输入来自哪些层), number(当前层次数 初定), module(当前层类别), args(当前层类参数 初定)
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # 遍历backbone和head的每一层
        # eval(string) 得到当前层的真实类名 例如: m= Focus -> <class 'models.common.Focus'>
        m = eval(m) if isinstance(m, str) else m

        # 没什么用
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass
        # ------------------- 更新当前层的args（参数）,计算c2（当前层的输出channel） -------------------
        # depth gain 控制深度  如v5s: n*0.33   n: 当前模块的次数(间接控制深度)
        n = max(round(n * gd), 1) if n > 1 else n
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, DWConv, MixConv2d,
                 Focus, CrossConv, BottleneckCSP, C3, C3TR, CBAM]:
            # c1: 当前层的输入的channel数  c2: 当前层的输出的channel数(初定)  ch: 记录着所有层的输出channel
            c1, c2 = ch[f], args[0]
            # if not output  no=75  只有最后一层c2=no  最后一层不用控制宽度，输出channel必须是no
            if c2 != no:
                # width gain 控制宽度  如v5s: c2*0.5  c2: 当前层的最终输出的channel数(间接控制宽度)
                c2 = make_divisible(c2 * gw, 8)

            # 在初始arg的基础上更新 加入当前层的输入channel并更新当前层
            # [in_channel, out_channel, *args[1:]]
            args = [c1, c2, *args[1:]]
            # 如果当前层是BottleneckCSP/C3/C3TR, 则需要在args中加入bottleneck的个数
            # [in_channel, out_channel, Bottleneck的个数n, bool(True表示有shortcut 默认，反之无)]
            if m in [BottleneckCSP, C3, C3TR]:
                args.insert(2, n)  # 在第二个位置插入bottleneck个数n
                n = 1  # 恢复默认值1
        elif m is nn.BatchNorm2d:
            # BN层只需要返回上一层的输出channel
            args = [ch[f]]
        elif m is Concat:
            # Concat层则将f中所有的输出累加得到这层的输出channel
            c2 = sum([ch[x] for x in f])
        elif m is Detect:  # Detect（YOLO Layer）层
            # 在args中加入三个Detect层的输出channel
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors  几乎不执行
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:  # 不怎么用
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:   # 不怎么用
            c2 = ch[f] // args[0] ** 2
        elif m is SELayer:  # 加入SE模块
            channel, re = args[0], args[1]
            channel = make_divisible(channel * gw, 8) if channel != no else channel
            args = [channel, re]
        else:
            # Upsample
            c2 = ch[f]  # args不变
        # -----------------------------------------------------------------------------------

        # m_: 得到当前层module  如果n>1就创建多个m(当前层结构), 如果n=1就创建一个m
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)

        # 打印当前层结构的一些基本信息
        t = str(m)[8:-2].replace('__main__.', '')  # t = module type   'modules.common.Focus'
        np = sum([x.numel() for x in m_.parameters()])  # number params  计算这一层的参数量
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # index, 'from' index, number, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print

        # append to savelist  把所有层结构中from不是-1的值记下  [6, 4, 14, 10, 17, 20, 23]
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)

        # 将当前层结构module加入layers中
        layers.append(m_)

        if i == 0:
            ch = []  # 去除输入channel [3]

        # 把当前层的输出channel数加入ch
        ch.append(c2)

    return nn.Sequential(*layers), sorted(save)

class DecoupledHead(nn.Module):
    def __init__(self, ch=256, nc=80, width=1.0, anchors=()):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.merge = Conv(ch, 256 * width, 1, 1)
        self.cls_convs1 = Conv(256 * width, 256 * width, 3, 1, 1)
        self.cls_convs2 = Conv(256 * width, 256 * width, 3, 1, 1)
        self.reg_convs1 = Conv(256 * width, 256 * width, 3, 1, 1)
        self.reg_convs2 = Conv(256 * width, 256 * width, 3, 1, 1)
        self.cls_preds = nn.Conv2d(256 * width, self.nc * self.na, 1)
        self.reg_preds = nn.Conv2d(256 * width, 4 * self.na, 1)
        self.obj_preds = nn.Conv2d(256 * width, 1 * self.na, 1)

    def forward(self, x):
        x = self.merge(x)
        x1 = self.cls_convs1(x)
        x1 = self.cls_convs2(x1)
        x1 = self.cls_preds(x1)
        x2 = self.reg_convs1(x)
        x2 = self.reg_convs2(x2)
        x21 = self.reg_preds(x2)
        x22 = self.obj_preds(x2)
        out = torch.cat([x21, x22, x1], 1)
        return out

class Detect(nn.Module):
    """Detect模块是用来构建Detect层的，将输入feature map 通过一个卷积操作和公式计算到我们想要的shape, 为后面的计算损失或者NMS作准备"""
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter 再export中这个参数会重新设为True

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        """
        detection layer 相当于yolov3中的YOLOLayer层
        :params nc: number of classes
        :params anchors: 传入3个feature map上的所有anchor的大小（P3、P4、P5）
        :params ch: [128, 256, 512] 3个输出feature map的channel
        """
        super(Detect, self).__init__()
        self.nc = nc  # number of classes VOC: 20
        self.no = nc + 5  # number of outputs per anchor  VOC: 5+20=25  xywhc+20classes
        self.nl = len(anchors)  # number of detection layers   Detect的个数 3
        self.na = len(anchors[0]) // 2  # number of anchors  每个feature map的anchor个数 3
        self.grid = [torch.zeros(1)] * self.nl  # init grid  {list: 3}  tensor([0.]) X 3
        # a=[3, 3, 2]  anchors以[w, h]对的形式存储  3个feature map 每个feature map上有三个anchor（w,h）
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        # register_buffer
        # 模型中需要保存的参数一般有两种：一种是反向传播需要被optimizer更新的，称为parameter; 另一种不要被更新称为buffer
        # buffer的参数更新是在forward中，而optim.step只能更新nn.parameter类型的参数
        # shape(nl,na,2)
        self.register_buffer('anchors', a)
        # shape(nl,1,na,1,1,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))
        # output conv 对每个输出的feature map都要调用一次conv1x1
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        # 调用解耦头 DecoupledHead
        # self.m = nn.ModuleList(DecoupledHead(x, nc, 1, anchors) for x in ch)
        # use in-place ops (e.g. slice assignment) 一般都是True 默认不使用AWS Inferentia加速
        self.inplace = inplace

    def forward(self, x):
        """
        :return train: 一个tensor list 存放三个元素   [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
                       分别是 [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]
                inference: 0 [1, 19200+4800+1200, 25] = [bs, anchor_num*grid_w*grid_h, xywh+c+20classes]
                           1 一个tensor list 存放三个元素 [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
                             [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]
        """
        # x = x.copy()  # for profiling
        z = []  # inference output
        for i in range(self.nl):  # 对三个feature map分别进行处理
            x[i] = self.m[i](x[i])  # conv  xi[bs, 128/256/512, 80, 80] to [bs, 75, 80, 80]
            bs, _, ny, nx = x[i].shape
            # [bs, 75, 80, 80] to [1, 3, 25, 80, 80] to [1, 3, 80, 80, 25]
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            # inference
            if not self.training:
                # 构造网格
                # 因为推理返回的不是归一化后的网格偏移量 需要再加上网格的位置 得到最终的推理坐标 再送入nms
                # 所以这里构建网格就是为了纪律每个grid的网格坐标 方面后面使用
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if self.inplace:
                    # 默认执行 不使用AWS Inferentia
                    # 这里的公式和yolov3、v4中使用的不一样 是yolov5作者自己用的 效果更好
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    # y[..., 2:4] = torch.exp(y[..., 2:4]) * self.anchor_wh     # wh yolo method
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh power method
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2) # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                # z是一个tensor list 三个元素 分别是[1, 19200, 25] [1, 4800, 25] [1, 1200, 25]
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        """
        构造网格
        """
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):
        """
        :params cfg:模型配置文件
        :params ch: input img channels 一般是3 RGB文件
        :params nc: number of classes 数据集的类别个数
        :anchors: 一般是None
        """
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:
            # is *.yaml  一般执行这里
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name  # cfg file name = yolov5s.yaml
            # 如果配置文件中有中文，打开时要加encoding参数
            with open(cfg, encoding='utf-8') as f:
                # model dict  取到配置文件中每条的信息（没有注释内容）
                self.yaml = yaml.safe_load(f)

        # input channels  ch=3
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        # 设置类别数 一般不执行, 因为nc=self.yaml['nc']恒成立
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        # 重写anchor，一般不执行, 因为传进来的anchors一般都是None
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value

        # 创建网络模型
        # self.model: 初始化的整个网络模型(包括Detect层结构)
        # self.save: 所有层结构中from不等于-1的序号，并排好序  [4, 6, 10, 14, 17, 20, 23]
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])

        # default class names ['0', '1', '2',..., '19']
        self.names = [str(i) for i in range(self.yaml['nc'])]

        # self.inplace=True  默认True  不使用加速推理
        # AWS Inferentia Inplace compatiability
        # https://github.com/ultralytics/yolov5/pull/2953
        self.inplace = self.yaml.get('inplace', True)
        # logger.info([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # 获取Detect模块的stride(相对输入图像的下采样率)和anchors在当前Detect输出的feature map的尺度
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            # 计算三个feature map下采样的倍率  [8, 16, 32]
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            # 求出相对当前feature map的anchor大小 如[10, 13]/8 -> [1.25, 1.625]
            m.anchors /= m.stride.view(-1, 1, 1)
            # 检查anchor顺序与stride顺序是否一致
            check_anchor_order(m)
            self.stride = m.stride
            # 调用解耦头这里要注释掉
            self._initialize_biases()  # only run once 初始化偏置
            # logger.info('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)  # 调用torch_utils.py下initialize_weights初始化模型权重
        self.info()  # 打印模型信息
        logger.info('')

    def forward(self, x, augment=False, profile=False):
        # augmented inference, None  上下flip/左右flip
        # 是否在测试时也使用数据增强  Test Time Augmentation(TTA)
        if augment:
            return self.forward_augment(x)
        else:
            # 默认执行 正常前向推理
            # single-scale inference, train
            return self.forward_once(x, profile)

    def forward_augment(self, x):
        """
        TTA Test Time Augmentation
        """
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales ratio
        f = [None, 3, None]  # flips (2-ud上下flip, 3-lr左右flip)
        y = []  # outputs
        for si, fi in zip(s, f):
            # scale_img缩放图片尺寸
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self.forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            # _descale_pred将推理结果恢复到相对原图图片尺寸
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        return torch.cat(y, 1), None  # augmented inference, train

    def forward_once(self, x, profile=False, feature_vis=False):
        """
        :params x: 输入图像
        :params profile: True 可以做一些性能评估
        :params feature_vis: True 可以做一些特征可视化
        :return train: 一个tensor list 存放三个元素   [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
                       分别是 [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]
                inference: 0 [1, 19200+4800+1200, 25] = [bs, anchor_num*grid_w*grid_h, xywh+c+20classes]
                           1 一个tensor list 存放三个元素 [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
                             [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]
        """
        # y: 存放着self.save=True的每一层的输出，因为后面的层结构concat等操作要用到
        # dt: 在profile中做性能评估时使用
        y, dt = [], []
        for m in self.model:
            # 前向推理每一层结构   m.i=index   m.f=from   m.type=类名   m.np=number of params
            # if not from previous layer   m.f=当前层的输入来自哪一层的输出  s的m.f都是-1
            if m.f != -1:
                # 这里需要做4个concat操作和1个Detect操作
                # concat操作如m.f=[-1, 6] x就有两个元素,一个是上一层的输出,另一个是index=6的层的输出 再送到x=m(x)做concat操作
                # Detect操作m.f=[17, 20, 23] x有三个元素,分别存放第17层第20层第23层的输出 再送到x=m(x)做Detect的forward
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            # 打印日志信息  FLOPs time等
            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                if m == self.model[0]:
                    logger.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
                logger.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')

            x = m(x)  # run正向推理  执行每一层的forward函数(除Concat和Detect操作)

            # 存放着self.save的每一层的输出，因为后面需要用来作concat等操作要用到  不在self.save层的输出就为None
            y.append(x if m.i in self.save else None)

            # 特征可视化 可以自己改动想要哪层的特征进行可视化
            if feature_vis and m.type == 'models.common.SPP':
                feature_visualization(x, m.type, m.i)

        # 打印日志信息  前向推理时间
        if profile:
            logger.info('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):
        """用在上面的__init__函数上
        initialize biases into Detect(), cf is class frequency
        https://arxiv.org/abs/1708.02002 section 3.3
        """
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def info(self, verbose=False, img_size=640):  # print model information
        """用在上面的__init__函数上
        调用torch_utils.py下model_info函数打印模型信息
        """
        model_info(self, verbose, img_size)

    def _descale_pred(self, p, flips, scale, img_size):
        """用在上面的__init__函数上
        将推理结果恢复到原图图片尺寸  Test Time Augmentation(TTA)中用到
        de-scale predictions following augmented inference (inverse operation)
        :params p: 推理结果
        :params flips:
        :params scale:
        :params img_size:
        """
        # 不同的方式前向推理使用公式不同 具体可看Detect函数
        if self.inplace:  # 默认执行 不使用AWS Inferentia
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _print_biases(self):
        """
        打印模型中最后Detect层的偏置bias信息(也可以任选哪些层bias信息)
        """
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            logger.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    def _print_weights(self):
        """
        打印模型中Bottleneck层的权重参数weights信息(也可以任选哪些层weights信息)
        """
        for m in self.model.modules():
            if type(m) is Bottleneck:
                logger.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):
        """用在detect.py、val.py
        fuse model Conv2d() + BatchNorm2d() layers
        调用torch_utils.py中的fuse_conv_and_bn函数和common.py中Conv模块的fuseforward函数
        """
        logger.info('Fusing layers... ')  # 日志
        # 遍历每一层结构
        for m in self.model.modules():
            # 如果当前层是卷积层Conv且有bn结构, 那么就调用fuse_conv_and_bn函数讲conv和bn进行融合, 加速推理
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # 融合 update conv
                delattr(m, 'bn')  # 移除bn remove batchnorm
                m.forward = m.fuseforward  # 更新前向传播 update forward (反向传播不用管, 因为这种推理只用在推理阶段)
        self.info()  # 打印conv+bn融合后的模型信息
        return self

    def nms(self, mode=True):
        """
        add or remove NMS module
        可以自选是否扩展model 增加模型nms功能  直接调用common.py中的NMS模块
         一般是用不到的 前向推理结束直接掉用non_max_suppression函数即可
        """
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            logger.info('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add nms module to model
            self.eval()  # nms 开启模型验证模式
        elif not mode and present:
            logger.info('Removing NMS... ')
            self.model = self.model[:-1]  # remove nms from model
        return self

    def autoshape(self):
        """
        add AutoShape module  直接调用common.py中的AutoShape模块  也是一个扩展模型功能的模块
        """
        logger.info('Adding AutoShape... ')
        # wrap model 扩展模型功能 此时模型包含前处理、推理、后处理的模块(预处理 + 推理 + nms)
        m = AutoShape(self)
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.info()
    # # model.train()
    #
    # # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    # # y = model(img, profile=True)
    #
    # # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # logger.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard

