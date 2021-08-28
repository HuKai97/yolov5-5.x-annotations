"""Export a YOLOv5 *.pt model to TorchScript, ONNX, CoreML formats

Usage:
    $ python path/to/export.py --weights yolov5s.pt --img 640 --batch 1
"""

import argparse             # 解析命令行参数模块
import sys                  # sys系统模块 包含了与Python解释器和它的环境有关的函数
import time                 # 时间模块 更底层
from pathlib import Path    # Path将str转换为Path对象 使字符串路径易于操作的模块

import torch                # PyTorch深度学习模块
import torch.nn as nn       # 对torch.nn.functional的类的封装 有很多和torch.nn.functional相同的函数
from torch.utils.mobile_optimizer import optimize_for_mobile  # 对模型进行移动端优化模块

FILE = Path(__file__).absolute()  # FILE = WindowsPath 'F:\yolo_v5\yolov5-U\export.py'
# 将'F:/yolo_v5/yolov5-U'加入系统的环境变量  该脚本结束后失效
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.common import Conv
from models.yolo import Detect
from models.experimental import attempt_load
from models.activations import Hardswish, SiLU
from utils.general import colorstr, check_img_size, check_requirements, file_size, set_logging
from utils.torch_utils import select_device


def run(weights='../weights/yolov5s.pt', img_size=(640, 640), batch_size=1, device='cpu',
        include=('torchscript', 'onnx', 'coreml'), half=False, inplace=False, train=False,
        optimize=False, dynamic=False, simplify=False, opset_version=12,):
    """
    weights: 要转换的权重文件pt地址 默认='../weights/best.pt'
    img-size: 输入模型的图片size=(height, width) 默认=[640, 640]
    batch-size: batch大小 默认=1
    device: 模型运行设备 cuda device, i.e. 0 or 0,1,2,3 or cpu 默认=cpu
    include: 要将pt文件转为什么格式 可以为单个原始也可以为list 默认=['torchscript', 'onnx', 'coreml']
    half: 是否使用半精度FP16export转换 默认=False
    inplace: 是否set YOLOv5 Detect() inplace=True  默认=False
    train: 是否开启model.train() mode 默认=True  coreml转换必须为True
    optimize: TorchScript转化参数 是否进行移动端优化  默认=False
    dynamic: ONNX转换参数  dynamic_axes  ONNX转换是否要进行批处理变量  默认=False
    simplify: ONNX转换参数 是否简化onnx模型  默认=False
    opset-version: ONNX转换参数 设置版本  默认=10
    """
    t = time.time()   # 获取当前时间
    include = [x.lower() for x in include]  # pt文件要转化的格式包括哪些
    img_size *= 2 if len(img_size) == 1 else 1  # expand

    # Load PyTorch model
    device = select_device(device)  # 选择设备
    assert not (device.type == 'cpu' and half), '--half only compatible with GPU export, i.e. use --device 0'
    model = attempt_load(weights, map_location=device)  # load FP32 model
    labels = model.names  # 载入数据集name

    # Input
    gs = int(max(model.stride))  # grid size (max stride)
    img_size = [check_img_size(x, gs) for x in img_size]  # verify img_size are gs-multiples
    img = torch.zeros(batch_size, 3, *img_size).to(device)  # 自定义一张图片 输入model

    # Update model
    # 是否采样半精度FP16训练or推理
    if half:
        img, model = img.half(), model.half()  # to FP16
    # 是否开启train模式
    model.train() if train else model.eval()  # training mode = no Detect() layer grid construction
    # 调整模型配置
    for k, m in model.named_modules():
        # pytorch 1.6.0 compatibility(关于版本兼容的设置) 使模型兼容pytorch 1.6.0
        m._non_persistent_buffers_set = set()
        # assign export-friendly activations(有些导出的格式是不兼容系统自带的nn.Hardswish、nn.SiLU的)
        if isinstance(m, Conv):
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        # 模型相关设置: Detect类的inplace参数和onnx_dynamic参数
        elif isinstance(m, Detect):
            m.inplace = inplace
            m.onnx_dynamic = dynamic   # 设置Detect的onnx_dynamic参数为dynamic
            # m.forward = m.forward_export  # assign forward (optional)

    for _ in range(2):
        y = model(img)  # dry runs  前向推理2次
    print(f"\n{colorstr('PyTorch:')} starting from {weights} ({file_size(weights):.1f} MB)")

    # ================================ 转换模型 ====================================
    # TorchScript export -----------------------------------------------------------------------------------------------
    if 'torchscript' in include or 'coreml' in include:
        prefix = colorstr('TorchScript:')
        try:
            print(f'\n{prefix} starting export with torch {torch.__version__}...')
            f = weights.replace('.pt', '.torchscript.pt')  # export torchscript filename
            ts = torch.jit.trace(model, img, strict=False)          # convert
            # optimize_for_mobile: 移动端优化
            (optimize_for_mobile(ts) if optimize else ts).save(f)   # save
            print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        except Exception as e:
            print(f'{prefix} export failure: {e}')

    # ONNX export ------------------------------------------------------------------------------------------------------
    if 'onnx' in include:
        prefix = colorstr('ONNX:')
        try:
            import onnx

            print(f'{prefix} starting export with onnx {onnx.__version__}...')  # 日志
            f = weights.replace('.pt', '.onnx')  # export filename
            # convert
            torch.onnx.export(model, img, f, verbose=False, opset_version=opset_version,
                              training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
                              do_constant_folding=not train,   # 是否执行常量折叠优化
                              input_names=['images'],     # 输入名
                              output_names=['output'],    # 输出名
                              # 批处理变量 若不想支持批处理或固定批处理大小,移除dynamic_axes字段即可
                              dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
                                            'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
                                            } if dynamic else None)

            # Checks
            model_onnx = onnx.load(f)  # load onnx model
            onnx.checker.check_model(model_onnx)  # check onnx model
            # print(onnx.helper.printable_graph(model_onnx.graph))  # print

            # Simplify
            if simplify:
                try:
                    check_requirements(['onnx-simplifier'])
                    import onnxsim

                    print(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                    # simplify 简化
                    model_onnx, check = onnxsim.simplify(
                        model_onnx,
                        dynamic_input_shape=dynamic,
                        input_shapes={'images': list(img.shape)} if dynamic else None)
                    assert check, 'assert check failed'
                    onnx.save(model_onnx, f)  # save
                except Exception as e:
                    print(f'{prefix} simplifier failure: {e}')
            print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        except Exception as e:
            print(f'{prefix} export failure: {e}')

    # CoreML export ----------------------------------------------------------------------------------------------------
    # 注意: 转换CoreML时必须设置model.train 即opt参数train为True
    if 'coreml' in include:
        prefix = colorstr('CoreML:')
        try:
            import coremltools as ct

            print(f'{prefix} starting export with coremltools {ct.__version__}...')
            assert train, 'CoreML exports should be placed in model.train() mode with `python export.py --train`'
            model = ct.convert(ts, inputs=[ct.ImageType('image', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])  # convert
            f = weights.replace('.pt', '.mlmodel')  # export coreml filename
            model.save(f)  # save
            print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        except Exception as e:
            print(f'{prefix} export failure: {e}')

    # Finish
    print(f'\nExport complete ({time.time() - t:.2f}s). Visualize with https://github.com/lutzroeder/netron.')


def parse_opt():
    """
    weights: 要转换的权重文件pt地址 默认='../weights/best.pt'
    img-size: 输入模型的图片size=(height, width) 默认=[640, 640]
    batch-size: batch大小 默认=1
    device: 模型运行设备 cuda device, i.e. 0 or 0,1,2,3 or cpu 默认=cpu
    include: 要将pt文件转为什么格式 可以为单个原始也可以为list 默认=['torchscript', 'onnx', 'coreml']
    half: 是否使用半精度FP16export转换 默认=False
    inplace: 是否set YOLOv5 Detect() inplace=True  默认=False
    train: 是否开启model.train() mode 默认=True  coreml转换必须为True
    optimize: TorchScript转化参数 是否进行移动端优化  默认=False
    dynamic: ONNX转换参数  dynamic_axes  ONNX转换是否要进行批处理变量  默认=False
    simplify: ONNX转换参数 是否简化onnx模型  默认=False
    opset-version: ONNX转换参数 设置版本  默认=10
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='../weights/best.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image (height, width)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--include', nargs='+', default=['torchscript', 'onnx', 'coreml'], help='include formats')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set YOLOv5 Detect() inplace=True')
    parser.add_argument('--train', default="True", action='store_true', help='model.train() mode')
    parser.add_argument('--optimize', action='store_true', help='TorchScript: optimize for mobile')
    parser.add_argument('--dynamic', action='store_true', help='ONNX: dynamic axes')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    parser.add_argument('--opset-version', type=int, default=10, help='ONNX: opset version')
    opt = parser.parse_args()
    return opt


def main(opt):
    # 初始化日志
    set_logging()
    print(colorstr('export: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))  # 彩色打印
    # 脚本主体
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
