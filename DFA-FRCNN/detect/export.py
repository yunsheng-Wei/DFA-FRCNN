
import argparse
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path
import onnx
import pandas as pd
import torch
import yaml
from torch.utils.mobile_optimizer import optimize_for_mobile

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))
from utils.torch_utils import select_device


def export_formats():
    x = [
        ['PyTorch', '-', '.pt', True, True],
        ['TorchScript', 'torchscript', '.torchscript', True, True],
        ['ONNX', 'onnx', '.onnx', True, True],
        ['OpenVINO', 'openvino', '_openvino_model', True, False],
        ['TensorRT', 'engine', '.engine', False, True],
        ['CoreML', 'coreml', '.mlmodel', True, False]]
    return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'CPU', 'GPU'])

def export_torchscript(model, im, file, optimize, prefix=colorstr('TorchScript:')):

    f = file.with_suffix('.torchscript')
    ts = torch.jit.trace(model, im, strict=False)
    d = {"shape": im.shape, "stride": int(max(model.stride)), "names": model.names}
    extra_files = {'config.txt': json.dumps(d)}
    if optimize:
        optimize_for_mobile(ts)._save_for_lite_interpreter(str(f), _extra_files=extra_files)
    else:
        ts.save(str(f), _extra_files=extra_files)
    return f



def export_onnx(model, im, file, opset, train, dynamic, simplify, prefix=colorstr('ONNX:')):
    check_requirements(('onnx',))
    f = file.with_suffix('.onnx')
    torch.onnx.export(
        model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
        im.cpu() if dynamic else im,
        f,
        verbose=False,
        opset_version=opset,
        training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
        do_constant_folding=not train,
        input_names=['images'],
        output_names=['outputs'],
        dynamic_axes={
            'images': {
                0: 'batch',
                2: 'height',
                3: 'width'},
            'output': {
                0: 'batch',
                1: 'anchors'}
        } if dynamic else None)
    model_onnx = onnx.load(f)
    onnx.checker.check_model(model_onnx)
    d = {'stride': int(max(model.stride)), 'names': model.names}
    for k, v in d.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)
    onnx.save(model_onnx, f)
    # Simplify
    if simplify:
        cuda = torch.cuda.is_available()
        check_requirements(('onnxruntime-gpu' if cuda else 'onnxruntime', 'onnx-simplifier>=0.4.1'))
        import onnxsim
        model_onnx, check = onnxsim.simplify(model_onnx)
        assert check, 'assert check failed'
        onnx.save(model_onnx, f)
    return f

def export_openvino(model, file, half, prefix=colorstr('OpenVINO:')):
    check_requirements(('openvino-dev',))
    import openvino.inference_engine as ie
    f = str(file).replace('.pt', f'_openvino_model{os.sep}')
    cmd = f"mo --input_model {file.with_suffix('.onnx')} --output_dir {f} --data_type {'FP16' if half else 'FP32'}"
    subprocess.check_output(cmd.split())  # export
    with open(Path(f) / file.with_suffix('.yaml').name, 'w') as g:
        yaml.dump({'stride': int(max(model.stride)), 'names': model.names}, g)  # add metadata.yaml

    return f

def export_engine(model, im, file, train, half, dynamic, simplify, workspace=4, verbose=False):
    prefix = colorstr('TensorRT:')
        assert im.device.type != 'cpu', 'export running on CPU but must be on GPU, i.e. `python export.py --device 0`'
        try:
            import tensorrt as trt
        except Exception:
            if platform.system() == 'Linux':
            import tensorrt as trt
        if trt.__version__[0] == '7':
            grid = model.model[-1].anchor_grid
            model.model[-1].anchor_grid = [a[..., :1, :1, :] for a in grid]
            export_onnx(model, im, file, 12, train, dynamic, simplify)  # opset 12
            model.model[-1].anchor_grid = grid
        else:  # TensorRT >= 8
            export_onnx(model, im, file, 13, train, dynamic, simplify)  # opset 13
        onnx = file.with_suffix('.onnx')
        assert onnx.exists(), f'failed to export ONNX file: {onnx}'
        f = file.with_suffix('.engine')  # TensorRT engine file
        logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            logger.min_severity = trt.Logger.Severity.VERBOSE
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        config.max_workspace_size = workspace * 1 << 30
        flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx)):
            raise RuntimeError(f'failed to load ONNX file: {onnx}')
        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        for inp in inputs:
        for out in outputs:
        if dynamic:
            if im.shape[0] <= 1:
            profile = builder.create_optimization_profile()
            for inp in inputs:
                profile.set_shape(inp.name, (1, *im.shape[1:]), (max(1, im.shape[0] // 2), *im.shape[1:]), im.shape)
            config.add_optimization_profile(profile)
        if builder.platform_has_fast_fp16 and half:
            config.set_flag(trt.BuilderFlag.FP16)
        with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
            t.write(engine.serialize())
        return f

def run(
        data=ROOT / 'data/coco128.yaml',  # 'dataset.yaml path'
        weights=ROOT / '',  # weights path
        imgsz=(640, 640),  # image (height, width)
        batch_size=1,  # batch size
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        include=('onnx'),  # include formats
        half=False,  # FP16 half-precision export
        train=False,  # model.train() mode
        optimize=False,  # TorchScript: optimize for mobile
        int8=False,  # CoreML/TF INT8 quantization
        dynamic=False,  # ONNX/TF/TensorRT: dynamic axes
        simplify=True,  # ONNX: simplify model
        opset=12,  # ONNX: opset version
        verbose=False,  # TensorRT: verbose log
        workspace=4,  # TensorRT: workspace size (GB)
):
    t = time.time()
    include = [x.lower() for x in include]  # to lowercase
    fmts = tuple(export_formats()['Argument'][1:])  # --include arguments
    flags = [x in include for x in fmts]
    assert sum(flags) == len(include), f'ERROR: Invalid --include {include}, valid --include arguments are {fmts}'
    jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs = flags  # export booleans
    file = Path(url2file(weights) if str(weights).startswith(('http:/', 'https:/')) else weights)  # PyTorch weights
    # Load PyTorch model
    device = select_device(device)
    if half:
        assert device.type != 'cpu' or coreml, '--half only compatible with GPU export, i.e. use --device 0'
        assert not dynamic, '--half not compatible with --dynamic, i.e. use either --half or --dynamic but not both'
    model = attempt_load(weights, device=device, inplace=True, fuse=True)  # load FP32 model
    # Checks
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    if optimize:
        assert device.type == 'cpu', '--optimize not compatible with cuda devices, i.e. use --device cpu'
    # Input
    gs = int(max(model.stride))
    imgsz = [check_img_size(x, gs) for x in imgsz]
    im = torch.zeros(batch_size, 3, *imgsz).to(device)
    # Update model
    model.train() if train else model.eval()
    for k, m in model.named_modules():
        if isinstance(m, Detect):
            m.inplace = inplace
            m.onnx_dynamic = dynamic
            m.export = True
    for _ in range(2):
        y = model(im)  # dry runs
    if half and not coreml:
        im, model = im.half(), model.half()  # to FP16
    shape = tuple((y[0] if isinstance(y, tuple) else y).shape)
    # Exports
    f = [''] * 10  # exported filenames
    warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)
    if jit:
        f[0] = export_torchscript(model, im, file, optimize)
    if engine:
        f[1] = export_engine(model, im, file, train, half, dynamic, simplify, workspace, verbose)
    if onnx or xml:
        f[2] = export_onnx(model, im, file, opset, train, dynamic, simplify)
    if xml:
        f[3] = export_openvino(model, file, half)
    # Finish
    f = [str(x) for x in f if x]  # filter out '' and None
    if any(f):
        h = '--half' if half else ''  # --half FP16 inference arg
    return f  # return list of exported files/dirs
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='', help='model.pt path(s)')
    parser.add_argument('--imgsize', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='image (h, w)')
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--optimize', action='store_true', help='TorchScript: optimize for mobile')
    parser.add_argument('--int8', action='store_true', help='CoreML/TF INT8 quantization')
    parser.add_argument('--dynamic', action='store_true', help='ONNX/TF/TensorRT: dynamic axes')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=12, help='ONNX: opset version')
    parser.add_argument('--include',
                        nargs='+',
                        default=['onnx'],
                        help='torchscript, onnx, openvino, engine')
    opt = parser.parse_args()
    return opt

def main(opt):
    for opt.weights in (opt.weights if isinstance(opt.weights, list) else [opt.weights]):
        run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
