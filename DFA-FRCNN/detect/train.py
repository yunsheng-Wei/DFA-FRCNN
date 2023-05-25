import os
import datetime

import torch

import transforms
from network_files import FasterRCNN, FastRCNNPredictor
# from backbone import resnet50_fpn_backbone
from backbone import epsanet50_fpn_backbone
from utils.my_dataset import VOC2012DataSet
from utils import train_eval_utils as utils

def select_device(device='', batch=0, newline=False, verbose=True):
    """Selects PyTorch Device. Options are device = None or 'cpu' or 0 or '0' or '0,1,2,3'."""
    s = f'DFA-FRCNN{__version__}  Python-{platform.python_version()} torch-{torch.__version__} '
    device = str(device).lower()
    for remove in 'cuda:', 'none', '(', ')', '[', ']', "'", ' ':
        device = device.replace(remove, '')  # to string, 'cuda:0' -> '0' and '(0, 1)' -> '0,1'
    cpu = device == 'cpu'
    mps = device == 'mps'  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
        if not (torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', ''))):
            LOGGER.info(s)
            install = 'See https://pytorch.org/get-started/locally/ for up-to-date torch install instructions if no ' \
                      'CUDA devices are seen by torch.\n' if torch.cuda.device_count() == 0 else ''
            raise ValueError(f"Invalid CUDA 'device={device}' requested."
                             f" Use 'device=cpu' or pass valid CUDA device(s) if available,"
                             f" i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU.\n"
                             f'\ntorch.cuda.is_available(): {torch.cuda.is_available()}'
                             f'\ntorch.cuda.device_count(): {torch.cuda.device_count()}'
                             f"\nos.environ['CUDA_VISIBLE_DEVICES']: {visible}\n"
                             f'{install}')

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch > 0 and batch % n != 0:  # check batch_size is divisible by device_count
            raise ValueError(f"'batch={batch}' must be a multiple of GPU count {n}. Try 'batch={batch // n * n}' or "
                             f"'batch={batch // n * n + n}', the nearest batch sizes evenly divisible by {n}.")
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
        arg = 'cuda:0'
    elif mps and getattr(torch, 'has_mps', False) and torch.backends.mps.is_available() and TORCH_2_X:
        # Prefer MPS if available
        s += 'MPS\n'
        arg = 'mps'
    else:  # revert to CPU
        s += 'CPU\n'
        arg = 'cpu'

    if verbose and RANK == -1:
        LOGGER.info(s if newline else s.rstrip())
    return torch.device(arg)


def create_model(num_classes, device):

    backbone = epsanet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d,
                                     trainable_layers=3)
    # num_classes
    model = FasterRCNN(backbone=backbone, num_classes=91)
    # https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
    weights_dict1 = torch.load("./weights/fasterrcnn_resnet50_fpn.pth", map_location=device)
    # for key, value in weights_dict1.items():
    #     print(key, value.size())
    # weights_dict2 = model.state_dict()
    # print("===============================================")
    # for key, value in weights_dict2.items():
    #     print(key, value.size())

    # for key, value in weights_dict2.items():
    #     if key[:13] == "backbone.body" and key in weights_dict1:
    #         value2 = weights_dict1[key]
    #         if value.size() == value2.size():
    #             weights_dict2[key] = weights_dict1[key]
    # # torch.save(weights_dict2, "./weights/resnet50.pth")
    # model.load_state_dict(weights_dict2)
    missing_keys, unexpected_keys = model.load_state_dict(weights_dict1, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def main(parser_data):
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    VOC_root = parser_data.data_path
    # check voc root
    if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
        raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(VOC_root))

    # load train data set
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> train.txt
    train_data_set = VOC2012DataSet(VOC_root, data_transform["train"], "train.txt")

    batch_size = parser_data.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)
    train_data_loader = torch.utils.data.DataLoader(train_data_set,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=nw,
                                                    collate_fn=train_data_set.collate_fn)

    # load validation data set
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
    val_data_set = VOC2012DataSet(VOC_root, data_transform["val"], "val.txt")
    val_data_set_loader = torch.utils.data.DataLoader(val_data_set,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      num_workers=nw,
                                                      collate_fn=train_data_set.collate_fn)

    # create model num_classes equal background + 20 classes
    model = create_model(num_classes=parser_data.num_classes + 1, device=device)
    # print(model)

    model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.33)

    if parser_data.resume != "":
        checkpoint = torch.load(parser_data.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        parser_data.start_epoch = checkpoint['epoch'] + 1
        print("the training process from epoch{}...".format(parser_data.start_epoch))

    train_loss = []
    learning_rate = []
    val_map = []

    for epoch in range(parser_data.start_epoch, parser_data.epochs):
        # train for one epoch, printing every 10 iterations
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device=device, epoch=epoch,
                                              print_freq=50, warmup=True)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        coco_info = utils.evaluate(model, val_data_set_loader, device=device)

        # write into txt
        with open(results_file, "a") as f:
            # 写入的数据包括coco指标还有loss和learning rate
            result_info = [str(round(i, 4)) for i in coco_info + [mean_loss.item(), lr]]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(coco_info[1])  # pascal mAP

        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        torch.save(save_files, "./save_weights/resNetFpn-model-{}.pth".format(epoch))

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)


    parser.add_argument('--device', default='cuda:0', help='device')

    parser.add_argument('--data-path', default='', help='dataset')

    parser.add_argument('--num-classes', default=20, type=int, help='num_classes')

    parser.add_argument('--output-dir', default='./save_weights', help='path where to save')

    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')

    parser.add_argument('--epochs', default=15, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--batch_size', default=2, type=int, metavar='N',
                        help='batch size when training.')

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
