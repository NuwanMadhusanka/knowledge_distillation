import os
import torch.utils.model_zoo as model_zoo
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import resnet


def parse_args():
    """Parse input arguments."""
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0]',
        default=0, type=int)
    parser.add_argument(
        '--num_epochs', dest='num_epochs', 
        help='Maximum number of training epochs.',
        default=50, type=int)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=64, type=int)
    parser.add_argument(
        '--lr', dest='lr', help='Base learning rate.',
        default=0.000001, type=float)
    parser.add_argument(
        '--dataset', dest='dataset', help='Dataset type.', 
        default='MNIST', type=str)
    parser.add_argument(
        '--num_classes', dest='num_classes', help='Number of classes.', 
        default=10 , type=int)
    parser.add_argument(
        '--teacher_model', dest='snapshot', help='Path of the teacher model.',
        default='', type=str)
    parser.add_argument(
        '--arch', dest='arch', 
        help='Network architecture of the student model, can be: ResNet18, ResNet34, [ResNet50], '
            'ResNet101, ResNet152, default='ResNet18', type=str)
   

    args = parser.parse_args()
    return args

def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    try:
        model.load_state_dict(model_dict)
    except RuntimeError as e:
        print('Ignoring "' + str(e) + '"')


if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id

    if not os.path.exists('output'):
        os.makedirs('output')

    #student architecture intialized
    if args.arch == 'ResNet34':
        student_model = hopenet.Hopenet(
            torchvision.models.resnet.BasicBlock, [3,4,6,3], 66)
        pre_url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
    elif args.arch == 'ResNet50':
        student_model = resnet.ResNetModel(
            torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
        pre_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
    elif args.arch == 'ResNet101':
        student_model = hopenet.Hopenet(
            torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], 66)
        pre_url = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
    elif args.arch == 'ResNet152':
        student_model = hopenet.Hopenet(
            torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], 66)
        pre_url = 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
    else:
        if args.arch != 'ResNet18':
            print('Invalid value for architecture is passed! '
                'The default value of ResNet18 will be used instead!')
        student_model = hopenet.Hopenet(
            torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], 66)
        pre_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'

    load_filtered_state_dict(student_model, model_zoo.load_url(pre_url))
    print(student_model)