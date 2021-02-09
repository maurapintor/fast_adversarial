import foolbox as fb
import torch

from fast_adv.utils import NormalizedModel

model_urls = {
    'MNIST': "https://www.dropbox.com/s/9onr3jfsuc3b4dh/mnist.pth?dl=1",
    'CIFAR10': "https://www.dropbox.com/s/ppydug8zefsrdqn/cifar10_wrn28-10.pth?dl=1"
}

def create(dataset='MNIST'):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    weights_path = fb.zoo.fetch_weights(
        model_urls[dataset],
        unzip=False
    )
    if dataset == 'MNIST':
        from fast_adv.models.mnist import SmallCNN
        model = SmallCNN()
    elif dataset == 'CIFAR10':
        from fast_adv.models.cifar10 import wide_resnet
        image_mean = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1)
        model = wide_resnet(num_classes=10, depth=28, widen_factor=10, dropRate=0.0)
        model = NormalizedModel(model=model, mean=image_mean, std=image_std).to(device)
    else:
        raise ValueError("Model not available.")


    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    fmodel = fb.models.PyTorchModel(model, bounds=(0, 1),
                                    device=device)

    return fmodel
