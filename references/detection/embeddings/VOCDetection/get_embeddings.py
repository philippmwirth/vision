import torch
import torch.nn as nn
import torchvision
import lightly


class EmbeddingVOC:

    def __init__(self, path: str):
        self.dataset = torchvision.datasets.VOCDetection(
            path,
            download=False,
            image_set='train',
            transform=torchvision.transforms.ToTensor()
        )

    def __getitem__(self, i):
        image, target = self.dataset[i]
        return image, 0, f'image_{i}.png'

    def __len__(self):
        return len(self.dataset)


def main():


    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    dataset = EmbeddingVOC('../../datasets')

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=8
    )

    resnet = lightly.models.ResNetGenerator('resnet-18', 1.0)
    features = nn.Sequential(
        lightly.models.batchnorm.get_norm_layer(3, 0),
        *list(resnet.children())[:-1],
        nn.Conv2d(512, 32, 1),
        nn.AdaptiveAvgPool2d(1),
    )

    model = lightly.models.SimCLR(
        features,
        num_ftrs=32,
        out_dim=128,
    ).to(device)

    url = lightly.models.zoo.ZOO['resnet-18/simclr/d32/w1.0']
    state_dict = lightly.cli._helpers.load_state_dict_from_url(
        url, map_location=device)['state_dict']
    lightly.cli._helpers.load_from_state_dict(model, state_dict)

    encoder = lightly.embedding.SelfSupervisedEmbedding(model, None, None, None)
    embeddings, labels, filenames = encoder.embed(dataloader, device=device)

    lightly.utils.save_embeddings('embeddings.csv', embeddings, labels, filenames)



if __name__ == '__main__':

    main()