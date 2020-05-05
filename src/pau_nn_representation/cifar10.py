import os, shutil
import torch
from argparse import ArgumentParser
from src.pau_nn_representation.pretrained.cifar10_module import CIFAR10_Module, CIFAR10_Module_represention
from pytorch_lightning import Trainer


def train_representations(hparams):
    # Reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)

    # Set GPU
    torch.cuda.set_device(hparams.gpu)

    # Train
    classifier = CIFAR10_Module_represention(hparams)
    trainer = Trainer(default_save_path=os.path.join(os.getcwd(), 'tensorboard_logs', hparams.classifier),
                      gpus=[hparams.gpu], max_epochs=hparams.max_epochs,
                      early_stop_callback=False)
    trainer.fit(classifier)

    # Save weights from checkpoint
    checkpoint_path = os.path.join(os.getcwd(), 'tensorboard_logs', hparams.classifier, 'lightning_logs', 'version_0',
                                   'checkpoints')
    classifier = CIFAR10_Module_represention.load_from_checkpoint(os.path.join(checkpoint_path, os.listdir(checkpoint_path)[0]))
    statedict_path = os.path.join(os.getcwd(), 'cifar10_models', 'state_dicts', hparams.classifier + '.pt')
    torch.save(classifier.model.state_dict(), statedict_path)


def finetune_representations(hparams):
    # Reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)

    # Set GPU
    torch.cuda.set_device(hparams.gpu)

    # Train
    classifier = CIFAR10_Module(hparams)
    trainer = Trainer(default_save_path=os.path.join(os.getcwd(), 'tensorboard_logs', hparams.classifier),
                      gpus=[hparams.gpu], max_epochs=hparams.max_epochs,
                      early_stop_callback=False)
    trainer.fit(classifier)

    # Save weights from checkpoint
    checkpoint_path = os.path.join(os.getcwd(), 'tensorboard_logs', hparams.classifier, 'lightning_logs', 'version_0',
                                   'checkpoints')
    classifier = CIFAR10_Module.load_from_checkpoint(os.path.join(checkpoint_path, os.listdir(checkpoint_path)[0]))
    statedict_path = os.path.join(os.getcwd(), 'trained_models', 'cifar10_models', 'state_dicts', hparams.classifier + '.pt')
    torch.save(classifier.model.state_dict(), statedict_path)


def prepare(hparams):
    torch.cuda.set_device(hparams.gpu)
    model = CIFAR10_Module(hparams)
    trainloader = model.train_dataloader()
    testloader = model.val_dataloader()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train_mode', type=str, default='train_representations')
    parser.add_argument('--classifier', type=str, default='vgg19_bn')
    parser.add_argument('--classifier_approx', type=str, default='vgg11_bn_pau')
    parser.add_argument('--data_dir', type=str, default='/media/disk2/datasets/pytorch_dataset/cifar10/')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--reduce_lr_per', type=int, default=50)
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'])
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--pretrained_approx', type=bool, default=False)
    parser.add_argument('--pretrained_classifier', type=bool, default=True)
    args = parser.parse_args()

    if args.train_mode == 'train_representations':
        train_representations(args)
    elif args.train_mode == 'finetune_representations':
        finetune_representations(args)
