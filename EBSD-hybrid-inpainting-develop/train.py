import argparse
import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from skimage.io import imsave
from torch.utils.data import DataLoader

from dataset import EBSDDataset
from geodesic import geodesic_mse
from model import Model
from model_tanimutomo import PConvUNet

EULER_CONVENTION = 'ZXZ'


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    channels_first = tensor.clone().detach().squeeze(0).cpu().numpy()
    image = channels_first.transpose((1, 2, 0))
    return image


def save_image(path, image_tensor):
    image = tensor_to_image(image_tensor)
    image *= np.array([2 * np.pi, np.pi, 2 * np.pi])
    image = (Rotation.from_euler(EULER_CONVENTION, image.reshape(-1, 3))
             .as_euler(EULER_CONVENTION).reshape(image.shape) % (2 * np.pi))
    scaled_image = image / np.array([2 * np.pi, np.pi, 2 * np.pi]) * 255
    imsave(path, scaled_image.astype(np.uint8))


def validate_args(args):
    for data_dir in (args.train_dir, args.val_dir):
        if not os.path.isdir(data_dir):
            print(f'"{args.data_dir}" is not a directory.')
            quit()
    if args.epochs <= 0:
        print('epochs must be positive')
        quit()


def log(file, message: str):
    print(message)
    file.write(message + '\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('train_dir', type=str)
    parser.add_argument('val_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('loss_function', type=str, choices=['RGB_MSE',
                                                            'geodesic_MSE'])
    parser.add_argument('device', type=str)
    parser.add_argument('epochs', type=int)
    parser.add_argument('--examples_per_epoch', type=int, default=10)
    parser.add_argument('--implementation', type=str,
                        choices=['ours', 'tanimutomo'], default='ours')
    parser.add_argument('--train_damage_size', type=int, default=20)
    parser.add_argument('--entire_image_loss',
                        action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    os.mkdir(args.output_dir)
    output_file = open(os.path.join(args.output_dir, 'log.txt'), 'x',
                       buffering=1)

    log(output_file, str(args)[10:-1])

    torch.set_default_device(args.device)

    if args.implementation == 'ours':
        model = Model()
    else:
        model = PConvUNet(False, layer_size=7)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    validate_args(args)

    train_dataset = EBSDDataset(args.train_dir, args.device,
                                missing_region_size=args.train_damage_size)
    val_dataset = EBSDDataset(args.val_dir, args.device, val=True)

    train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True,
                                  generator=torch.Generator(args.device))
    val_dataloader = DataLoader(val_dataset, batch_size=100)

    example_noisy_images = []
    example_known_masks = []
    example_clean_images = []
    example_input_images = []
    for i in range(args.examples_per_epoch):
        example_noisy_image, example_known_mask, example_clean_image = val_dataset[i]
        example_input_image = example_noisy_image * example_known_mask
        example_noisy_images.append(example_noisy_image)
        example_known_masks.append(example_known_mask)
        example_clean_images.append(example_clean_image)
        example_input_images.append(example_input_image)
        save_image(os.path.join(args.output_dir,
                                f'clean_{i}.png'),
                   example_clean_image)
        save_image(os.path.join(args.output_dir,
                                f'noisy_{i}.png'),
                   example_noisy_image)
        example_damaged_image = example_noisy_image.clone()
        example_damaged_image[example_known_mask == 0] = 1
        imsave(os.path.join(args.output_dir,
                            f'damaged_{i}.png'),
               (tensor_to_image(example_damaged_image) * 255)
               .astype(np.uint8))

    for epoch in range(args.epochs):
        total_training_loss = torch.tensor(0.)
        for step, (noisy_image, known_mask, clean_image) in enumerate(train_dataloader):
            input_image = noisy_image * known_mask
            output_image, _ = model(input_image, known_mask)
            loss = geodesic_mse(clean_image, output_image, known_mask,
                                unknown_only=not args.entire_image_loss)
            total_training_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            log(output_file,
                f'Epoch: {epoch} Step: {step} Training loss: {loss}')
        training_loss = total_training_loss / len(train_dataloader)
        model.eval()
        with torch.no_grad():
            total_validation_loss = torch.tensor(0.)
            for step, (noisy_image, known_mask, clean_image) in enumerate(val_dataloader):
                input_image = noisy_image * known_mask
                output_image, _ = model(input_image, known_mask)
                total_validation_loss += geodesic_mse(clean_image,
                                                      output_image,
                                                      known_mask,
                                                      unknown_only=True)
            validation_loss = total_validation_loss / len(val_dataloader)
            log(output_file,
                f'Training loss: {training_loss} '
                f'Validation loss: {validation_loss}')
            model_checkpoint_file = os.path.join(
                args.output_dir, f'epoch_{epoch}_model.pt')
            optimizer_checkpoint_file = os.path.join(
                args.output_dir, f'epoch_{epoch}_optimizer.pt')
            torch.save(model.state_dict(), model_checkpoint_file)
            torch.save(optimizer.state_dict(), optimizer_checkpoint_file)
            for i in range(args.examples_per_epoch):
                example_predicted_image, _ = model(
                    example_input_images[i].unsqueeze(0),
                    example_known_masks[i].unsqueeze(0))
                save_image(os.path.join(args.output_dir,
                                        f'epoch_{epoch}_predicted_{i}.png'),
                           example_predicted_image)
                example_combined_image = (example_noisy_images[i]
                                          * example_known_masks[i]
                                          + example_predicted_image
                                          * (1 - example_known_masks[i]))
                save_image(os.path.join(args.output_dir,
                                        f'epoch_{epoch}_combined_{i}.png'),
                           example_combined_image)
        model.train()

    output_file.close()
