import itertools

import torch
from pytorch3d.transforms import euler_angles_to_matrix
from torch import Tensor, nn

DTYPE = torch.float32
DEVICE = 'cpu'

EULER_ANGLE_CONVENTION = 'ZXZ'


@torch.no_grad()
def initial_gradient(padded_image: Tensor, padded_mask: Tensor,
                     using_template: bool, patch_size: int) -> Tensor:
    gradient = torch.tensor(torch.gradient(padded_image))
    gradient_magnitude = torch.norm(gradient, dim=0)


@torch.no_grad()
def update_gradient(gradient: Tensor, padded_image: Tensor,
                    padded_mask: Tensor, using_template: bool,
                    patch_size: int) -> None:
    pass


@torch.no_grad()
def update_confidence(confidence: Tensor, point: tuple[int, int],
                      patch_size: int) -> None:
    pass


@torch.no_grad()
def initial_data(padded_image: Tensor, padded_mask: Tensor, gradient: Tensor,
                 using_template: bool, patch_size: int) -> Tensor:
    pass


@torch.no_grad()
def update_data(padded_image: Tensor, padded_mask: Tensor, data: Tensor,
                gradient: Tensor, point: tuple[int, int], patch_size: int
                ) -> None:
    pass


@torch.no_grad()
def find_exemplar(target_matrices_patch: Tensor, matrices_patches) -> None:
    pass


@torch.no_grad()
def criminisi_inpaint(image: Tensor, mask: Tensor, template: Tensor = None,
                      patch_size: int = 3, euclidean_penalty: float = 0,
                      template_weight: float = 0) -> Tensor:

    assert(patch_size % 2 == 1)
    assert(image.shape[-2:] == mask.shape[-2:])
    if template is not None:
        assert(image.shape[-2:] == template.shape[-2:])

    image = torch.clone(image)


    if template is not None:
        image[mask == 0] = template[mask == 0]

    padding = tuple(itertools.repeat(patch_size // 2, 4))
    padded_image: Tensor = nn.ReplicationPad2d(padding)(image)
    padded_mask: Tensor = nn.ReplicationPad2d(padding)(mask)

    confidence = torch.clone(mask)
    using_template = template is not None
    gradient = initial_gradient(padded_image, padded_mask, using_template, patch_size)
    data = initial_data(padded_image, padded_mask, gradient, using_template, patch_size)

    channels_last = torch.permute(padded_image, (1, 2, 0))
    matrices_image = euler_angles_to_matrix(channels_last, EULER_ANGLE_CONVENTION)
    matrices_patches_grid = matrices_image.unfold(0, patch_size, 1).unfold(1, patch_size, 1)
    matrices_patches = matrices_patches_grid.reshape(-1, 3, 3)

    finished = torch.clone(mask)
    while not torch.all(finished):
        priority_point = torch.argmax(confidence * data)
        target_matrices_patch = matrices_patches_grid[priority_point]
        update_confidence(confidence, priority_point, patch_size)
        update_data(padded_image, padded_mask, data, gradient, priority_point, patch_size)