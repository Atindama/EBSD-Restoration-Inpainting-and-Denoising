import argparse
from distutils.util import strtobool
import os

from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from src.model import PConvUNet as PConvUNetT


def main(args):
    # Define the used device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Define the model
    print("Loading the Model...")
    modelT = PConvUNetT(finetune=False, layer_size=7)
    modelT.load_state_dict(torch.load(args.model, map_location=device)['model'])
    modelT.to(device)
    modelT.eval()

    # Loading Input and Mask
    print("Loading the inputs...")
    orgT = Image.open(args.img)
    orgT = TF.to_tensor(orgT.convert('RGB'))
    maskT = Image.open(args.mask)
    maskT = TF.to_tensor(maskT.convert('RGB'))
    inpT = orgT * maskT

    # Model prediction
    print("Model Prediction...")
    with torch.no_grad():
        inp_ = inpT.unsqueeze(0).to(device)
        mask_ = maskT.unsqueeze(0).to(device)
        if args.resize:
            org_size = inp_.shape[-2:]
            inp_ = F.interpolate(inp_, size=256)
            mask_ = F.interpolate(mask_, size=256)
        raw_outT, _ = modelT(inp_, mask_)
    if args.resize:
        raw_outT = F.interpolate(raw_outT, size=org_size)

    # Post process
    raw_outT = raw_outT.to(torch.device('cpu')).squeeze()
    raw_outT = raw_outT.clamp(0.0, 1.0)
    outT = maskT * inpT + (1 - maskT) * raw_outT

    # Saving an output image
    print("Saving the output...")
    outT = TF.to_pil_image(outT)
    img_nameT = args.img.split('/')[-1]
    outT.save(os.path.join("examples", "out_{}".format(img_nameT)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify the inputs")
    parser.add_argument('--img', type=str, default="examples/img0.jpg")
    parser.add_argument('--mask', type=str, default="examples/mask0.png")
    parser.add_argument('--model', type=str, default="pretrained_pconv.pth")
    parser.add_argument('--resize', type=strtobool, default=False)
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()
    
    main(args)
