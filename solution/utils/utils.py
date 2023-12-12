import os
import torch
from typing import Tuple, List
from argparse import ArgumentParser, Namespace
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from imageio import imread


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO make sure this is the path to our custom model pkl
MODEL_FILE = "model.pkl"

SIZE: List[int] = [512, 512]

# TODO this needs to be specific to whatever model we end up using/modifying
INPUT_SIZE_X = 256
INPUT_SIZE = [INPUT_SIZE_X, INPUT_SIZE_X]

MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
STANDARD_DEVIATION: Tuple[float, float, float] = (0.229, 0.224, 0.225)


class SegmentationDataset(Dataset):
    __slots__ = ["root_dir", "image_list", "out_list", "transform"]

    def __init__(self, root_dir, out_dir):
        self.root_dir = root_dir
        self.image_list = [
            os.path.join(self.root_dir, i) for i in os.listdir(root_dir)
        ]
        self.out_list = [
            os.path.join(out_dir, i) for i in os.listdir(root_dir)
        ]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # TODO we may not need this size/interpolation mode
                transforms.Resize(
                    tuple(INPUT_SIZE),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.Normalize(mean=MEAN, std=STANDARD_DEVIATION),
            ]
        )

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        return (
            self.transform(imread(uri=self.image_list[idx])),
            self.out_list[idx],
        )


def load_segmentation_dataset(image_root_dir, image_output_dir) -> DataLoader:
    return DataLoader(
        SegmentationDataset(root_dir=image_root_dir, out_dir=image_output_dir),
        persistent_workers=True,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        prefetch_factor=2,
        pin_memory=True,
    )


def get_parser() -> ArgumentParser:
    programName: str = "LPCV 2023 Tiny Espresso Net"
    authors: list[str] = ["Holden Babineaux, Joseph Fontenot"]

    prog: str = programName
    usage: str = f"This is the {programName}"
    description: str = (
        f"This {programName} does create a single"
        + " segmentation map of areal scenes of disaster environments"
        + " captured by unmanned areal vehicles (UAVs)"
    )
    epilog: str = f"This {programName} was created by {''.join(authors)}"

    return ArgumentParser(prog, usage, description, epilog)


def get_solution_args() -> Namespace:
    parser: ArgumentParser = get_parser()
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Filepath to an image to create a segmentation map of",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Filepath to the corresponding output segmentation map",
    )
    return parser.parse_args()
