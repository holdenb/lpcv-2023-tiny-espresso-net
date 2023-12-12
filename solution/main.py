from argparse import Namespace
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.fanet import FANet
from imageio import imwrite
import pkg_resources
import gc
import time as t
from concurrent.futures import ThreadPoolExecutor


from utils.utils import (
    DEVICE,
    SIZE,
    MODEL_FILE,
    load_segmentation_dataset,
    get_solution_args,
)


def write_image(outArray, output_image_path):
    imwrite(output_image_path, outArray, format="png")


def main() -> None:
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    args: Namespace = get_solution_args()

    with pkg_resources.resource_stream(__name__, MODEL_FILE) as model_file:
        model: FANet = FANet()
        model.to(DEVICE)
        model.load_state_dict(
            state_dict=torch.load(f=model_file, map_location=DEVICE),
            strict=False,
        )
        model.eval()

        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )

        data_loader: DataLoader = load_segmentation_dataset(
            args.input, args.output
        )

        with torch.no_grad():
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

            # Warm up run
            input_id = 0
            for input, filenames in data_loader:
                input = input.to(DEVICE)
                input_id += 1
                if input_id > 100 // 3:
                    break
                _ = model(input)

            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

            time = 0
            # Actual run
            for input, filenames in data_loader:
                input = input.to(DEVICE)

                start.record()
                outTensor: torch.Tensor = model(input)
                end.record()

                torch.cuda.synchronize()
                time += start.elapsed_time(end)

                interp_mode = "bilinear"

                _, _, h, w = outTensor.shape
                while h < 256:
                    h *= 2
                    w *= 2
                    outTensor: torch.Tensor = F.interpolate(
                        outTensor, (h, w), mode=interp_mode, align_corners=True
                    )

                outArray = F.interpolate(
                    outTensor, SIZE, mode=interp_mode, align_corners=True
                ).data.max(1)[1]

                outArray = outArray.cpu().numpy().astype(np.uint8)

                executor = ThreadPoolExecutor(max_workers=8)
                for outData, filename in zip(outArray, filenames):
                    executor.submit(write_image, outData, filename)
                executor.shutdown(wait=True)

        print(time / 1000)

        gc.collect()
        torch.cuda.empty_cache()
        model_file.close()
