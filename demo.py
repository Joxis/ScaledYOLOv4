import torch
import cv2 as cv

from models.experimental import attempt_load
from utils.general import check_img_size


def main(weights):
    imgsz = 640
    device = torch.device("cpu")
    half = device.type != 'cpu'  # half precision only supported on CUDA
    image_np = cv.imread("data/images/result.png")
    image = torch.from_numpy(image_np).unsqueeze(dim=0)
    image = image.transpose(3, 1).byte()
    print(image.dtype)
    image = image.to(device, non_blocking=True)
    image = image.half() if half else image.float()  # uint8 to fp16/32
    image /= 255.0  # 0 - 255 to 0.0 - 1.0

    model = attempt_load(weights, map_location=device)
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    model.eval()
    image = torch.zeros((1, 3, imgsz, imgsz), device=device)
    _ = model(
        image.half() if False else image) if device.type != 'cpu' else None

    out = model(image)
    print(out)


if __name__ == '__main__':
    main("weights/yolov4-p5.pt")
