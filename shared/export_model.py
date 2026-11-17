# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "torch==2.15.0",
#     "torchvision==0.29.2",
#     "onnx==1.23.2",
#     "onnxscript==0.7.2",
#     "scikit-image==0.26.0",
#     "pillow==13.0.0",
# ]
# ///
# Run once: export MobileNetV2 and ResNet50 (ImageNet weights) to ONNX, and write the labels and test images.
from pathlib import Path

import torch
from PIL import Image
from skimage import data
from torchvision.models import MobileNet_V2_Weights, ResNet50_Weights, mobilenet_v2, resnet50

here = Path(__file__).parent
model_dir = here / "model"
model_dir.mkdir(exist_ok=True)
labels_path = model_dir / "labels.txt"
cat_path = here / "cat.png"
jpg_path = here / "cat.jpg"
large_path = here / "cat-large.jpg"


def export(model, path):
    dummy = torch.randn(1, 3, 224, 224)
    program = torch.onnx.export(
        model.eval(),
        (dummy,),
        dynamo=True,
        input_names=["input"],
        output_names=["logits"],
    )
    program.save(path, external_data=False)


models = {
    model_dir / "mobilenet_v2.onnx": mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2),
    model_dir / "resnet50.onnx": resnet50(weights=ResNet50_Weights.IMAGENET1K_V2),
}
for path, model in models.items():
    export(model, path)

categories = MobileNet_V2_Weights.IMAGENET1K_V2.meta["categories"]
assert categories == ResNet50_Weights.IMAGENET1K_V2.meta["categories"]
labels_path.write_text("\n".join(categories) + "\n")

# Test images: the original PNG, a quality-90 JPEG of the same picture, and an ENLARGED version
# at 1920 px wide (Lanczos), also quality-90 JPEG. No EXIF/ICC, so the output is deterministic.
cat = Image.fromarray(data.chelsea())
cat.save(cat_path)
cat.save(jpg_path, quality=90, subsampling="4:2:0")
w, h = cat.size
large = cat.resize((1920, round(h * 1920 / w)), Image.LANCZOS)
large.save(large_path, quality=90, subsampling="4:2:0")

for path in models:
    print(f"model : {path} ({path.stat().st_size / 1e6:.2f} MB)")
print(f"label : {labels_path} ({len(categories)} classes)")
for path in (cat_path, jpg_path, large_path):
    with Image.open(path) as img:
        print(f"image : {path} ({img.width}x{img.height}, {path.stat().st_size / 1e3:.0f} KB)")
