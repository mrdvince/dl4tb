import argparse

import onnxruntime as ort
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class TchPredictor:
    def __init__(self, model_path, onnx):
        self.labels = ["negative", "positive"]
        if onnx:
            self.sess = ort.InferenceSession(model_path)
        else:
            self.model = torch.jit.load(model_path)

    def pt_predict(self, x):
        preds = F.softmax(self.model(x), dim=1)
        return self.labels[torch.argmax(preds).item()]

    def onnx_predict(self, x):
        ort_inputs = {self.sess.get_inputs()[0].name: self._to_numpy(x)}
        ort_outs = self.sess.run(None, ort_inputs)
        preds = F.softmax(torch.from_numpy(ort_outs[0]), dim=1)
        return self.labels[torch.argmax(preds).item()]

    def _to_numpy(self, tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )


def predict(img_path, model_path, onnx=True):
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose(
        [transforms.Resize([224, 224]), transforms.ToTensor()]
    )
    img_tensor = transform(img).unsqueeze_(0)
    predictor = TchPredictor(model_path, onnx)
    if onnx:
        return predictor.onnx_predict(img_tensor)
    return predictor.pt_predict(img_tensor)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-img", "--img_path", type=str, default="data/tb_data/test/KUTPOASE.png"
    )
    argparser.add_argument("-m", "--model_path", type=str, required=True)
    argparser.add_argument("-o", "--onnx", action="store_true")

    args = argparser.parse_args()
    prediction = predict(args.img_path, args.model_path, onnx=args.onnx)
    print(prediction)
