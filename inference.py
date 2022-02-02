import torch
from torchvision import transforms

from data import DataModule
from model import Model


class DltbPredictor:
    def __init__(self, model_path):
        self.model = Model.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.freeze()
        self.processor = DataModule()
        self.softmax = torch.nn.Softmax(dim=1)
        self.labels = ["negative", "positive"]  # self.processor.get_labels()

    def predict(self, x):
        transforms_ = transforms.Compose(
            [transforms.ToTensor(), transforms.CenterCrop(224)]
        )
        output = self.model(transforms_(x).unsqueeze(0))
        preds = self.softmax(output)
        return self.labels[torch.argmax(preds).item()]


if __name__ == "__main__":
    model = DltbPredictor("saved/checkpoints/model_checkpoint_epoch=2.ckpt")
    from PIL import Image

    img = Image.open("data/tb_data/test/CHDDLHBE.png").convert("RGB")
    print(model.predict(img))
