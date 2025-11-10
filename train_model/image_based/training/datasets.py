from glob import glob

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class ChordDataset(Dataset):
    def __init__(
        self,
        root,
        image_size=(224, 224),
        loader=pil_loader,
    ):
        self.root = root
        self.items = glob(f"{root}/*/*.png")
        self.classes = ["C", "D", "G"]
        self.tfs = transforms.Compose(
            [
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
            ]
        )
        self.loader = loader

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        data = {}

        image_path = self.items[idx]
        image = self.loader(image_path)
        image = self.tfs(image)
        data["image"] = image

        chord = image_path.split("/")[-2]
        if chord == "C":
            chord_label = 0
        elif chord == "D":
            chord_label = 1
        elif chord == "G":
            chord_label = 2
        else:
            raise ValueError(f"Unknown chord label: {chord}")
        data["label"] = chord_label

        return data
