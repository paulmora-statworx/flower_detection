# %% Packages

from src.utils.args import get_args
from src.utils.config import process_config
from src.model import OxfordFlower102Model
from src.data_loader import OxfordFlower102DataLoader
from src.trainer import OxfordFlower102Trainer

# %% Main Script


def main():

    args = get_args()
    config = process_config(args.config)

    print("Creating the Data Generator!")
    data_loader = OxfordFlower102DataLoader(config)

    print("Creating the Model!")
    model = OxfordFlower102Model(config)

    print("Creating the Trainer!")
    trainer = OxfordFlower102Trainer(model, data_loader.get_train_generator, config)

    print("Start training the Model!")
    trainer.train()


if __name__ == "__main__":
    main()
