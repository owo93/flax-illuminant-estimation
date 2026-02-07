from flax_illuminant_estimation.model import Model


def main() -> None:
    print("Hello from flax-illuminant-estimation!")
    model = Model()
    print(model.display())
