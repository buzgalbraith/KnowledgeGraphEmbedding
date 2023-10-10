def txt_to_dict(path_to_txt: str) -> None:
    """takes a text file and creates a dictionary file with the same name.

    Args:
        path_to_txt (str) : path to text file to convert.
    """
    path_to_dict = path_to_txt.replace(".txt", ".dict")
    with open(path_to_txt, "r") as f:
        key_value = 0
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line.split("\n") for line in lines]
        with open(path_to_dict, "w") as f:
            for key_value, value in enumerate(lines):
                f.write(f"{key_value}\t{value[0]}\n")
        f.close()


def tsv_to_txt(path_to_tsv: str) -> None:
    """converts .tsv file to .txt.

    Args:
        path_to_tsv (str) : path to tsv file to convert.
    """
    txt_path = path_to_tsv.replace(".tsv", ".txt")
    with open(path_to_tsv, "r") as f:
        lines = f.readlines()
        with open(txt_path, "w") as f:
            for line in lines:
                f.write(f"{line}")
        f.close()


def train_to_train_and_val(path: str, validation_rate: float = 0.8) -> None:
    """takes a training file and splits it into a training and validation file.

    Args:
        path(str) : path to train.txt
        validation_rate(float, optional) : ratio of train to validation that is len(train)/len(train \cup validation)

    """
    assert validation_rate <= 1.0 and validation_rate >= 0.0
    with open(path, "r") as f:
        lines = f.readlines()
        train = lines[: int(len(lines) * validation_rate)]
        val = lines[int(len(lines) * validation_rate) :]
        path_to_train = path.replace("train.tsv", "train.txt")
        path_to_val = path.replace("train.tsv", "valid.txt")
        with open(path_to_train, "w") as f:
            for line in train:
                f.write(f"{line}")
        f.close()
        with open(path_to_val, "w") as f:
            for line in val:
                f.write(f"{line}")
        f.close()


if __name__ == "__main__":
    relations_path = "data/umls/relations.txt"
    entities_path = "data/umls/entities.txt"
    txt_to_dict(relations_path)
    txt_to_dict(entities_path)
    train_path = "./data/umls/train.tsv"
    train_to_train_and_val(train_path, validation_rate=1.0)
    test_path = "./data/umls/test.tsv"
    tsv_to_txt(test_path)
