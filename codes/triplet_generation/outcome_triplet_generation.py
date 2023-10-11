import os
import re


def generate_outcome_triplets(path: str) -> None:
    """Takes text file with outcomes and patient id and generates triplets.
    Args:
        path (str): Path to text file with outcomes and patient id.
    Returns:
        None
    """
    outcome_triplets_path = path.replace("patient_outcome.txt", "outcome_triplets.txt")
    read_outcomes = []
    with open(path, "r") as f:
        for count, line in enumerate(f):
            if count % 2 == 0:
                read_outcomes.append(line)
            else:
                read_outcomes[-1] += line
    split_outcomes = [outcome.split("\t") for outcome in read_outcomes]
    pattern = "The overall survival status of the patient was (\w+)"
    outcome_triplets = []
    for outcome in split_outcomes:
        if len(outcome) > 1:
            try:
                outcome_triplets.append(
                    [outcome[0], "OUTCOME\t", re.search(pattern, outcome[1]).group(1)]
                )
            except:
                raise ValueError("The pattern did not match the outcome sentence")
    with open(outcome_triplets_path, "w") as f:
        for triplet in outcome_triplets:
            f.write("\t".join(triplet) + "\n")
    f.close()


if __name__ == "__main__":
    cwd = os.getcwd()
    outcomes_path = os.path.join(cwd, "data/patient_outcome.txt")
    outcome_triplets = generate_outcome_triplets(outcomes_path)
