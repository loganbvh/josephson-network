import os
import glob
import json
import shutil
import traceback

from pint import UnitRegistry

ureg = UnitRegistry()


def sort_solutions_by_energy(directory):
    directory = os.path.abspath(directory)
    solutions = {}
    for subdir in os.listdir(directory):
        try:
            int(subdir)
        except Exception:
            continue
        if int(subdir) % 2:
            json_path = os.path.join(directory, subdir, "metadata.json")
            with open(json_path, "r") as f:
                metadata = json.load(f)
            energy = ureg(metadata["energy"]).to("eV").magnitude
            solutions[os.path.dirname(json_path)] = energy
    return {k: v for k, v in sorted(solutions.items(), key=lambda x: x[1])}


def sort_solutions_slurm(job_directory):
    job_directory = os.path.abspath(job_directory)
    json_files = glob.glob(os.path.join(job_directory, "*/*/*/metadata.json"))
    solutions = {}
    for json_path in json_files:
        pardir = os.path.basename(os.path.dirname(json_path))
        if int(pardir) % 2:
            with open(json_path, "r") as f:
                metadata = json.load(f)
            energy = ureg(metadata["energy"]).to("eV").magnitude
            solutions[os.path.dirname(json_path)] = energy
    return {k: v for k, v in sorted(solutions.items(), key=lambda x: x[1])}


def copy_lowest_energy_solution(job_directory, to_directory=None):
    job_id = os.path.basename(job_directory)
    if to_directory is None:
        to_directory = os.path.join(
            os.environ["HOME"],
            "josephson-network",
            "results",
        )
    to_directory = os.path.join(to_directory, job_id)
    solutions = sort_solutions_slurm(job_directory)
    if not solutions:
        print(f"No solutions in {job_directory}.")
        return
    solution, energy = list(solutions.items())[0]
    print(solution, energy)
    print(f"Copying {solution} -> {to_directory}.")
    try:
        shutil.copytree(solution, to_directory)
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(
            os.environ["GROUP_SCRATCH"],
            "josephson-network",
            "results",
        ),
    )
    parser.add_argument(
        "--jobs",
        type=str,
        nargs="+",
    )

    args = parser.parse_args()

    directory = os.path.abspath(args.directory)
    jobs = [val.lower() for val in args.jobs]
    if jobs[0] == "all":
        jobs = os.listdir(directory)
    for job in jobs:
        copy_lowest_energy_solution(os.path.join(directory, job))
