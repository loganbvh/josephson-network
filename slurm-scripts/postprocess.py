import os
import glob
import json
import shutil
import traceback

import numpy as np
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
    json_files = glob.glob(
        os.path.join(job_directory, "**/metadata.json"), recursive=True
    )
    solutions = {}
    for json_path in json_files:
        pardir = os.path.basename(os.path.dirname(json_path))
        if int(pardir) % 2:
            with open(json_path, "r") as f:
                metadata = json.load(f)
            energy = ureg(metadata["energy"]).to("eV").magnitude
            if metadata["solver_status"] != "ok":
                energy = np.inf
            elif "optimal" not in "".join(metadata["solver_info"]).lower():
                energy = np.inf
            solutions[os.path.dirname(json_path)] = energy
    return {k: v for k, v in sorted(solutions.items(), key=lambda x: x[1])}


def copy_lowest_energy_solution(
    job_directory,
    to_directory=None,
    dry_run=False,
    force=False,
):
    job_id = os.path.basename(job_directory)
    if to_directory is None:
        to_directory = os.path.join(
            os.environ["HOME"],
            "josephson-network",
            "results",
        )
    to_directory = os.path.join(to_directory, job_id)
    if os.path.isdir(to_directory):
        if force:
            if dry_run:
                print(f"DRY RUN: Directory {to_directory} exists - removing it.")
            else:
                print(f"Directory {to_directory} exists - removing it.")
                shutil.rmtree(to_directory)
        else:
            print(f"Skipping {to_directory} - directory already exists.")
            return
    solutions = sort_solutions_slurm(job_directory)
    if not solutions:
        print(f"No solutions in {job_directory}.")
        return
    print(f"Found {len(solutions)} total solutions.")
    solution, energy = list(solutions.items())[0]
    print(solution, energy)
    if dry_run:
        print(f"DRY RUN: Copying {solution} -> {to_directory}.")
        return
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
    parser.add_argument(
        "--force",
        action="store_true",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    parser.add_argument("--to", type=str, default=None, help="Destination directory.")

    args = parser.parse_args()

    directory = os.path.abspath(args.directory)
    target_jobs = [val.lower() for val in args.jobs]
    if target_jobs[0] == "all":
        jobs = sorted(glob.glob(directory))
    else:
        # Match patterns
        jobs = []
        for job in target_jobs:
            jobs.extend(glob.glob(os.path.join(directory, job)))
    for job in jobs:
        copy_lowest_energy_solution(
            job,
            to_directory=args.to,
            dry_run=args.dry_run,
            force=args.force,
        )
