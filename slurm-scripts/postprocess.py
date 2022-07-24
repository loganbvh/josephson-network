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
    for json_path in sorted(json_files):
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


def copy_lowest_energy_solution_image(
    array_job_directory,
    to_directory=None,
    dry_run=False,
    force=False,
):
    job_id = os.path.basename(array_job_directory)
    if to_directory is None:
        to_directory = os.path.join(
            os.environ["HOME"],
            "josephson-network",
            "results",
        )
    to_directory = os.path.join(to_directory, job_id)
    jobs = []
    for row in os.listdir(array_job_directory):
        try:
            row = int(row)
            jobs.append(row)
        except ValueError:
            pass
    jobs = [str(row) for row in sorted(jobs)]
    row_dirs = []
    for row in jobs:
        d = os.path.join(array_job_directory, row)
        row_dirs.append(os.path.join(d, os.listdir(d)[0]))
    row_dirs = sorted(row_dirs)
    for row, path in zip(jobs, row_dirs):
        outdir = os.path.join(to_directory, row)
        if not dry_run:
            os.makedirs(outdir)
        for i, col in enumerate(sorted(os.listdir(path))):
            copy_lowest_energy_solution(
                os.path.join(path, col),
                to_directory=os.path.join(outdir, str(i)),
                force=force,
                dry_run=dry_run,
            )


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
    parser.add_argument("--image", action="store_true")
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
        if args.image:
            copy_lowest_energy_solution_image(
                job,
                to_directory=args.to,
                dry_run=args.dry_run,
                force=args.force,
            )
        else:
            copy_lowest_energy_solution(
                job,
                to_directory=args.to,
                dry_run=args.dry_run,
                force=args.force,
            )
