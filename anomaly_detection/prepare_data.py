# fmt: off

"""Convert PSRFITS files into normalized HDF5 time-frequency slices."""

import argparse
import glob
import multiprocessing as mp
import os
import tempfile
import warnings
from contextlib import ExitStack
from datetime import datetime

import h5py
import numpy as np
from astropy.io import fits
from tqdm import tqdm

IMAGE_SIZE         = 512
FITS_SUFFIXES      = {".fit", ".fits"}
DEFAULT_DOWN_TIMES = (8, 16, 32, 64, 128, 256)


def load_fits_file(file_name):
    """Read a PSRFITS SUBINT table and return a (time, frequency) array."""
    try:
        import fitsio

        table, header = fitsio.read(file_name, ext=1, header=True)
    except (ImportError, OSError, RuntimeError):
        with fits.open(file_name, memmap=False) as hdus:
            if len(hdus) < 2 or hdus[1].data is None:
                raise ValueError(
                    f"FITS SUBINT table has no data: {file_name}"
                ) from None
            header = hdus[1].header
            table  = hdus[1].data

    try:
        n_time = int(header["NAXIS2"]) * int(header["NSBLK"])
        n_pol  = int(header["NPOL"])
        n_chan = int(header["NCHAN"])
        data   = table["DATA"].reshape(n_time, n_pol, n_chan)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid PSRFITS DATA layout: {file_name}") from error

    return data[:, : min(2, n_pol), :].mean(axis=1, dtype=np.float32)


def preprocess_slice(data2d):
    """Clip a two-dimensional slice to finite p1-p99 values and scale to [0, 1]."""
    data = np.asarray(data2d, dtype=np.float32).copy()
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D slice, got {data.shape}")

    finite = np.isfinite(data)
    if not finite.any():
        raise ValueError("Slice contains no finite values")
    if not finite.all():
        data[~finite] = np.median(data[finite])

    lower, upper = np.percentile(data, [1, 99])
    if upper <= lower:
        return np.zeros_like(data, dtype=np.float32)

    data = np.clip(data, lower, upper)
    data = (data - lower) / (upper - lower)
    return data.astype(np.float32, copy=False)


def processing_fits_to_slices(data, down_time_list):
    """Generate normalized 512x512 slices for each requested time factor."""
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"Expected (time, frequency) data, got {data.shape}")

    factors = np.asarray(list(down_time_list), dtype=np.int32).reshape(-1)
    if factors.size == 0 or np.any(factors <= 0):
        raise ValueError("down_time_list must contain positive integers")
    factors = np.unique(factors)

    n_time, n_frequency = data.shape
    if n_frequency < IMAGE_SIZE or n_frequency % IMAGE_SIZE:
        raise ValueError(
            f"Frequency dimension {n_frequency} is not divisible by {IMAGE_SIZE}"
        )

    data_512 = data.reshape(n_time, IMAGE_SIZE, n_frequency // IMAGE_SIZE).mean(
        axis=2, dtype=np.float32
    )

    output = []
    for raw_factor in factors:
        factor            = int(raw_factor)
        samples_per_slice = factor * IMAGE_SIZE
        usable_time       = n_time // samples_per_slice * samples_per_slice
        if usable_time == 0:
            continue

        downsampled = (
            data_512[:usable_time]
            .reshape(usable_time // factor, factor, IMAGE_SIZE)
            .mean(axis=1, dtype=np.float32)
        )
        slices = downsampled.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)
        for index in range(slices.shape[0]):
            slices[index] = preprocess_slice(slices[index])
        output.append((factor, slices.astype(np.float32, copy=False)))

    return output


def _collect_fits_files(paths):
    """Collect FITS files from explicit paths, directories, or glob patterns."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]

    files = []
    for raw_path in paths:
        path = os.fspath(raw_path)
        if os.path.isdir(path):
            files.extend(
                candidate
                for candidate in glob.glob(
                    os.path.join(path, "**", "*"), recursive=True
                )
                if os.path.isfile(candidate)
            )
        elif any(character in path for character in "*?["):
            files.extend(glob.glob(path, recursive=True))
        else:
            files.append(path)

    return sorted(
        {
            os.path.abspath(path)
            for path in files
            if os.path.isfile(path)
            and os.path.splitext(path)[1].lower() in FITS_SUFFIXES
        }
    )


def _worker_process_one_file(args):
    file_path, down_time_list = args
    try:
        slices_list = processing_fits_to_slices(
            load_fits_file(file_path), down_time_list
        )
        images_blocks = []
        names_blocks  = []
        down_blocks   = []
        index_blocks  = []

        for down_time, images in slices_list:
            count = images.shape[0]
            if count == 0:
                continue
            images_blocks.append(images)
            names_blocks.append(np.full((count, 1), file_path, dtype=object))
            down_blocks.append(np.full((count, 1), down_time, dtype=np.int32))
            index_blocks.append(np.arange(count, dtype=np.int32).reshape(count, 1))

        if not images_blocks:
            return file_path, None, None, None, None, None

        return (
            file_path,
            np.concatenate(images_blocks).astype(np.float32, copy=False),
            np.vstack(names_blocks),
            np.vstack(down_blocks),
            np.vstack(index_blocks),
            None,
        )
    except Exception as error:
        return file_path, None, None, None, None, repr(error)


def _create_h5_datasets(h5_file):
    h5_file.create_dataset(
        "images",
        shape            = (0, IMAGE_SIZE, IMAGE_SIZE),
        maxshape         = (None, IMAGE_SIZE, IMAGE_SIZE),
        dtype            = np.float32,
        chunks           = (1, IMAGE_SIZE, IMAGE_SIZE),
        compression      = "gzip",
        compression_opts = 4,
        shuffle          = True,
    )
    h5_file.create_dataset(
        "names",
        shape    = (0, 1),
        maxshape = (None, 1),
        dtype    = h5py.string_dtype(encoding="utf-8"),
        chunks   = (1024, 1),
    )
    for name in ("down_time", "slice_idx"):
        h5_file.create_dataset(
            name,
            shape    = (0, 1),
            maxshape = (None, 1),
            dtype    = np.int32,
            chunks   = (1024, 1),
        )


def _append_to_h5(h5_file, images, names, down_time, slice_idx):
    count = images.shape[0]
    if count == 0:
        return

    start = h5_file["images"].shape[0]
    end   = start + count
    for key in ("images", "names", "down_time", "slice_idx"):
        h5_file[key].resize((end, *h5_file[key].shape[1:]))

    h5_file["images"][start:end]    = images
    h5_file["names"][start:end]     = names
    h5_file["down_time"][start:end] = down_time
    h5_file["slice_idx"][start:end] = slice_idx


def build_h5_from_fits_paths(
    fits_paths,
    out_h5_path=None,
    down_time_list=DEFAULT_DOWN_TIMES,
    num_workers=None,
):
    """Build an HDF5 dataset and return its path, factors, and worker count."""
    files = _collect_fits_files(fits_paths)
    if not files:
        raise FileNotFoundError("No FITS files found")

    factors = np.asarray(list(down_time_list), dtype=np.int32).reshape(-1)
    if factors.size == 0 or np.any(factors <= 0):
        raise ValueError("down_time_list must contain positive integers")
    factors = np.unique(factors)

    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8, len(files))
    num_workers = int(num_workers)
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    num_workers = min(num_workers, len(files))

    if out_h5_path is None:
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_h5_path = f"down_data_{timestamp}.h5"
    out_h5_path   = os.path.abspath(out_h5_path)
    output_parent = os.path.dirname(out_h5_path)
    os.makedirs(output_parent, exist_ok=True)

    descriptor, temporary_path = tempfile.mkstemp(
        prefix = f".{os.path.basename(out_h5_path)}.",
        suffix = ".tmp",
        dir    = output_parent,
    )
    os.close(descriptor)
    errors = []
    tasks  = [(path, factors) for path in files]

    try:
        with h5py.File(temporary_path, "w") as h5_file:
            _create_h5_datasets(h5_file)
            with ExitStack() as stack:
                if num_workers == 1:
                    results = map(_worker_process_one_file, tasks)
                else:
                    pool = stack.enter_context(
                        mp.get_context("spawn").Pool(processes=num_workers)
                    )
                    results = pool.imap(_worker_process_one_file, tasks)

                for file_path, images, names, down_time, slice_idx, error in tqdm(
                    results,
                    total = len(tasks),
                    desc  = "Processing FITS",
                    ncols = 100,
                    ascii = True,
                ):
                    if error is not None:
                        errors.append((file_path, error))
                    elif images is not None:
                        _append_to_h5(h5_file, images, names, down_time, slice_idx)

            sample_count                     = h5_file["images"].shape[0]
            h5_file.attrs["down_time_list"]  = factors
            h5_file.attrs["num_workers"]     = num_workers
            h5_file.attrs["num_input_files"] = len(files)
            h5_file.attrs["num_errors"]      = len(errors)
            if errors:
                h5_file.attrs["error_examples"] = np.array(
                    [f"{path} :: {error}" for path, error in errors[:50]],
                    dtype=h5py.string_dtype("utf-8"),
                )

        if sample_count == 0:
            raise ValueError(
                f"No slices were generated from {len(files)} FITS files; "
                f"{len(errors)} files failed"
            )
        os.replace(temporary_path, out_h5_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

    if errors:
        warnings.warn(
            f"Skipped {len(errors)} of {len(files)} FITS files; details are stored "
            "in the HDF5 attributes",
            stacklevel=2,
        )
    return out_h5_path, factors, num_workers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fits_paths",
        nargs = "+",
        help  = "FITS files, directories, or glob patterns",
    )
    parser.add_argument("--output", default=None, help="Output HDF5 path")
    parser.add_argument(
        "--down_times",
        type    = int,
        nargs   = "+",
        default = list(DEFAULT_DOWN_TIMES),
        help    = "Time-downsampling factors",
    )
    parser.add_argument("--num_workers", type=int, default=None)
    arguments = parser.parse_args()

    output_path, used_factors, worker_count = build_h5_from_fits_paths(
        arguments.fits_paths,
        out_h5_path    = arguments.output,
        down_time_list = arguments.down_times,
        num_workers    = arguments.num_workers,
    )
    print(f"Saved: {output_path}")
    print(f"down_time_list: {used_factors.tolist()}")
    print(f"num_workers: {worker_count}")

# fmt: on
