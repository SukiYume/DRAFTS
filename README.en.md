# DRAFTS

DRAFTS is a deep-learning FRB/single-pulse search workspace. The current
pipeline uses a CenterNet detector and a ConvNeXt binary classifier for
training-data generation, model training, real FAST observation searches, and
raw8/packed2 injection studies.

The original DRAFTS publication and public assets are available here:

- Paper: [DRAFTS: A Deep Learning-Based Radio Fast Transient Search Pipeline](https://arxiv.org/abs/2410.03200)
- Training data: [TorchLight/DRAFTS on Hugging Face](https://huggingface.co/datasets/TorchLight/DRAFTS)
- Published models: [TorchLight/DRAFTS on Hugging Face](https://huggingface.co/TorchLight/DRAFTS)
- Detailed Chinese documentation: [README.md](README.md)

## Repository layout

| Path | Purpose |
|---|---|
| `generate_burst/` | Inject simulated FRBs into FAST backgrounds and generate CenterNet time-DM H5 training sets. |
| `object_detection/` | Train and evaluate the current CenterNet detector. |
| `binary_classification/` | Train and evaluate ConvNeXt binary candidate filters. |
| `runcode/` | Deployable real-observation search entry points and benchmark tools. |
| `injection_experiment/` | Generate raw8/packed2 injections, launch searches, match truth, aggregate metrics, and build paper figures. |
| `output/` | Local review outputs. Only its documentation is versioned. |

The previous `BinaryClass`, `ObjectDet`, `RunCode`, and `CheckRes` layout has
been replaced by these lowercase workflow directories. Runtime imports and
server deployment scripts rely on the current directory boundaries, so each
workflow remains self-contained.

## Installation

Install the common Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Install PyTorch and CuPy using packages compatible with the target machine's
CUDA driver. The production search environment may instead use the explicit
Conda/PyTorch commands documented in `runcode/requirements.txt`.

## Data and model policy

Git tracks source code, shell entry points, documentation, manuscript sources,
lightweight configuration, and compact evaluation summaries. It deliberately
does not track:

- raw FITS observations or generated H5/NumPy datasets;
- model checkpoints and exported runtimes;
- training logs, search outputs, batch state, or reproducible result tables;
- locally fetched literature full text or LaTeX build products.

Download published data and models from the links above, or place local
artifacts in the paths documented by each workflow README. The ignore rules
keep those files available locally without adding hundreds of gigabytes to
Git history.

## Quick validation

From the repository root:

```bash
python -m compileall -q \
  generate_burst object_detection binary_classification runcode \
  injection_experiment
```

See the README in each workflow directory for commands, model naming, output
contracts, and deployment notes.

## License

MIT; see [LICENSE](LICENSE).
