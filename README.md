# CoDeX

CoDeX contains learned data compression tools for JAX.

You can use this library to build your own ML models with end-to-end optimized
data compression built in. It's useful to find storage-efficient representations
of your data (images, features, examples, etc.) while only sacrificing a small
fraction of model performance.

For a more in-depth introduction from a classical data compression perspective,
consider our [paper on nonlinear transform
coding](https://arxiv.org/abs/2007.03034), or watch @jonycgn's [talk on learned
image compression](https://www.youtube.com/watch?v=x_q7cZviXkY). For an
introduction to lossy data compression from a machine learning perspective, take
a look at @yiboyang's [review paper](https://arxiv.org/abs/2202.06533).

## Documentation & getting help

Please post all questions or comments on
[Discussions](https://github.com/google/codex/discussions). Only file
[Issues](https://github.com/google/codex/issues) for actual bugs or feature
requests. On Discussions, you may get a faster answer, and you help other people
find the question or answer more easily later.

## Installation

To install CoDeX via `pip`, run the following command:

```bash
pip install jax-codex
```

To test that the installation works correctly, you can run the unit tests with:

```bash
python -m codex.all_tests
```

Once the command finishes, you should see a message ```OK (skipped=29)``` or
similar in the last line.

## Usage

We recommend importing the library from your Python code as follows:

```python
import codex as cdx
```

## Citation

If you use this library for research purposes, please cite:
```
@software{codex_github,
  author = {Ballé, Johannes and Agustsson, Eirikur},
  title = {{CoDeX}: Learned Data Compression in {JAX}},
  url = {http://github.com/google/codex},
  version = {0.0.0},
  year = {2022},
}
```
In the above BibTeX entry, names are top contributors sorted by number of
commits. Please adjust version number and year according to the version that was
actually used.

Note that this is not an officially supported Google product.
