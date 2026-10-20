<h1 align="center">Seismic</h1>
<p align="center">
    <img width="400px" src="imgs/logo.webp" />
</p>

<p align="center">
    <a href="https://dl.acm.org/doi/pdf/10.1145/3626772.3657769"><img src="https://badgen.net/static/paper/SIGIR 2024/green" /></a>
    <a href="https://dl.acm.org/doi/pdf/10.1145/3627673.3679977"><img src="https://badgen.net/static/paper/CIKM 2024/blue" /></a>
    <a href="https://arxiv.org/abs/2501.11628"><img src="https://badgen.net/static/paper/ECIR 2025/yellow" /></a>
    <a href="http://arxiv.org/abs/2404.18812"><img src="https://badgen.net/static/arXiv/2404.18812/red" /></a>
</p>

<p align="center">
    <a href="https://crates.io/crates/seismic"><img src="https://badgen.infra.medigy.com/crates/v/seismic" /></a>
    <a href="https://crates.io/crates/seismic"><img src="https://badgen.infra.medigy.com/crates/d/seismic" /></a>
    <a href="LICENSE.md"><img src="https://badgen.net/static/license/MIT/blue" /></a>
</p>

Seismic is a fast and lightweight search engine for learned sparse embeddings, written in Rust with Python bindings. It indexes sparse vector collections and retrieves results in microseconds with near-exact accuracy.


### Requirements

- **Python** >= 3.8
- **Rust** toolchain (only needed if installing from source for hardware-specific optimizations)

### Installation

The easiest way to use Seismic is via its Python API, which can be installed in two different ways:

1) the easiest way is via pip as follows:
```bash
pip install pyseismic-lsr
```

2) via Rust compilation that allows deeper hardware optimizations as follows (requires a working Rust toolchain, installable via [rustup](https://rustup.rs/)):
```bash
RUSTFLAGS="-C target-cpu=native" pip install --no-binary :all: pyseismic-lsr
```

Check [docs/PythonUsage.md](docs/PythonUsage.md) for more details.


### Quick Start
Given a collection as a `jsonl` file, you can quickly index it by running
```python
from seismic import SeismicIndex

json_input_file = "" # Your data collection

index = SeismicIndex.build(json_input_file)
print("Number of documents:", index.len)
print("Avg number of non-zero components:", index.nnz / index.len)
print("Dimensionality of the vectors:", index.dim)

index.print_space_usage_byte()
```

and then exploit Seismic to retrieve your set of queries quickly

```python
import numpy as np
from seismic import get_seismic_string

string_type = get_seismic_string()  # 'U100': fixed-width dtype for token strings

query = {"a": 3.5, "certain": 3.5, "query": 0.4}
query_id = "0"
query_components = np.array(list(query.keys()), dtype=string_type)
query_values = np.array(list(query.values()), dtype=np.float32)

results = index.search(
    query_id=query_id,
    query_components=query_components,
    query_values=query_values,
    k=10,
    query_cut=3,
    heap_factor=0.8,
)
```

Each document in the `jsonl` file should be a JSON object with an `id` (integer), an optional `content` (string), and a `vector` (dictionary mapping tokens to scores, e.g., `{"dog": 2.45}`). See [docs/RunExperiments.md](docs/RunExperiments.md#data-format) for full format details.


### Features

- **Multiple index variants** — Standard (`SeismicIndex`), compressed ([`SeismicIndexDotPacking8`](examples/DotPacking8Index.ipynb)), and large vocabulary ([`SeismicIndexLV`](examples/LargeVocabulary.ipynb)) for collections with >65K unique tokens
- **RAG-ready** — Build the index with `load_content=True` and retrieve document texts alongside scores ([example](examples/RAG.ipynb))
- **Python & Rust APIs** — Use from Python via `pyseismic-lsr` or integrate directly in Rust via `cargo add seismic` ([docs](docs/RustUsage.md))
- **Parallel batch search** — Multi-threaded query processing via `batch_search`


### Examples

Interactive Jupyter notebooks are available in the [`examples/`](examples/) folder:

- [**HandsOnSeismic.ipynb**](examples/HandsOnSeismic.ipynb) — Quick 2-minute overview of building and querying an index
- [**SeismicGuide.ipynb**](examples/SeismicGuide.ipynb) — Comprehensive guide covering all features: indexing, k-NN graphs, search, evaluation
- [**RAG.ipynb**](examples/RAG.ipynb) — Plug Seismic into a RAG pipeline with document content retrieval
- [**DotPacking8Index.ipynb**](examples/DotPacking8Index.ipynb) — Memory-efficient compressed index variant
- [**LargeVocabulary.ipynb**](examples/LargeVocabulary.ipynb) — Handling collections with large vocabularies (>65K tokens)



### Comparison with the state-of-the-art
Comparison with Dynamic Superblock Pruning ([DSP](https://arxiv.org/pdf/2504.17045)) using the `splade-v3` encoding of the MS MARCO dataset. 


| Index    | MRR@10 | AQT (μs) | Memory (GB) |
|----------|-------:|---------:|------------:|
| DSP      |  40.28 |      745 |        24.0 |
| Seismic  |  40.27 |      185 |         7.9 |

Experiments performed in single-threaded mode on an Intel Core Ultra 7265K CPU, equipped with 124 GB of RAM. 



### Best Results

Seismic is an approximate algorithm designed for high-performance retrieval over learned sparse representations. We provide **pre-optimized configurations** for several common datasets, e.g., MsMarco. Check the detailed documentation in [docs/BestResults.md](docs/BestResults.md) and the available optimized configurations in [experiments/best_configs](experiments/best_configs).


### Resources
Check out our `docs` folder for detailed guides:

- **[PythonUsage.md](docs/PythonUsage.md)** - How to use the Seismic Python API.
- **[RustUsage.md](docs/RustUsage.md)** - How to use Seismic directly in Rust.
- **[Guidelines.md](docs/Guidelines.md)** - Step-by-step guide to build your Seismic index with hyperparameter tuning tips.
- **[BestResults.md](docs/BestResults.md)** - A detailed guide on how to replicate results with optimized configurations.
- **[RunExperiments.md](docs/RunExperiments.md)** - How to run custom experiments, download datasets, and data format details.
- **[TomlInstructions.md](docs/TomlInstructions.md)** - TOML configuration reference.


### Bibliography

<details>
<summary>Click to expand citations</summary>

1. Sebastian Bruch, Franco Maria Nardini, Cosimo Rulli, and Rossano Venturini. "*Efficient Inverted Indexes for Approximate Retrieval over Learned Sparse Representations*." Proc. ACM SIGIR. 2024.
2. Sebastian Bruch, Franco Maria Nardini, Cosimo Rulli, and Rossano Venturini. "*Pairing Clustered Inverted Indexes with κ-NN Graphs for Fast Approximate Retrieval over Learned Sparse Representations*."  Proc. ACM CIKM. 2024.
3. Sebastian Bruch, Franco Maria Nardini, Cosimo Rulli, Rossano Venturini, and Leonardo Venuta. "*Investigating the Scalability of Approximate Sparse Retrieval Algorithms to Massive Datasets*." Proc. ECIR. 2025.
4. Bruch, Sebastian and Fontana, Martino and Nardini, Franco Maria and Rulli, Cosimo and Venturini, Rossano. "*Forward Index Compression for Learned Sparse Retrieval*", ECIR 2025 (*to appear*)

SIGIR 2024
```bibtex
@inproceedings{bruch2024seismic,
  author    = {Bruch, Sebastian and Nardini, Franco Maria and Rulli, Cosimo and Venturini, Rossano},
  title     = {Efficient Inverted Indexes for Approximate Retrieval over Learned Sparse Representations},
  booktitle = {Proceedings of the 47th International {ACM} {SIGIR} {C}onference on Research and Development in Information Retrieval ({SIGIR})},
  pages     = {152--162},
  publisher = {{ACM}},
  year      = {2024},
  url       = {https://doi.org/10.1145/3626772.3657769},
  doi       = {10.1145/3626772.3657769}
}
```

CIKM 2024
```bibtex
@inproceedings{bruch2024pairing,
  author    = {Bruch, Sebastian and Nardini, Franco Maria and Rulli, Cosimo and Venturini, Rossano},
  title     = {Pairing Clustered Inverted Indexes with $\kappa$-NN Graphs for Fast Approximate Retrieval over Learned Sparse Representations},
  booktitle = {Proceedings of the 33rd International {ACM} {C}onference on {I}nformation and {K}nowledge {M}anagement ({CIKM})},
  pages     = {3642--3646},
  publisher = {{ACM}},
  year      = {2024},
  url       = {https://doi.org/10.1145/3627673.3679977},
  doi       = {10.1145/3627673.3679977}
}
```

ECIR 2025
```bibtex
@inproceedings{bruch2025investigating,
  author    = {Bruch, Sebastian and Nardini, Franco Maria and Rulli, Cosimo and Venturini, Rossano and Venuta, Leonardo},
  title     = {Investigating the Scalability of Approximate Sparse Retrieval Algorithms to Massive Datasets},
  booktitle = {Advances in Information Retrieval},
  pages     = {437--445},
  publisher = {Springer Nature Switzerland},
  year      = {2025},
  url       = {https://doi.org/10.1007/978-3-031-88714-7_43},
  doi       = {10.1007/978-3-031-88714-7_43}
}
```


ECIR 2026 (Accepted, to appear)

```bibtex
@article{bruch2026forward,
  title={Forward Index Compression for Learned Sparse Retrieval},
  author={Bruch, Sebastian and Fontana, Martino and Nardini, Franco Maria and Rulli, Cosimo and Venturini, Rossano},
  journal={European Conference on Information Retrieval 2026 (to appear)},
  year={2026}
}
```

Journal of ACM (Under Review)


```bibtex
@article{bruch2025efficient,
  title={Efficient Sketching and Nearest Neighbor Search Algorithms for Sparse Vector Sets},
  author={Bruch, Sebastian and Nardini, Franco Maria and Rulli, Cosimo and Venturini, Rossano},
  journal={arXiv preprint arXiv:2509.24815},
  year={2025}
}
```


### Citation License
The source code in this repository is subject to the following citation license:

By downloading and using this software, you agree to cite the papers listed in the Bibliography section above in any kind of material you produce where it was used to conduct a search or experimentation, whether be it a research paper, dissertation, article, poster, presentation, or documentation. By using this software, you have agreed to the citation license.


</details>


