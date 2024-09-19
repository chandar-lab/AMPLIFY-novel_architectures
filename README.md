# Protein Language Model

Public protein sequence databases contain samples from the fitness landscape explored by nature. Protein language models (pLMs) pre-trained on these sequences aim to capture this landscape for tasks like property prediction and protein design. Standard practice has been to scale pLMs following the successful trend seen in natural language processing. However, the premise that scale leads to better performance assumes that source databases provide accurate representation of the underlying fitness landscape, which is likely false. By developing an efficient codebase, designing a modern architecture, and addressing data quality concerns such as sample bias, we introduce AMPLIFY, a best-in-class pLM that is orders of magnitude less expensive to train and deploy than previous models. Furthermore, to support the scientific community and democratize the training of pLMs, we have open-sourced AMPLIFY's pre-training codebase, data, and model checkpoints.

Link to the pre-print: TBA

## Installation as a Local Pip Package

The repository functions can be built into a Python virtual environment as:

```
python3 -m venv env && \
source env/bin/activate && \
python3 -m pip install --upgrade pip && \
python3 -m pip install --editable $REPO_DIR[dev]
```

Note that `[dev]` includes the necessary dependencies to verify the installation
and build the Sphinx documentation.

Verify the installation is working (GPU required) with:

```
cd $REPO_DIR && python3 -m pytest
```

## Building the Docs

The API documentation is available in Sphinx format.

To build the associated HTML pages, ensure Sphinx is installed
in the currently active Python environment, and run:

```
sphinx-build -M html docs/source/ docs/build/
```

The top-level page is located at: `docs/build/html/index.html`

## Quickstart

### Usage

Build the docs and see `usage.html`, or the `.rst` source at `docs/source/usage.rst` for examples.

### Measuring Similarity to Human Language Text

The package includes a public-facing function `compare_sequences_to_human_text` that reproduces cosine similarities such as those in the "Frankenstein" analysis in the AMPLIFY paper. Given a version of the model and a text
file, it can produce similarity measures between a set of sequences and the text-embedding-average, as in the example below:

```
import amplify

# load the model

config_path = "/local/path/to/model/config/config.yaml"
checkpoint_file = "/local/path/to/model/checkpoint/model.safetensors"

model, tokenizer = amplify.AMPLIFY.load(checkpoint_file, config_path)
model = model.eval()

example_target_sequences = [
    "AACGGEVWVTDEAAAAA",
    "AAAAACGGGVWWTDEAAAAA",
    "AAAADGGVWVTECDA",
]

# calculate the similarities
text_path = "/local/path/to/text_source/example.txt"
similarity_measures = amplify.inference.compare_sequences_to_human_text(
    tokenizer=tokenizer,
    model=model,
    text_path=text_path,
    target_sequences=example_target_sequences,
)
```