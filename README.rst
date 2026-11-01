.. image:: https://unimelbmdap.github.io/collectra/_images/collectra-banner.png

.. start-badges

|testing badge| |coverage badge| |docs badge| |black badge|

.. |testing badge| image:: https://github.com/unimelbmdap/collectra/actions/workflows/testing.yml/badge.svg
    :target: https://github.com/unimelbmdap/collectra/actions

.. |docs badge| image:: https://github.com/unimelbmdap/collectra/actions/workflows/docs.yml/badge.svg
    :target: https://unimelbmdap.github.io/collectra

.. |black badge| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. |coverage badge| image:: https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/rbturnbull/f1d9191993105301fd6f813fe1e659f6/raw/coverage-badge.json
    :target: https://unimelbmdap.github.io/collectra/coverage/

.. end-badges

.. start-quickstart

**This is a pre-release version of Collectra. The API and CLI may change in future releases.**

Collectra builds reusable pipelines for extracting data from collection images.
Define the tasks and their connections in YAML, run them over your images, and
save the resulting crops, text, and classifications with their source relationships.


Installation
==================================

Use Python 3.11–3.14. Create a virtual environment and install from PyPI:

.. code-block:: bash

    python -m venv .venv
    source .venv/bin/activate
    pip install collectra

On Windows, activate the environment with ``.venv\Scripts\activate`` instead.

The default install supports the GUI without Torch. You can inspect existing
results, edit text, and view pipelines containing ML tasks without installing
their model backends. To run Torch-based inference or training (RF-DETR, DETR,
Torchvision/Hugging Face classifiers, or the image orienter), install:

.. code-block:: bash

    pip install "collectra[ml]"

This extra includes Torch, Torchvision, timm, Transformers, RF-DETR and
``ensemble-boxes`` (with its Numba/llvmlite dependencies) for box fusion.
It also includes ``llmloader`` and its provider integrations for executing LLM
tasks. Viewing a pipeline containing LLM tasks does not require ``llmloader``.
Model execution still requires compatible hardware and backend wheels.

YOLO tasks and their annotation previews require the optional ``yolo`` extra:

.. code-block:: bash

    pip install "collectra[yolo]"

The default installation excludes Torch, Ultralytics and drawyolo. The YOLO extra
installs both; Ultralytics is AGPL-licensed and its terms apply where relevant
to combined applications. Making it optional does not remove those obligations.
Other backends and the RLSA line detector do not require this extra.

Collectra's own code remains Apache-2.0. Optional backends and model weights
retain their own licences: in particular, check Surya's version and weights,
and the GUI backend selected by pywebview. See the
`licensing guide <https://unimelbmdap.github.io/collectra/licensing.html>`_
before distributing a pipeline or bundled application.

For a source checkout, ``uv sync`` and ``uv run collectra ...`` use the default
installation without Torch. Add ``--extra ml`` for the Torch backends or
``--extra yolo`` for YOLO. To run the complete test suite, use
``uv sync --group dev --extra ml --extra yolo`` followed by
``uv run --extra ml --extra yolo pytest``.

Alternatively, install the development version directly from GitHub:

.. code-block:: bash

    python -m pip install git+https://github.com/unimelbmdap/collectra.git

Graph rendering also requires the Graphviz system executable (``dot``) on your
``PATH``. Model-based tasks may download weights on their first run.


Install a pipeline command
==================================

If you already have a Collectra pipeline, keep its ``pipeline.yaml`` together
with any model weights and other files it references. Install a launcher for it:

.. code-block:: bash

    collectra --pipeline /path/to/pipeline.yaml install my-pipeline
    my-pipeline --help
    my-pipeline run images --output results
    my-pipeline gui results

This creates a command for an existing pipeline; it does not download a pipeline
or its model files. The launcher is written to ``~/.local/bin`` by default.
Ensure that directory is on your ``PATH``, or choose another destination with
``--bin-dir``. The launcher references the pipeline and Python environment at
their current locations, so recreate it if either moves.

You can also run it directly without installing a launcher:

.. code-block:: bash

    collectra --pipeline /path/to/pipeline.yaml run images --output results

To create your own pipeline, follow the example below.


Design principles and direction
==================================

Collectra aims to help researchers and curators turn collection images and
documents into structured, searchable data while keeping domain expertise
central to the process. Its design is guided by these principles:

* **Compose and reuse tasks.** Connect detection, classification, OCR, and LLM
  tasks into a pipeline suited to your collection. Adapt an existing pipeline
  or implement a custom task in Python.
* **Keep results connected to their sources.** Store artefacts and their
  relationships in result bundles, with the same format for predictions and
  training annotations. RO-Crate compatibility is a goal for these bundles.
* **Make human review part of the workflow.** Inspect and correct predictions
  before using them as research data or training examples.
* **Improve models with reviewed examples.** Fine-tune supported models on
  corrected annotations and repeat the predict, review, and train cycle.
* **Make annotation effort count.** Planned clustering and embedding-based
  browsing will help reveal similar items and prioritise diverse examples.

**Many of these features are still to come.** Today, pipelines are defined in
YAML, the GUI supports reviewing and editing results, and supported models are
trained through the CLI. The visual pipeline builder, training from the GUI,
RO-Crate-compatible packaging, interactive clustering, and diversity-based
annotation ordering remain planned work. Landmark/pose detection, segmentation,
and oriented bounding boxes are also part of the intended task range.

See the `design principles and development direction
<https://unimelbmdap.github.io/collectra/design_principles.html>`_ for more detail.


Build your first pipeline
==================================

A pipeline connects **tasks** (operations) through named **artefacts** (data).
For example, a line detector takes an ``Image`` and produces ``ImageCrop``
artefacts. An OCR task can consume those crops and produce ``Text`` artefacts.
Tasks declare these connections using ``input`` and ``output`` names.

Create a working directory with the following layout:

.. code-block:: text

    my-project/
        pipeline.yaml
        images/
            page001.png
            page002.png

Save this as ``pipeline.yaml``:

.. code-block:: yaml

    printed-pages:
      type: collectra.Pipeline
      ext: collectra
      version: "1.0"

    detect_lines:
      type: collectra.LineDetectorRLSA
      input: page
      output: lines
      deskew: true

The first entry declares the pipeline itself: its key is the pipeline's name,
``type`` is ``collectra.Pipeline`` (or your own subclass of it), and the other
keys (``ext``, ``version``, ``logo``, ``theme``...) are its attributes. Older
files that use a ``collectra_pipeline_metadata:`` block with a ``name:`` inside
still load, but that form is deprecated and gives a warning.

This pipeline detects lines of printed text without loading a pretrained model.
Run the commands below from ``my-project``.

The metadata defines the pipeline's name, version, and result-directory suffix:
``ext: collectra`` produces directories such as ``page001.collectra``. The
``page`` node receives each input image. ``detect_lines`` is the task name used
in CLI commands; ``lines`` is the output artefact name. Collectra infers the
``Image`` input and ``ImageCrop`` output types from the task, so separate
``page`` and ``lines`` definitions are not needed.

Every task needs a ``type`` identifying an importable Python class. Additional
keys, such as ``deskew`` or ``model``, configure that task. Input and output names
can be strings or lists. Connecting another task to ``lines`` makes it run
after the detector. Keep these connections acyclic.


Run a pipeline
==================================

Process one image or a directory of images:

.. code-block:: bash

    collectra --pipeline pipeline.yaml run images/page001.png --output results
    collectra --pipeline pipeline.yaml run images --output results

You can also supply multiple input paths. ``--pipeline`` accepts either a YAML
file or a directory containing ``pipeline.yaml``. Without ``--output``, result
directories are created alongside the input images.

Discover commands and task-specific options with ``--help``:

.. code-block:: bash

    collectra --pipeline pipeline.yaml --help
    collectra --pipeline pipeline.yaml run --help
    collectra --pipeline pipeline.yaml task --help
    collectra --pipeline pipeline.yaml task detect_lines run --help

Add ``--verbose`` to see detailed processing logs, or ``--render`` to generate
workflow diagrams while processing. The commands available under ``task``
depend on the tasks defined in your pipeline.


Inspect and reuse results
==================================

Each input gets a result directory:

.. code-block:: text

    results/
        page001.collectra/
            page001.png
            results.yaml
        page002.collectra/
            page002.png
            results.yaml

``results.yaml`` records the pipeline metadata, artefact types, IDs, data, and
parent relationships. Crops record their bounds and source image; they need
not be separate image files. Other tasks may create additional files in the
same result directory.

Open results in the desktop GUI to inspect the image, crops, and other outputs:

.. code-block:: bash

    collectra --pipeline pipeline.yaml gui results/page001.collectra
    collectra --pipeline pipeline.yaml gui results

The GUI requires a desktop environment and a working pywebview backend. It opens
existing results; run the pipeline first to create them.

Run just one task on existing results:

.. code-block:: bash

    collectra --pipeline pipeline.yaml task detect_lines run results/page001.collectra

To start at a task and continue through its downstream tasks, use:

.. code-block:: bash

    collectra --pipeline pipeline.yaml run results/page001.collectra --task detect_lines

The task's inputs must already exist in the results. A task-level ``run`` runs
only that task; a pipeline-level ``run --task ...`` also runs its downstream
tasks. Running on an existing result directory updates its outputs.


Add OCR and join the text
==================================

To extend the example, append these tasks to ``pipeline.yaml``:

.. code-block:: yaml

    read_lines:
      type: collectra.OCRSuraya
      input: lines
      output: line_text

    join_text:
      type: collectra.ConcatenateText
      input: line_text
      output: transcription
      separator: "\n"

The data flow is now ``page → detect_lines → lines → read_lines → line_text →
join_text → transcription``. OCR runs on each crop; ``ConcatenateText`` collects
the resulting text artefacts and joins them into one transcription.

``OCRSuraya`` is the Collectra class name for the Surya OCR backend. It requires
the ``surya-ocr`` package in your environment and downloads its model weights
on first use. The initial line-detection example does not require this backend.

After installing the OCR backend, process new images with the same ``run``
command, or start OCR on previously detected lines:

.. code-block:: bash

    collectra --pipeline pipeline.yaml run results/page001.collectra --task read_lines


Choose a task
==================================

Built-in task types include:

* ``collectra.LineDetectorRLSA``: detect printed-text lines using image processing.
* ``collectra.LineDetectorSurya``: detect text lines with Surya models.
* ``collectra.OCRSuraya``: recognize text in images or crops.
* ``collectra.OCRHuggingFace``: recognize text with Transformers image-to-text
  models such as DeepSeek-OCR-2 or PaddleOCR-VL, with fine-tuning.
* ``collectra.ConcatenateText``: combine text artefacts.
* ``collectra.ObjectDetectionYOLO``, ``collectra.ObjectDetectionDETR``, and
  ``collectra.ObjectDetectionRFDETR``: produce labelled object crops.
* ``collectra.ImageClassifierYOLO``, ``collectra.ImageClassifierTorchvision``, and
  ``collectra.ImageClassifierHuggingFace``: classify images or crops, producing
  ``Link`` artefacts that retain a reference to the classified image.
* ``collectra.LLM``: process inputs with a configured language model.

Choose model weights trained for the objects or classes in your collection.
Task-specific backends, model configuration, and credentials may be needed;
use the documentation and the configured task's ``--help`` to inspect its options.


Train a classifier
==================================

Trainable tasks expose their own ``train`` command. For example, save this
separate pipeline as ``classifier/pipeline.yaml``:

.. code-block:: yaml

    specimen-classifier:
      type: collectra.Pipeline
      ext: collectra
      version: "1.0"

    classify:
      type: collectra.ImageClassifierTorchvision
      model: resnet18
      input: image
      output: [Pollen, Spore]

Prepare annotated ``*.collectra`` result directories under ``training-data``.
For classification, each label is a ``Link`` named after an output class and
linked to its source image or crop. Set the ``partition`` of the file's
``results`` entry to ``validation`` for held-out examples. A minimal annotated
``results.yaml`` is:

.. code-block:: yaml

    results:
      type: collectra.Results
      partition: validation

    image:
      type: collectra.Image
      id: image-001
      data: specimen.png

    Pollen:
      type: collectra.Link
      id: label-001
      parents: image-001

The first entry, ``results``, describes the file itself (its ``partition``,
``timestamp`` and so on). It is a ``collectra.Results``, which you can subclass to
record more. Older files that begin with a ``collectra_results_metadata:`` mapping
still load, but that form is deprecated and gives a warning; Collectra updates a
file when it next saves it, and ``scripts/upgrade_results_files.py`` updates many
files at once.

Place ``specimen.png`` alongside this file. Provide training examples for each
class as well, using a different partition such as ``training``. Detection
tasks instead train from labelled ``ImageCrop`` annotations.

.. code-block:: bash

    collectra --pipeline classifier/pipeline.yaml task classify train --help
    collectra --pipeline classifier/pipeline.yaml task classify train training-data \
        --validation validation --epochs 10 --batch 16 --workers 0

Training saves the best checkpoint into the pipeline directory and updates the
task's ``model`` entry in ``pipeline.yaml``. Training logs are retained by
default; ``--no-keep-log`` removes the training directory after saving the model.
For this Torchvision classifier, Weights & Biases logging is off by default.
Add ``--wandb`` to opt in.

Use the trained pipeline on new images:

.. code-block:: bash

    collectra --pipeline classifier/pipeline.yaml run images --output classified


Use a pipeline from Python
==================================

The same YAML configuration can be loaded and run from Python:

.. code-block:: python

    from collectra import Collectra

    pipeline = Collectra.from_file("pipeline.yaml")
    pipeline.cli_run(["images"], output="results")

    # Start at a particular task using existing artefacts.
    pipeline.cli_run(
        ["results/page001.collectra"],
        task="detect_lines",
        single=True,
    )

    # Requires Graphviz's dot executable.
    pipeline.render("pipeline-graph", render=True)

``cli_run`` resolves file and directory inputs in the same way as the CLI.
For an explicit list of files, you can also use
``pipeline.run(files=["images/page001.png"], output="results")``.


For more task-specific guidance, see the
`Collectra documentation <https://unimelbmdap.github.io/collectra/>`_.

.. end-quickstart


Building the documentation
==================================

With ``uv`` installed, build the HTML documentation from a source checkout:

.. code-block:: bash

    ./mkdocs.sh

The script supplies Sphinx and its theme in an isolated environment, independent
of any active virtual environment. Open ``docs/_build/html/index.html`` to view
the result. To treat documentation warnings as errors, run
``./mkdocs.sh -W --keep-going``.

Credits
==================================

.. start-credits

- `Robert Turnbull <https://robturnbull.com>`_
- Natalia Orel
- James Quang

Collectra is supported by the
`Enhanced Analytics <https://ardc.edu.au/project/enhanced-analytics-for-hass-and-indigenous-data/>`_
project at the Melbourne Data Analytics Platform. Enhanced Analytics is a
co-investment partnership with the Australian Research Data Commons (ARDC)
through the HASS and Indigenous Research Data Commons
(DOI: `10.3565/6q82-h815 <https://doi.org/10.3565/6q82-h815>`_).
The ARDC is enabled by the Australian Government’s National Collaborative
Research Infrastructure Strategy (NCRIS).

.. image:: https://ardc.edu.au/wp-content/uploads/2022/09/ardc-logo.svg
   :alt: Australian Research Data Commons (ARDC)
   :width: 250px
   :target: https://ardc.edu.au/

.. end-credits
