Python API
==========

Load a YAML pipeline with ``Pipeline.from_file`` and run it from Python:

.. code-block:: python

    from collectra import Pipeline

    pipeline = Pipeline.from_file("pipeline.yaml")
    pipeline.cli_run(["images"], output="results")

``cli_run`` accepts file and directory inputs, resolving them in the same way
as the command line. ``run`` accepts an explicit list of input paths:

.. code-block:: python

    pipeline.run(files=["images/page001.png"], output="results")

Run one task against existing results, using the task name in your pipeline:

.. code-block:: python

    pipeline.cli_run(
        ["results/page001.collectra"],
        task="detect_lines",
        single=True,
    )

Omit ``single=True`` to continue through downstream tasks. The selected task's
input artefacts must already exist in the result directory.

To render the pipeline graph, install the Graphviz system executable and call:

.. code-block:: python

    pipeline.render("pipeline-graph", render=True)

See :doc:`quickstart` for pipeline configuration and :doc:`artefact_display`
for implementing custom artefact displays.
