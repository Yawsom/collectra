================================
Command Line Interface Reference
================================

The CLI is built from the tasks and artefacts in the selected pipeline. Supply
``--pipeline`` with a YAML file or a directory containing ``pipeline.yaml``.
Use these commands to inspect the options available for your pipeline::

    collectra --pipeline pipeline.yaml --help
    collectra --pipeline pipeline.yaml run --help
    collectra --pipeline pipeline.yaml task --help
    collectra --pipeline pipeline.yaml artefact --help

Run a pipeline over images, or run one configured task::

    collectra --pipeline pipeline.yaml run images --output results
    collectra --pipeline pipeline.yaml task detect_lines run results/page001.collectra

Replace ``detect_lines`` with a task name from your YAML configuration. Task
commands expose their own help, including backend-specific training options::

    collectra --pipeline pipeline.yaml task classify train --help

See :doc:`quickstart` for complete pipeline and training examples.

Partitioning results
--------------------

Installed pipeline commands expose ``partition`` directly::

    htr partition results --train 80% --validation 20% --seed 42
    htr partition results --train 0.8 --val 0.1 --test 0.1
    htr partition results --partition training=80% --partition heldout=20%

The equivalent generic command is::

    collectra --pipeline pipeline.yaml partition results --train 80% --validation 20%

Inputs may be individual result folders, ``results.yaml`` files, or parent
directories. Parent directories are searched recursively using the pipeline's
result extension. Duplicate results are handled once. Values may be percentages,
fractions or integer counts; use one format throughout, with a total of at most
100%, 1 or the number of results respectively. At least one partition is required.
``--val`` assigns the label ``val``; ``--validation`` assigns ``validation``.
Custom labels use repeated ``--partition NAME=VALUE`` options.

The command updates the ``partition`` of the ``results`` entry (the file's
header) in each selected result. Results left over after the specified allocations keep their existing
partition. Inputs are sorted before shuffling, so a fixed seed reproduces the
same assignment for the same set of results.

Opening results in the GUI
--------------------------

Pass optional positional paths to populate the item list at startup::

    collectra --pipeline pipeline.yaml gui first.collectra second.collectra/results.yaml
    palynomorph gui predictions/*.palynomorph

Inputs can be result folders, files inside a result folder (such as
``results.yaml``), or parent directories containing result folders with the
pipeline's extension. Items retain argument order; parent-directory children
are sorted, and duplicate folders appear only once. The first item opens
automatically. Inputs must contain existing results; opening the GUI does not
create results for raw images.

Without paths, ``gui`` opens with the folder picker as before. ``--debug`` can
be used with or without input paths. The Python equivalent is::

    pipeline.launch_gui("first.collectra", "second.collectra", debug=True)
