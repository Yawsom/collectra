LLM
===

``collectra.LLM`` sends image and/or text artefacts to a language model and
returns ``Text``. Configure a model identifier supported by your ``llmloader``
installation and the credentials required by its provider.

.. code-block:: yaml

    summarize:
      type: collectra.LLM
      model: YOUR_MODEL_IDENTIFIER
      input: transcription
      output: summary
      template: "Summarize this transcription: {transcription}"
      temperature: 0.2

Replace ``YOUR_MODEL_IDENTIFIER`` before using this example. Template
placeholders refer to input artefact names. ``prompt`` is accepted in place of
``template``, as in other tasks; set only one of them. For image inputs, the selected
model must support images. ``system``, ``preamble``, and ``max_tokens`` can also
be configured in YAML.

Several outputs
---------------

List more than one ``output`` node and the model answers field by field, one
string per node, named exactly as the node. Every node gets a ``Text``; a field
the model did not find is an empty string. Describe each field in the prompt
template. The model must support tool calling, because that is how the
structured answer is requested.

.. code-block:: yaml

    label_reader:
      type: collectra.LLM
      model: YOUR_MODEL_IDENTIFIER
      input: label_text
      output: [genus_draft, age_draft]
      template: |
        Read the genus and the geological age from this label.
        {label_text}

Looking things up
-----------------

An ``EntityLookup`` is a named list of known values that the model can search
before it answers. It is declared at the top level, not as a graph node, and a
task names it under ``tools``:

.. code-block:: yaml

    genera_lookup:
      type: collectra.EntityLookup
      file: genera.txt              # one entity per line, relative to the pipeline
      description: Genus names of graptolites; not exhaustive

    genus_reader:
      type: collectra.LLM
      model: YOUR_MODEL_IDENTIFIER
      input: [label_text, genus_image]
      output: genus
      tools: [genera_lookup]
      template: |
        Field: genus
        {label_text}
        {genus_image}

The model calls the lookup with the text it read, or a variant spelling, and
gets back up to ``limit`` candidates (default 10), best first, each with a
relative score; an exact match is marked. It can look up more than once, up to
``max_tool_calls`` times per prompt (default 5). Matching is BM25 over words and
character n-grams, so a misspelling still finds the entry and a single word finds
a longer entry that contains it. The index is built in memory; there are no index
files. A name in ``tools`` that no resource declares is reported when the
pipeline loads. Tools need a model that supports tool calling. Tools can be
combined with several outputs: the model searches as it needs to, then gives its
answer by calling one last tool whose fields are the output nodes, so one call
can read a whole label and check each field against its own list. With
``--batch-size``, prompts run their lookups side by side.

Commands
--------

``run`` is the only task command. There is no training or chat command.
It accepts the :doc:`shared run options <index>`; add ``--usage`` to record
token-usage information supplied by the backend.

.. code-block:: bash

    collectra --pipeline pipeline.yaml task summarize run results/page001.collectra --usage
    collectra --pipeline pipeline.yaml task summarize run --help

The input ``transcription`` must already exist. To generate it and the summary
in one pass, run the full pipeline. See :doc:`canonicalisation` for tasks that
combine retrieval with an LLM.
