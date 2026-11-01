ObjectDetectionYOLO
===================

Uses Ultralytics YOLO weights to detect objects and return labelled image crops.
Install its optional dependencies with ``pip install "collectra[yolo]"``.
For example, configure a detector trained to identify specimen labels:

.. code-block:: yaml

    detect_labels:
      type: collectra.ObjectDetectionYOLO
      model: models/labels.pt
      input: image
      output: [label]

Replace the checkpoint path and output labels with those for your model.

Commands
--------

* ``run INPUTS...``: detect objects. In addition to the shared run options,
  ``--model`` overrides the checkpoint, ``--resolution`` sets the image size and
  ``--threshold`` the minimum confidence.
* ``train INPUTS...``: train on annotated result directories. Supports model,
  epoch, batch, resolution, learning-rate, optimizer, augmentation, device, and checkpoint options.

.. code-block:: bash

    collectra --pipeline pipeline.yaml task detect_labels run images --output results
    collectra --pipeline pipeline.yaml task detect_labels run images --model models/other.pt --resolution 1280
    collectra --pipeline pipeline.yaml task detect_labels train training-data --epochs 20 --validation validation
    collectra --pipeline pipeline.yaml task detect_labels train --help

Model-specific options and their defaults are described in ``run --help`` and
``train --help``.

Multiview feature fusion
-----------------------

The task command ``adapt-feature-fusion`` adapts a pretrained RGB detection checkpoint
for spatially registered RGB views. Every view passes through shared backbone
weights. Reference-conditioned attention combines the backbone feature maps
feeding the neck, including skip connections; the existing neck and detection
head then run once. Zero-initialized residual gates preserve the selected RGB
view's initial predictions in evaluation mode.

.. code-block:: bash

    collectra --pipeline pipeline.yaml task detect_labels adapt-feature-fusion yolo11n.pt 5 --reference-view 0 --output models/yolo11n-fusion.pt

Set the task's model to the resulting checkpoint and use the usual Collectra
training command. Collectra retains the fusion architecture during training.
Reference indices are zero-based. Input TIFFs must contain aligned uint8 RGB
triplets ``R1,G1,B1,R2,G2,B2,...``. Direct tensor prediction accepts BCHW floats
in ``[0,1]``. Dataset YAML must declare ``channels: 3*num_views``; Collectra
prepares this automatically from image headers.

From Python, use ``load_adapted_model`` and ``save_adapted_model`` from
``collectra.tasks.object_detection.yolo_feature_fusion``. Native ``.pt`` files
contain importable custom model classes, so this Collectra installation is
required when loading them. Use the helper or Collectra for training; ordinary
Ultralytics reconstruction is deliberately rejected to prevent loss of fusion
weights. The implementation supports standard RGB YOLO detection backbones
starting with ``Conv``; YOLOv8, YOLO11 and YOLO26 graphs are tested. It currently
supports one training device and native PyTorch prediction, not exported
runtimes, profiling or embedding extraction.

Backbone compute and activation memory grow with view count. Shared BatchNorm
uses all views during training, so exact reference preservation applies in
evaluation mode. Prediction can fold BatchNorm into convolutions; reload an
unfused checkpoint before subsequent training. Compare performance against
both RGB and the existing input-channel expansion on the same validation split.

The command reloads and validates the saved checkpoint by default; use
``--no-validate`` to skip that check. Use ``--overwrite`` to replace an existing
output. Outputs must differ from the input checkpoint. Without ``--output``,
the checkpoint is saved beside the input as ``<name>-<N>views-fusion.pt``.
The command leaves the configured task model unchanged.
