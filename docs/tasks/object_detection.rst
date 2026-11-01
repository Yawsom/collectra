Object Detection
================

Object detectors consume an ``Image`` and produce labelled ``ImageCrop``
artefacts. Their output names should match the labels in the trained model.
Use annotated crops with source-image relationships for training.

.. toctree::
   :maxdepth: 1

   object_detection_yolo
   object_detection_rfdetr
   object_detection_detr

Shared settings
---------------

The three detectors inherit from ``collectra.ObjectDetectionTask``, which
declares the settings they have in common. Give them in the task's
``pipeline.yaml`` entry; ``run`` and ``train`` use them unless an option such as
``--epochs`` or ``--resolution`` is given for that command.

.. code-block:: yaml

    label_detector:
      type: collectra.ObjectDetectionYOLO
      model: label_detector.pt
      input: specimen_sheet
      output: [primary_label, handwritten_data]
      resolution: 1280
      threshold: 0.3
      epochs: 200

``resolution``
    Image size in pixels (default 1280). Training always uses it. Running uses
    it only when you set it, so a model otherwise keeps the size it was trained
    with.
``threshold``
    Minimum confidence for a detection, from 0 to 1 (default 0.25).
``epochs``, ``batch``, ``learning_rate``, ``weight_decay``
    Training settings. Each backend keeps the default that suits its library
    (for example ``batch`` is 16 for YOLO, 4 for RF-DETR and 2 for DETR).

Settings only one backend has, such as YOLO's ``iou``, are listed in its own
page. A setting a detector does not know is reported with a warning, naming the
settings it does understand. The old name ``imgsz`` and YOLO's ``lr0`` are still
accepted, with a warning, as ``resolution`` and ``learning_rate``.

To write your own detector, subclass ``ObjectDetectionTask`` and implement
``_init_model``, ``run`` and ``train``. A detector that can predict several
images at once also implements ``_predict_batch``, and ``prefetch`` then works
for it without more code.

All three detectors expose ``run`` and ``train``. To extract crop images or
evaluate predicted boxes, use the commands on :doc:`../artefacts/image_crop`.
