Artefact display
================

The GUI calls ``artefact.display(context)`` on the actual artefact instance.
Custom types can inherit an existing display or return a JSON-compatible view
without changing the GUI. The class must be importable using its serialized
``type`` path. If using Collectra's default serialization, export the class from
your package root, or override ``get_class_path`` with its full module path.

The context resolves data paths relative to the results directory; it does not
change the process working directory. ``context.artefact(id)`` resolves another
artefact, and ``context.publish_image(pillow_image)`` creates a PNG data URL.

Supported views
---------------

* ``{"kind": "properties", "data": {...}}``: the base Artefact display,
  rendered as text without interpreting HTML.
* ``{"kind": "text", "format": "plain", "text": "...", "editable": False}``:
  a text box. Formats ``markdown`` and ``xml`` use the existing editors.
  Built-in editable text views identify the artefact with ``target_id``.
* ``{"kind": "image", "source": "data:image/png;base64,..."}``: an
  OpenSeadragon image viewer. ``Image.display`` supplies the source asset path,
  annotation target, and child boxes, and enables crop creation. Custom image
  previews without those capabilities are read-only.

For example, an external Image subclass automatically inherits its viewer::

    from collectra import Image

    class MicroscopeImage(Image):
        pass

A custom artefact can describe a read-only value::

    from collectra.types.base import Artefact

    class Measurement(Artefact):
        value: float = 0.0

        def __call__(self):
            return self.value

        def display(self, context):
            return {
                "kind": "text",
                "format": "plain",
                "text": f"{self.value} mm",
                "editable": False,
            }

Images and annotations
----------------------

ImageCrop inherits Image.display and overrides only ``display_bounds()``.
The bounds are in original-image pixels. The shared display crops and rotates
the preview, and shows immediate child ImageCrop and ImageSegmentation instances
referencing the same source file. Segmentation masks appear as translucent
overlays using their DAG label colours. Stored crop coordinates remain relative to the original image;
the backend converts coordinates when displaying, creating, and editing boxes.
A newly drawn box belongs to the displayed image/crop.

TIFF previews use the TIFF axis-aware reader. For a multispectral TIFF the
default preview uses its first three bands (or first band if fewer than three).
When the channel count is divisible by three, Previous/Next buttons at the top
of OpenSeadragon switch between consecutive RGB triplets. A 75-channel TIFF has
25 views. Click the view counter to choose any view from its dropdown. The
caret buttons step backward and forward, and the controls retain their width
during loading. Switching preserves zoom, pan,
and child bounding boxes, including when viewing a crop or a Link. TIFFs with
one channel per page decode only the three pages needed for the chosen view.
Non-uint8 previews are scaled to 0–255 for display. Override ``display`` to
choose a different projection. Preview generation never rewrites source data.

Link delegates to its resolved target's display, including its edit target.

Text and persistence
--------------------

Inline Text uses a plain text editor. If its serialized data references an
existing file through ``path`` (or a legacy ``data`` reference), the suffix chooses the editor: ``.md``/``.markdown`` for Markdown,
``.xml`` for XML, and plain text otherwise. Node labels do not select the format.
Saving writes to that referenced file and retains its reference in results.yaml.

The GUI retains extra serialized fields of custom artefacts when saving edits.

New browser widgets
-------------------

New Python types can reuse the standard views without JavaScript. For a new
interactive widget, return ``renderer: context.renderer(module_path)`` in the
view descriptor. Collectra loads the JavaScript ES module from the artefact
package and mounts it in the panel. No changes to ``index.html`` or manual
script registration are needed. See :doc:`artefacts/display` for an example,
the renderer lifecycle contract, and the built-in display module structure.
