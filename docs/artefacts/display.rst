Custom artefact displays
========================

The GUI asks the imported artefact instance for ``display(context)``. Rendering
does not depend on the class name. An artefact package on Python's import path
can reuse a built-in view or supply its own JavaScript renderer, without changing
Collectra's GUI or copying files into its installation.

Reusing built-in views
----------------------

Return a JSON-compatible dictionary from ``display(context)``:

* ``{"kind": "text", "format": "plain", "text": "...", "editable": False}``
  shows text. Formats ``markdown`` and ``xml`` are also supported. Set
  ``editable`` only when the artefact supports the text update API.
* ``context.image(self)`` renders an ``Image`` subclass, with zooming, rotation,
  child crop boxes and child segmentation masks.
* ``{"kind": "properties", "data": {...}}`` shows structured properties.
  This is the default for ``Artefact``.

``ImageCrop`` and ``ImageSegmentation`` inherit the image view and render their
own cropped or masked image. ``Link`` delegates to its target's display.

Built-in image previews and mask textures default to a limit of 4096 pixels on their
longest edge to keep browser canvas memory bounded for large manuscript pages.
Use ``gui --max-pixels 2048`` to choose a smaller limit for the session.
JPEG previews use reduced-resolution decoding when possible. This affects only
GUI previews: source images, stored masks, extraction and training retain their
original resolution. Crop overlays use normalized coordinates, while the mask
editor translates preview positions and brush sizes back to source pixels.
Custom image renderers should similarly bound textures rather than allocating
full-resolution canvases for every segmentation.

Providing a renderer
--------------------

For example, place these files in an importable ``my_artefacts`` package.

``measurement.py``:

.. code-block:: python

    from pathlib import Path
    from collectra.types.base import Artefact

    class Measurement(Artefact):
        value: float = 0

        def __call__(self):
            return self.value

        def display(self, context):
            return {
                "kind": "measurement",
                "value": self.value,
                "renderer": context.renderer(
                    Path(__file__).with_name("measurement.js")
                ),
            }

``measurement.js``:

.. code-block:: javascript

    export default function render({element, view, identity, api, signal}) {
        const label = document.createElement('p');
        label.textContent = `${identity.id}: ${view.value}`;
        element.appendChild(label);

        // Return cleanup for subscriptions, timers, charts, etc.
        return () => label.remove();
    }

Reference the class in results as ``my_artefacts.measurement.Measurement``.
Include the JavaScript file as package data when distributing your package.
``context.renderer`` reads that local file and publishes it as an ES module;
it does not require an HTTP server or a browser-accessible filesystem path.
The module must be self-contained: bundle dependencies into it rather than
using relative imports. Use ``context.publish_image(pil_image)`` to include
image data in the view, and ``context.path(value)`` to resolve result-relative
paths on the Python side.

Renderer contract
-----------------

The module exports a default function receiving:

* ``element``: a dedicated DOM container for this rendering. Only modify this
  container; several panels may display artefacts at the same time.
* ``view``: the dictionary returned by Python, with ``target_id`` defaulting
  to the displayed artefact ID.
* ``identity``: the displayed artefact's ``id`` and Python ``type``.
* ``api``: the GUI's ``window.pywebview.api`` bridge.
* ``signal``: an ``AbortSignal`` aborted when the display is cleared, the tab
  changes, or the panel is closed. Check it after asynchronous work.

The function may be asynchronous and may return a synchronous cleanup function.
Cleanup runs once on leaving the display, including when navigation happens
before asynchronous rendering completes. Each rendering has its own container,
so late work cannot replace the next artefact's content. Modules are imported
once per source; keep per-panel state inside the render function, not at module
scope. Errors appear in the panel using the GUI's normal display error handling.

Renderer code runs with the same GUI access as Collectra's built-in code; it is
code supplied by the installed artefact package.

Built-in implementation
-----------------------

``collectra/gui/artefactDisplay.js`` dispatches views and manages custom module
loading and cleanup. Built-in implementations live in ``collectra/gui/displays``:

* ``image.js``: image viewer, zoom, rotation, annotation lifecycle.
* ``imageCrop.js``: crop geometry and drawing controls.
* ``imageSegmentation.js``: coloured instance mask overlays.
* ``imageInteractions.js``: shared child hit testing and double-click navigation.
* ``segmentationEditor.js`` and ``segmentationEditor.css``: dimmed context,
  circular mask brushes and autosave.
* ``text.js``: plain text, Markdown and XML editors, toolbar and autosave.
* ``properties.js``: structured property display.

``index.html`` owns the panel layout and calls these renderers. It does not need
to know about an extension's artefact type or view kind.

Python's shared ``DisplayContext`` delegates built-in descriptor construction
to ``collectra/displays/images.py`` and ``collectra/displays/texts.py``. The
context's public methods remain available to custom artefact types.
``collectra/displays/image_segmentation.py`` supplies mask-editing descriptors;
``collectra/gui/segmentation.py`` validates strokes and writes binary masks
atomically.
