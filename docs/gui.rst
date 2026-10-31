GUI
===

The desktop GUI displays existing Collectra results using the selected
pipeline's task and artefact graph. Its scripts, styles, fonts, and image-viewer
controls are bundled locally, so the GUI itself works offline. It requires
pywebview, a supported native
webview backend, and a desktop session.

The default Collectra installation supports the GUI without Torch or the ML
backends. Install with ``pip install collectra`` (or ``uv sync`` in a source
checkout). Existing results and pipeline graphs remain viewable even when the
pipeline contains model tasks. Executing those tasks requires the appropriate
extra: ``collectra[ml]`` for the Torch/RF-DETR/Transformers stack, or
``collectra[yolo]`` for Ultralytics. A pipeline's own custom dependencies must
also avoid requiring Torch if it is to run on a machine without Torch.

Backend licences differ: PyQt is GPLv3 or commercially licensed, even though
pywebview itself is BSD-licensed. See :doc:`licensing` before choosing a backend
for a distributed application.

Visual pipeline construction, training models from the GUI, and interactive
clustering are planned features. See :doc:`design_principles` for the broader
design goals and development direction.

Open results
------------

.. code-block:: bash

    collectra --pipeline pipeline.yaml gui
    collectra --pipeline pipeline.yaml gui results/page001.collectra
    collectra --pipeline pipeline.yaml gui results/page001.collectra/results.yaml
    collectra --pipeline pipeline.yaml gui results
    collectra --pipeline pipeline.yaml gui --help
    collectra --pipeline pipeline.yaml gui --max-pixels 2048

Image and mask previews default to at most 4096 pixels on their longest edge.
``--max-pixels`` sets this limit for the session, including mask editing previews.
It must be a positive integer. The original images and masks stay at full
resolution, and brush positions and sizes are mapped back to source pixels.
The setting is preserved when using the GUI's restart action.

Without paths, select a results folder in the GUI. You can supply multiple
result paths or parent directories. Result directories must use the extension
configured by the pipeline and contain ``results.yaml``. The GUI opens existing
results; it does not run a pipeline over raw images when opening a path.

The GUI remembers the open result folders separately for each pipeline in the
per-user cache directory (on macOS, ``~/Library/Caches/collectra/gui-sessions``).
On the next launch without explicit input paths, it restores that list and opens
the last active folder (or the first if that folder is unavailable). The active
folder is saved in a separate small cache file each time it changes.
Missing or unreadable folders are skipped. Passing paths to
``gui`` replaces the remembered session, as does choosing another folder in the
GUI. The cache stores folder paths only; result data remains in its own files.

An installed pipeline command offers the same interface::

    my-pipeline gui results

macOS Applications launcher
---------------------------

Create a clickable app for a pipeline with the Collectra icon:

.. code-block:: bash

    collectra --pipeline pipeline.yaml install-app --name "My Pipeline"

This creates ``~/Applications/My Pipeline.app``. Double-click it in Finder to
open the GUI without a Terminal window. You can also drag it to the Dock.
The launcher uses the current Python environment and the pipeline's absolute
path, and starts with the pipeline directory as its working directory. Keep
that environment and pipeline at their installed locations.

To open particular results on launch, pass their paths. To choose a destination
or a custom icon, use ``--app-dir`` or ``--icon``:

.. code-block:: bash

    collectra --pipeline pipeline.yaml install-app results --name "My Pipeline"
    collectra --pipeline pipeline.yaml install-app --name "My Pipeline" --app-dir /Applications --icon custom.icns

By default, the launcher uses ``icon`` from ``collectra_pipeline_metadata``
when configured, resolving relative paths from the pipeline directory:

.. code-block:: yaml

    collectra_pipeline_metadata:
      name: My Pipeline
      ext: collectra
      version: "1.0"
      icon: assets/pipeline.png

``--icon`` overrides this setting. If ``icon`` is absent, the pipeline's
``logo`` is used. Without either metadata setting, the Collectra icon is used.
Icons may be ``.icns`` or PNG files; PNGs are converted to macOS icons. The command requires macOS and
refuses to overwrite an existing app unless ``--force`` is supplied.
For example, reinstall with ``install-app --name "My Pipeline" --force``.
The replacement is built before the existing app is replaced. Writing to ``/Applications`` requires write access;
the default ``~/Applications`` is the user's Applications folder.

Native menus
------------

The desktop menu bar provides the same actions as the viewer controls:

* ``File``: open results, reveal the current results file, and close the current tab.
* ``Navigate``: previous/next file, file search, artefact search, and search matches.
* ``Panels``: sidebar, graph, artefact/task lists, fit views, split, collapse, and remove panels.
* ``Layouts``: open the live layout chooser or save the current layout.
* ``Tools``: source control, statistics, and load details.

Panel actions target the last focused panel. macOS retains its standard
application, Edit, and View menus. Restart the GUI to pick up menu changes.

``Tools → Check for Updates…`` checks the pipeline directory and the imported
Collectra source checkout independently. Each must have a ``.git`` directory
or worktree file and a configured remote tracking branch. A packaged Collectra
installation without a source checkout is reported as unavailable.

Checking fetches remote commits without changing working files. When a clean
checkout is behind its remote and can fast-forward, the dialog offers an Update
button. It rechecks the repository before applying the update. Local changes,
diverged branches, missing tracking branches, and fetch failures are reported;
the updater never resets, stashes, or merges local work. Restart the GUI after
updating. Updating source does not install new Python dependencies.
After a successful update, ``Restart Now`` saves pending text edits and the
layout, then restarts the app with the same Python environment and pipeline.
Open results and the selected page are restored from the saved session.

Text editing
------------

Editable text, Markdown, and TEI fields autosave after half a second without
typing. Unchanged content is not written again. Pending edits are also saved
when leaving the editor, and page navigation waits for pending text saves.
Save status and any errors appear in the editor's header. Further edits retry
a failed save.

Inline text is written to ``results.yaml``. File-backed Markdown, TEI, and text
are written to their linked files, preserving the references in ``results.yaml``.

Workflow layouts
----------------

List layout YAML files in the pipeline metadata, with paths relative to the
pipeline directory:

.. code-block:: yaml

    collectra_pipeline_metadata:
      name: My Pipeline
      ext: collectra
      version: "1.0"
      layouts:
        - layouts/default.yaml
        - layouts/review.yaml

The first entry supplies the workflow's default layout when the user has no
valid saved layout. All entries appear under ``<Pipeline name> Layouts`` in
the Customize Layout menu, in metadata order. Later entries are not tried as
fallback defaults. Missing or invalid defaults leave the GUI
using its built-in layout.

User changes auto-save in the user's cache directory under
``collectra/gui-arrangements/<pipeline name>/<pipeline path hash>.yaml``.
On macOS the cache directory is ``~/Library/Caches``. Existing layout files
from the previous flat cache directory are still read, and subsequent saves
use the new directory. A saved user layout takes priority over the workflow
default. Copy a saved layout YAML into the pipeline directory and list it
above to share it as a workflow default. Layout files contain panel splits,
sizes, tabs, focus, and collapsed panels.

The Customize Layout menu also lists named YAML files from the pipeline's
user layout directory under ``Custom Layouts``. Choose ``Save current layout…``
to save a named copy of the current arrangement. Existing names are preserved;
choose a different name to save another copy. Selecting a layout makes it the
current user arrangement and updates the panels in place, retaining the loaded
results, sidebar filter, and selected page. Workflow layout files are never
overwritten by this action.

Sidebar and folder actions
---------------------------

When multiple results are open, the sidebar lists each result folder. Use the
search box above the list to filter folders by name.

Right-click a folder to reveal its ``results.yaml`` in the OS file manager
(Finder on macOS, Explorer on Windows), selected within its result folder.

Source control
--------------

Click the Git icon in the top-right toolbar to open source control. It shows
status and the repository path for the currently open results folder. Git finds
the repository in that folder or an ancestor; linked worktrees are supported.
Open a results folder before using these actions.

* **Status** shows ``git status``.
* **Pull** runs ``git pull --no-edit`` using the configured remote and branch.
* **Add** runs ``git add .`` from the repository root, staging all changes there.
* **Commit** opens a message popup and commits staged changes with that message.
* **Push** runs ``git push`` using the configured remote and branch.

Command output and errors appear in the viewer. Actions run one at a time.
Git must be installed, and remote authentication must already be configured;
terminal credential prompts are disabled. After pulling changes, reopen the
results folder to load the updated files into the viewer.

Inspect and edit artefacts
--------------------------

Select an item and a node in its graph to inspect the corresponding artefacts.
Images and crops display their source imagery, child boxes and coloured child
segmentation masks. Text artefacts
use the relevant text editor, and Links display their referenced targets.
Hovering a child crop or segmentation strengthens its colour. Double-click a
child crop or segmentation to open that instance in an artefact tab.
Image views support box creation and editing. Hold Shift and drag on an image
to draw a new crop using the annotation popup. Ordinary dragging pans the
image; Escape cancels drawing. Shift-drag follows the toolbar drawing button's
availability rules. The clockwise rotation icon
rotates the displayed image by 90 degrees and saves its orientation, together
with the orientations of descendant image artefacts, in ``results.yaml``.
Crop coordinates remain relative to the original source image. Other open
image panels refresh to show the new orientations. Edits are saved back to the
results.

For multichannel TIFFs, the viewer uses a preview of the channels. When the
channel count is divisible by three, view controls switch between RGB triplets;
the preview does not rewrite the source image.

Running and training tasks are documented under :doc:`tasks/index`.
Extraction and evaluation commands are documented under :doc:`artefacts/index`.

Debugging and Python usage
--------------------------

Add ``--debug`` to enable the webview's developer tools:

.. code-block:: bash

    collectra --pipeline pipeline.yaml gui results --debug

The Python equivalent is:

.. code-block:: python

    from collectra import Pipeline

    pipeline = Pipeline.from_file("pipeline.yaml")
    pipeline.launch_gui("results/page001.collectra", debug=True)

Custom displays
---------------

.. toctree::
   :maxdepth: 1

   artefact_display
