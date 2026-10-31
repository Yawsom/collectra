"""The viewer must work even when Torch cannot be installed on the machine."""

import subprocess
import sys
import textwrap


def test_gui_and_pipeline_discovery_with_ml_backends_unavailable():
    script = textwrap.dedent('''
        import importlib.abc
        import sys
        import tempfile
        from pathlib import Path

        blocked = {'torch', 'torchvision', 'timm', 'rfdetr', 'transformers', 'onnxruntime', 'ultralytics', 'ensemble_boxes', 'numba', 'llvmlite', 'llmloader', 'langchain_openai', 'sam3'}
        class WithoutML(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in blocked:
                    raise ModuleNotFoundError(fullname, name=fullname.split('.')[0])
        sys.meta_path.insert(0, WithoutML())

        import cappa
        import yaml
        from PIL import Image as PILImage
        from collectra import Pipeline
        from collectra.cli import invoke
        from collectra.gui.backend import GUIBackend
        from collectra.pipelines.ensemble import weighted_boxes_fusion
        try:
            weighted_boxes_fusion([], [], [])
        except ImportError as error:
            assert 'collectra[ml]' in str(error)
        else:
            raise AssertionError('Expected the optional ML dependency message')

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = {'collectra_pipeline_metadata': {'name': 'Viewer', 'ext': 'collectra', 'version': '1'}}
            for name, kind in [
                ('rfdetr', 'ObjectDetectionRFDETR'), ('detr', 'ObjectDetectionDETR'),
                ('yolo', 'ObjectDetectionYOLO'), ('classifier', 'ImageClassifierTorchvision'),
                ('huggingface', 'ImageClassifierHuggingFace'), ('orienter', 'ImageOrienter'),
                ('llm', 'LLM'),
                ('sam3', 'ImageSegmentationSam3'),
                ('seg_rfdetr', 'ImageSegmentationRFDETR'),
                ('seg_yolo', 'ImageSegmentationYOLO'),
            ]:
                config[name] = {'type': 'collectra.' + kind, 'model': 'unused.pt'}
            pipeline_path = root / 'pipeline.yaml'
            pipeline_path.write_text(yaml.safe_dump(config))
            pipeline = Pipeline.from_file(pipeline_path)
            pipeline._ensure_workflow_connected()
            from collectra.tasks.llms import LLM
            try:
                LLM('test', model='dummy').llm
            except ImportError as error:
                assert 'collectra[ml]' in str(error)
            else:
                raise AssertionError('Expected the optional LLM dependency message')
            assert pipeline.gui_graph()['nodes']
            try:
                invoke(pipeline, ['gui', '--help'])
            except cappa.HelpExit:
                pass
            api = GUIBackend(pipeline)
            assert api.get_pipeline_graph()['success']
            page = root / 'page.collectra'
            page.mkdir()
            PILImage.new('RGB', (20, 20), 'white').save(page / 'page.png')
            (page / 'results.yaml').write_text(yaml.safe_dump({
                'image': {'type': 'collectra.Image', 'id': 'image', 'data': 'page.png'},
                'text': {'type': 'collectra.Text', 'id': 'text', 'data': 'original'},
            }))
            assert api.load_yaml(str(page / 'results.yaml'))['success']
            assert api.get_artefact_display('image')['success']
            assert api.get_artefact_display('text')['success']
            assert api.update_display_text('text', 'edited')['success']
            assert yaml.safe_load((page / 'results.yaml').read_text())['text']['data'] == 'edited'
        assert not (blocked & set(sys.modules))
    ''')
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
