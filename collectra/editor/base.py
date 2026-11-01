import json
import webbrowser
from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from pathlib import Path

import yaml
from jinja2 import Environment, PackageLoader, select_autoescape

from collectra.commons.results import pop_header
from collectra.utils import change_dir

__all__ = ["Editor"]


class Editor(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    file: Path
    env: Any = Field(default=None, exclude=True, repr=False)

    def model_post_init(self, __context: Any) -> None:
        self.env = Environment(
            loader=PackageLoader("collectra.editor"), autoescape=select_autoescape()
        )

    def view(self):
        template = self.env.get_template("imagebb.html")
        with change_dir(self.file):
            results = Path("results.yaml")
            if not results.exists():
                raise Exception(f"{self.file} is an invalid file. Exiting...")
            with open(results, "r") as f:
                data = yaml.safe_load(f)
            header = pop_header(data, source=results)
            src_img = ""
            bounding_boxes: list = list()
            text_items: list = list()
            for k, v in data.items():
                if not isinstance(v, list):
                    v = [v]
                for item in v:
                    type_ = item.get("type", "")
                    if type_ in [
                        "collectra.Image",
                        "collectra.ImageCrop",
                        "collectra.types.images.Image",
                        "collectra.types.images.ImageCrop",
                    ]:
                        src_img = item.get("data", src_img)
                        src_img = item.get("path", src_img)
                        keys = list(item.keys())
                        bb_params = set(
                            [
                                "x_center",
                                "y_center",
                                "width_relative",
                                "height_relative",
                            ]
                        )
                        if not bb_params.issubset(keys):
                            continue
                        bb = {bb_param: item[bb_param] for bb_param in bb_params}
                        bb["label"] = k
                        bounding_boxes.append(bb)
                    elif type_ in ["collectra.Text", "collectra.types.texts.Text"]:
                        text = item.get("data", "")
                        text_items.append({"name": k, "text": text})
            # --- Prepare Context for Jinja2 Template ---
            context = {
                "src_img": src_img,
                "bounding_boxes": json.dumps(bounding_boxes),
                "text_items": text_items,
            }

            # --- Render Template ---
            rendered_html = template.render(context)
            viewer_file = Path.cwd() / "viewer.html"
            with open(viewer_file, "w", encoding="utf-8") as f:
                f.write(rendered_html)
            webbrowser.open_new_tab(f"file:///{str(viewer_file)}")
