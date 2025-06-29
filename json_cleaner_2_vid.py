import shotstack_sdk as shotstack
from shotstack_sdk.api import edit_api
from shotstack_sdk.model.video_asset import VideoAsset
from shotstack_sdk.model.clip import Clip
from shotstack_sdk.model.track import Track
from shotstack_sdk.model.timeline import Timeline
from shotstack_sdk.model.output import Output
from shotstack_sdk.model.edit import Edit

from json import load
import logging

logger = logging.getLogger("JSON_CLEANER_v2")

configuration = shotstack.Configuration(host="https://api.shotstack.io/stage")
configuration.api_key["DeveloperKey"] = "Lrb0qJF3qsJGvE9bEzLHtt5fC4bgvkgSSE91NElI"

_gemini_out_file = open("shotstack_hp.json")
gemini_output = load(_gemini_out_file)
_gemini_out_file.close()


def list_to_dict(_list: list) -> dict:
    return_dict = {}
    for i in range(len(_list)):
        return_dict[_list[i]] = _list[i + 1]
    return return_dict


for edit in gemini_output:
    if "parameters" in edit:
        edit_parameters = list_to_dict(edit.parameters)
    match edit.edit_type:
        case "color_correction":
            pass
        case _:
            logger.error(f"Unexpected edit_type: {edit.edit_type}")
