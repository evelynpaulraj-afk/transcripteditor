import json
import torch

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor
)

from qwen_vl_utils import process_vision_info


MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"


class VisualAnalyzer:

    def __init__(self):

        self.model = (
            Qwen2_5_VLForConditionalGeneration
            .from_pretrained(
                MODEL_NAME,
                torch_dtype="auto",
                device_map="auto"
            )
        )

        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME
        )

    def analyze_images(
        self,
        image_paths,
        shot_start,
        shot_end
    ):

        content = []

        for image_path in image_paths:

            content.append({
                "type": "image",
                "image": image_path
            })

        prompt = f"""
You are analyzing frames from one video shot.

Shot timestamp:
{shot_start:.2f} - {shot_end:.2f}

Extract ONLY visual information useful for video editing.

Return valid JSON:

{{
    "people": [],
    "actions": [],
    "objects": [],
    "visual_content": [],
    "screen_or_slide_content": [],
    "camera_framing": "",
    "editing_relevant_observation": ""
}}

Pay particular attention to:

- graphs
- charts
- slides
- screens
- demonstrations
- products
- objects being discussed
- gestures
- pointing
- changes in camera framing
- visual evidence corresponding to what a speaker might be saying

Do NOT describe irrelevant visual details.

Do NOT invent information.
"""

        messages = [
            {
                "role": "user",
                "content": content + [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(
            messages
        )

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        inputs = inputs.to(self.model.device)

        with torch.no_grad():

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=300
            )

        generated_ids_trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids
            in zip(
                inputs.input_ids,
                generated_ids
            )
        ]

        output = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return output
