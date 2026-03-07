import os
from openai import OpenAI

api_key = os.getenv('ARK_API_KEY')

client = OpenAI(
    base_url="https://ark-cn-beijing.bytedance.net/api/v3",
    api_key=api_key,
)

response = client.responses.create(
    model="ep-20260304201856-twtk9",
    input=[
        {
            "role": "user",
            "content": [

                {
                    "type": "input_image",
                    "image_url": "https://ark-project.tos-cn-beijing.volces.com/doc_image/ark_demo_img_1.png"
                },
                {
                    "type": "input_text",
                    "text": "Doubao-Seed-2.0-lite，Doubao-Seed-1.8，Doubao-1.5-vision，这几个模型给出排名和理由"
                },
            ],
        }
    ]
)

print(response)