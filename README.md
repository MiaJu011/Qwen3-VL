<p align="center">
    <img src="https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen3-VL/qwen3-vl.jpg" width="400"/>
<p>
<br>

<p align="center">
        Qwen3-VL <a href="https://huggingface.co/Qwen">🤗</a>&nbsp&nbsp | &nbsp&nbsp Qwen3-VL-Chat Demo <a href="https://modelscope.cn/studios/Qwen/Qwen3-VL">🤖</a>&nbsp&nbsp | &nbsp&nbsp Technical Blog <a href="https://qwenlm.github.io/blog/qwen3-vl/">📖</a>&nbsp&nbsp ｜ &nbsp&nbsp API <a href="https://help.aliyun.com/zh/model-studio/developer-reference/qwen-vl-api">📡</a> 
<br>
<a href="assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp DingTalk (钉钉) <a href="https://qr.dingtalk.com/action/joingroup?code=v1,k1,yKJTJF3c4TYIfDQenNrDIvQO55dfsLaG8exmDLW9tho=&_dt_no_comment=1&origin=11">
        <br>
<a href="https://discord.gg/yPEP2vHTu4">Discord</a>&nbsp&nbsp | &nbsp&nbsp<a href="https://github.com/QwenLM/Qwen3-VL/blob/main/assets/mm_wechat.png">WeChat (微信)</a>
</p>

<p align="center">
<a href="https://opensource.org/license/apache-2-0">License</a>
</p>


[**中文**](README_CN.md) | English

Visit our Hugging Face or ModelScope organization (click links above), search checkpoints with names starting with `Qwen3-VL-` or visit the [official web page](https://qwenlm.github.io/) for more information.

To learn more about Qwen3-VL, feel free to read our [blog](https://qwenlm.github.io/blog/qwen3-vl/)!

---

## News and Updates

* 2025.10.09 🔥 We released the **Qwen3-VL** series, including `Qwen3-VL-2B` and `Qwen3-VL-8B`. `Qwen3-VL-2B` is the first 2B model in the Qwen-VL family. Qwen3-VL series achieves significant improvements in several benchmarks, especially in multilingual capabilities.

---

## Performance

We evaluate Qwen3-VL on a variety of benchmarks, including general VQA, OCR, grounding, document understanding, multilingual, video understanding, agent, and other tasks. Qwen3-VL achieves significant improvements in several benchmarks, especially in multilingual capabilities.

<p align="center">
    <img src="https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen3-VL/performance.jpg" width="800"/>
<p>
<br>


## Requirements

* python 3.8 and above
* pytorch 2.0 and above, 2.4 and above are recommended
* CUDA 11.4 and above are recommended (this is for GPU users, flash-attention users, etc.)
  
## Quickstart

Below, we provide simple examples to show how to use Qwen3-VL with 🤗 Transformers.

Before running the code, make sure you have setup the environment and installed the required packages. Make sure you meet the above requirements, and then install the dependent libraries.

```bash
pip install qwen-vl-utils
```

Now you can start with Transformers. More usage aways for RAG, tools, and others as well as deployment methods can be found in our [documentation](https://qwen.readthedocs.io/en/latest/).

#### 🤗 Transformers

Qwen3-VL can accept various types of visual inputs, including images, videos, and interleaved image-text contents. You can use the following code snippets to quickly start using Qwen3-VL:

##### Image Understanding

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct", torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen3-VL-8B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

<details>
  <summary>With flash_attention_2</summary>

```python
# Requires transformers>=4.40.0
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
```
</details>

##### Video Understanding

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "file:///path/to/video.mp4",
                "max_pixels": 360 * 420,
                "fps": 1.0,
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]
```

##### Batch Inference

We also support batch inference:

```python
messages1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "file:///path/to/image1.jpg"},
            {"type": "image", "image": "file:///path/to/image2.jpg"},
            {"type": "text", "text": "What are the common elements in these pictures?"},
        ],
    }
]
messages2 = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who are you?"},
]
# Preparation for inference
texts = [
    processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    for msg in [messages1, messages2]
]
image_inputs, video_inputs = process_vision_info(messages1)
inputs = processor(
    text=texts,
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

##### Image-Text Interleaved Input

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "file:///path/to/image1.jpg"},
            {"type": "image", "image": "file:///path/to/image2.jpg"},
            {"type": "text", "text": "Identify the similarities between these images."},
        ],
    }
]
```

#### 🤖 ModelScope
We strongly advise users especially those in mainland China to use ModelScope. `snapshot_download` can help you solve issues concerning downloading checkpoints.

#### More Usage Tips

For more usage tips, please refer to our [tutorial](TUTORIAL.md). If you want to learn more about the usage of vLLM, please refer to our [vLLM Usage Guide](vLLM.md).

## Demo

### Web UI

We provide code for users to build a web UI demo (thanks to @wysaid). Before you start, make sure you install the following packages:

```bash
pip install -r requirements_web_demo.txt
```

Then run the command below and click on the generated link:

```bash
python web_demo_mm.py
```

## FAQ

If you meet problems, please refer to [FAQ](FAQ.md) and the issues first to search a solution before you launch a new issue.

## License Agreement

Check the license of each model inside its HF repo. It is NOT necessary for you to submit a request for commercial usage.

## Citation

If you find our work helpful, feel free to give us a cite.

```
@article{Qwen3-VL,
  title={Qwen3-VL: Towards Large-Scale Vision-Language Understanding},
  author={Qwen Team},
  year={2025}
}
```

## Contact Us

If you are interested to leave a message to either our research team or product team, feel free to send an email to qianwen_opensource@alibabacloud.com.

1. related project [DeepSeek-VL2](https://github.com/deepseek-ai/DeepSeek-VL2)
2. related project [Aria](https://github.com/rhymes-ai/Aria)
3. related project [Kimi-VL](https://github.com/MoonshotAI/Kimi-VL)