import torch

from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig, TextStreamer, GenerationConfig)
from peft import PeftModel, PeftConfig


class LoadLlm:
    """
    LLM 로딩 객체

    """
    def __init__(self):
        BASE_MODEL = 'kyujinpy/KoT-platypus2-13B'
        PEFT_MODEL_NAME = 'models/llama2-ft-edit-model'
        self.config = PeftConfig.from_pretrained(PEFT_MODEL_NAME)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=False,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.config.base_model_name_or_path = BASE_MODEL
        self.model = AutoModelForCausalLM.from_pretrained(self.config.base_model_name_or_path, quantization_config=bnb_config)
        self.model = PeftModel.from_pretrained(self.model, PEFT_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name_or_path)
        self.streamer = TextStreamer(self.tokenizer)

    def gen(self, x):
        generation_config = GenerationConfig(
            temperature=0.9,
            top_p=0.9,
            top_k=100,
            max_new_tokens=1024,
            early_stopping=True,
            do_sample=True,
        )
        q = f"### instruction: {x}\n\n### Response: "
        gened = self.model.generate(
            **self.tokenizer(
                q,
                return_tensors='pt',
                return_token_type_ids=False
            ),
            generation_config=generation_config,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            streamer=self.streamer,
        )
        result_str = self.tokenizer.decode(gened[0])

        start_tag = f"\n\n### Response: "
        start_index = result_str.find(start_tag)

        if start_index != -1:
            result_str = result_str[start_index + len(start_tag):].strip()
            result_str = result_str.replace('</s>', '')
        return result_str
    
if __name__ == '__main__':
    llm = LoadLlm()
    llm.gen('설치 명령어 확인: pip install -i https://test.pypi.org/simple/ bitsandbytes 명령어는 Test PyPI에서 패키지를 설치하라는 것을 의미합니다. 정식 PyPI에서 최신 버전의 패키지를 설치해야 할 수 있으므로, pip install bitsandbytes를 사용해보세요.')