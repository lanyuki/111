import logging
from packaging import version

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
# from lavis.models.blip2_models.modeling_opt import OPTForCausalLM, OPTConfig
from transformers import AutoTokenizer, OPTForCausalLM, OPTConfig,GLMForCausalLM
import transformers
import torch.nn.functional as F
import math
from openai import OpenAI
import json
import os
from zhipuai import ZhipuAI
from tqdm import tqdm


# 初始化ZhipuAI客户端
client = ZhipuAI(api_key="37f403ad7e1e00048a738cc98742eae5.2XvzU7T35ArxyUbK")  # 填写您自己的APIKey
@registry.register_model("blip2_glm")
class Blip2glm(Blip2Base):
    """
    BLIP2 GLM model.
    Supported model types:
        - pretrained_glm: pretrained GLM model
        - caption_coco_glm: finetuned image captioning model with GLM
    """

    def __init__(
            self,
            vit_model="eva_clip_g",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            num_query_token=32,
            glm_model="THU-KEG/GLM-2-6B",
            prompt="",
            max_txt_len=32,
            apply_lemmatizer=False,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()

        # Ensure transformers version is compatible
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.27"), "BLIP-2 GLM requires transformers>=4.27"

        # Initialize tokenizer and visual encoder
        self.tokenizer = self.init_tokenizer()

        # Initialize vision encoder
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        # Initialize query tokens and Qformer
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        # Initialize GLM model and tokenizer
        self.glm_tokenizer = AutoTokenizer.from_pretrained(glm_model, use_fast=False)
        self.glm_model = GLMForCausalLM.from_pretrained(glm_model, torch_dtype=torch.float16)

        for name, param in self.glm_model.named_parameters():
            param.requires_grad = False

        self.eos_token_id = self.glm_tokenizer("\n", add_special_tokens=False).input_ids[0]

        # Projection layer for GLM
        self.glm_proj = nn.Linear(self.Qformer.config.hidden_size, self.glm_model.config.hidden_size)

        # TODO: maybe error from here to ~line102,front optional,a litte dif line114-122
        # self.max_txt_len = max_txt_len
        # self.prompt = prompt
        # prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt") // prompt转tkonen, 这还蛮不一样的
        # self.prompt_length = prompt_tokens.attention_mask.sum(1)
        # self._apply_lemmatizer = apply_lemmatizer
        # self._lemmatizer = None
        # self.voca_size = len(self.glm_tokenizer)
        # # 一步步投到768
        # self.classifier = nn.Linear(768, self.voca_size)


        # Optional: Emotion word handling and projection layers
        self.emo_clss_proj = nn.Linear(2560, 1408)
        self.sub_proj = nn.Linear(1408, 768)

        with open('./emo_cap_dataloader/all_emo_words.txt', 'r') as emo_file:
            self.lines = emo_file.readlines()

        self.emo_words_string = ' '.join([line.strip() for line in self.lines])

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.glm_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

        self.voca_size = len(self.glm_tokenizer)

        # Agglomeration and classification layers
        self.aggfc = nn.Linear(1408 * 2, 1408)
        self.aggfc2 = nn.Linear(1408, 1, bias=False)

    def forward(self, samples):
        image = samples["image"]

        # 图像特征提取
        B, C, T, H, W = image.shape
        image = image.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))

        # 图像加入时间维度，保留通道维度
        sub_cap_feas = image_embeds.view(B, -1, image_embeds.shape[-1])

        image_embeds = image_embeds.view(B, -1, image_embeds.shape[-1])
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        # 情感嵌入
        emo_tokens = self.glm_tokenizer(
            [self.emo_words_string],
            return_tensors="pt",
        ).to(image_embeds.device)
        emo_tokens.input_ids = emo_tokens.input_ids[:, 1:]

        emo_embeds = self.glm_model.model.decoder.embed_tokens(emo_tokens.input_ids)
        emo_embeds = emo_embeds.repeat(image_embeds.shape[0], 1, 1)
        emo_embeds = emo_embeds.type(image_embeds.dtype)

        emo_embeds = self.emo_clss_proj(emo_embeds)
        emo_atts = torch.ones(emo_embeds.size()[:-1], dtype=torch.long).to(
            image_embeds.device
        )

        # q和情感融合
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        emo_class_fea = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=emo_embeds,
            encoder_attention_mask=emo_atts,
            return_dict=True,
        )

        # 图和情感
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        all_em_words = []

        for t in samples["text_input"]:
            unique_words = set()
            for word in t.split():
                word = word.strip('.,').lower()
                if word in self.emo_words_string:
                    unique_words.add(word)

            all_em_words.append(' '.join([line.strip() for line in list(unique_words)]) + "\n")

        em_cls_tokens = self.glm_tokenizer(
            all_em_words,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            add_special_tokens=False,
            max_length=10,
        ).to(image_embeds.device)

        em_mean_feats = emo_class_fea.last_hidden_state.mean(1)
        em_mean_feats = self.classifier(em_mean_feats)
        em_logits = torch.log_softmax(em_mean_feats, dim=1)
        cls_emo = em_logits.view(image_embeds.shape[0], -1, 1).repeat(1, 1, 10)
        cls_loss = F.nll_loss(
            cls_emo.permute(2, 0, 1).reshape(-1, self.voca_size),
            em_cls_tokens.input_ids.transpose(0, 1).contiguous().view(-1),
            ignore_index=self.glm_tokenizer.pad_token_id
        )
        global_k = 5

        sub_cap_feas = self.sub_proj(sub_cap_feas)

        sub_atts = torch.ones(sub_cap_feas.size()[:-1], dtype=torch.long).to(
            image.device
        )

        sub_cap_feas = self.Qformer.bert(
            query_embeds=sub_cap_feas,
            encoder_hidden_states=emo_embeds,
            encoder_attention_mask=emo_atts,
            return_dict=True,
        )
        # 全局特征融合
        d = emo_class_fea.last_hidden_state.shape[-1]
        sorce = torch.bmm(emo_class_fea.last_hidden_state, sub_cap_feas.last_hidden_state.transpose(1, 2)) / math.sqrt(
            d)
        att_weight = F.softmax(sorce, dim=-1)
        global_feats = torch.bmm(att_weight, sub_cap_feas.last_hidden_state)

        query_output.last_hidden_state = torch.cat(
            (global_feats, query_output.last_hidden_state, emo_class_fea.last_hidden_state), dim=1)

        inputs_glm = self.glm_proj(query_output.last_hidden_state)

        atts_glm = torch.ones(inputs_glm.size()[:-1], dtype=torch.long).to(image.device)

        # 文本输入处理
        self.glm_tokenizer.padding_side = "right"
        text = [t + "\n" for t in samples["text_input"]]
        glm_tokens = self.glm_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        targets = glm_tokens.input_ids.masked_fill(
            glm_tokens.input_ids == self.glm_tokenizer.pad_token_id, -100
        )
        if self.prompt:
            targets[:, : self.prompt_length] = -100  # 不计算prompt部分的loss

        # 创建一个空的target以对齐输入和目标
        empty_targets = (
            torch.ones(atts_glm.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.glm_model.model.decoder.embed_tokens(glm_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_glm, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_glm, glm_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.glm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss + cls_loss, "emo_loss": cls_loss}

    @torch.no_grad()
    def generate(
            self,
            samples,
            use_nucleus_sampling=False,
            num_beams=5,
            max_length=30,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=1.0,
            num_captions=1,
            temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        B, C, T, H, W = image.shape

        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        if isinstance(prompt, str):
            prompt = [prompt] * B
        else:
            assert len(prompt) == B, "The number of prompts must be equal to the batch size."

        # For TextCaps
        if "ocr_tokens" in samples.keys() and "{}" in prompt[0]:
            prompt = [p.format(', '.join(samples['ocr_tokens'][i][:30])) for i, p in enumerate(prompt)]

        image = image.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))

            # 图像和情感的变换，有时间要细看一下
            sub_cap_feas = image_embeds.view(B, -1, image_embeds.shape[-1])

            image_embeds = image_embeds.view(B, -1, image_embeds.shape[-1])
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            # 获取情感token
            emo_tokens = self.glm_tokenizer(
                [self.emo_words_string],
                return_tensors="pt",
            ).to(image_embeds.device)

            emo_tokens.input_ids = emo_tokens.input_ids[:, 1:]

            emo_embeds = self.glm_model.model.decoder.embed_tokens(emo_tokens.input_ids)

            emo_embeds = emo_embeds.repeat(image_embeds.shape[0], 1, 1)

            emo_embeds = emo_embeds.type(image_embeds.dtype)

            emo_embeds = self.emo_clss_proj(emo_embeds)

            emo_atts = torch.ones(emo_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

            # 加情感的embedding融合
            emo_class_fea = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=emo_embeds,
                encoder_attention_mask=emo_atts,
                return_dict=True,
            )

            # 输出图像特征
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            # --------------------------------------------------------------------------------
            global_k = 5

            sub_cap_feas = self.sub_proj(sub_cap_feas)

            sub_atts = torch.ones(sub_cap_feas.size()[:-1], dtype=torch.long).to(image.device)

            sub_cap_feas = self.Qformer.bert(
                query_embeds=sub_cap_feas,
                encoder_hidden_states=emo_embeds,
                encoder_attention_mask=emo_atts,
                return_dict=True,
            )

            # global features fusion
            d = emo_class_fea.last_hidden_state.shape[-1]
            sorce = torch.bmm(emo_class_fea.last_hidden_state,
                              sub_cap_feas.last_hidden_state.transpose(1, 2)) / math.sqrt(d)
            att_weight = F.softmax(sorce, dim=-1)

            global_feats = torch.bmm(att_weight, sub_cap_feas.last_hidden_state)

            query_output.last_hidden_state = torch.cat(
                (global_feats, query_output.last_hidden_state, emo_class_fea.last_hidden_state), dim=1)

            # GLM projection
            inputs_glm = self.glm_proj(query_output.last_hidden_state)

            atts_glm = torch.ones(inputs_glm.size()[:-1], dtype=torch.long).to(image.device)

            # GLM tokenizer processing
            glm_tokens = self.glm_tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)

            attention_mask = torch.cat([atts_glm, glm_tokens.attention_mask], dim=1)

            # Get input embeddings for GLM
            inputs_embeds = self.glm_model.model.decoder.embed_tokens(glm_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_glm, inputs_embeds], dim=1)

            # Generate text using GLM model
            outputs = self.glm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

            output_text = self.glm_tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Clean up the generated text
            output_text = [text.strip() for text in output_text]

            return output_text

    def predict_answers(
            self,
            samples,
            num_beams=5,
            inference_method="generate",
            max_len=10,
            min_len=1,
            num_ans_candidates=128,
            answer_list=None,
            prompt="",
            length_penalty=0,
            **kwargs
    ):
        image = samples["image"]
        with self.maybe_autocast():
            # 获取图像嵌入
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            # 获取查询tokens
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            # GLM 投影
            inputs_glm = self.glm_proj(query_output.last_hidden_state)
            atts_glm = torch.ones(inputs_glm.size()[:-1], dtype=torch.long).to(image.device)

            # 设置文本输入
            if isinstance(samples["text_input"], str):
                samples["text_input"] = [samples["text_input"]]
            if prompt:
                text_input = [prompt.format(question) for question in samples["text_input"]]
            else:
                text_input = samples["text_input"]

            # Tokenizer处理
            self.glm_tokenizer.padding_side = "left"
            glm_tokens = self.glm_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)

            attention_mask = torch.cat([atts_glm, glm_tokens.attention_mask], dim=1)

            # 获取GLM的输入嵌入
            inputs_embeds = self.glm_model.model.decoder.embed_tokens(glm_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_glm, inputs_embeds], dim=1)

            # 使用GLM模型生成答案
            outputs = self.glm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=num_beams,
                max_length=max_len,
                min_length=min_len,
                eos_token_id=self.eos_token_id,
                length_penalty=length_penalty,
            )


            output_text = self.glm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_text = [text.strip() for text in output_text]

        # lemmatization
        if self._apply_lemmatizer or ("apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]):
            output_text = self._lemmatize(output_text)

        return output_text

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")  # 这里依然使用ViT模型，除非您也打算替换为GLM的视觉编码器
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")

        # 修改：使用GLM模型的加载配置
        glm_model = cfg.get("glm_model")  # 获取GLM模型的配置参数（例如预训练模型路径或名称）

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        # 修改：将GLM模型传递给构造函数
        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            glm_model=glm_model,  # GLM模型的配置
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
        )

        model.load_checkpoint_from_config(cfg)

        return model
