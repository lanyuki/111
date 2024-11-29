"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
from packaging import version

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
# from lavis.models.blip2_models.modeling_opt import OPTForCausalLM, OPTConfig
# TODO:qwen的config还没有导入
from transformers import  OPTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig
import transformers
import torch.nn.functional as F
import math


@registry.register_model("blip2_qwen2")
class Blip2LlamaModel(Blip2Base):

    # 预训练模型导入
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_qwen2.7b": "configs/models/blip2/blip2_instruct_qwen2.7b.yaml",

    }

    def __init__(
            self,
            vit_model="eva_clip_g",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            num_query_token=32,
            qwen_model="llama3.2",  # 使用 Qwen-2 模型名称
            prompt="",
            max_txt_len=32,
            apply_lemmatizer=False,
    ):
        """
        apply_lemmatizer: 当 apply_lemmatizer 设置为 True 时，predict_answers() 函数返回的结果会经过词形还原处理。
        """
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.37")

        # 初始化 tokenizer 和 vision encoder
        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        # 初始化 Q-Former
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        # 使用 Qwen-2 tokenizer 和模型
        self.qwen_tokenizer =AutoTokenizer.from_pretrained(qwen_model)
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            qwen_model,
            torch_dtype="auto",
            device_map="auto"
        )
        for name, param in self.qwen_model.named_parameters():
            param.requires_grad = False
        # TODO:这里是是直接从opt里抄改过来的，可能有错eos_token_id不一定有，所以先写了个判断，如果没有就直接看\n
        # 设置 eos_token_id。如果 Qwen-2 没有明确的 \n 作为结束符号，我们可能需要使用特殊的 Qwen-2 特定结束符
        # self.eos_token_id = self.qwen_tokenizer.eos_token_id if hasattr(self.qwen_tokenizer, "eos_token_id") else \
        # self.qwen_tokenizer(
        #     "\n", add_special_tokens=False).input_ids[0]
        self.eos_token_id = self.qwen_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]
        # TODO:这一块写的也可能出问题，有点简单粗暴的意思了,而且改过来还有点没改好
        # 初始化投影层，将 Q-Former 输出投影到 Qwen-2 的隐层维度
        # qformer_hidden_size = self.Qformer.config.hidden_size
        # qwen_hidden_size = self.qwen_model.config.hidden_size
        # if qformer_hidden_size != qwen_hidden_size:
        #     self.qwen_model= nn.Linear(qformer_hidden_size, qwen_hidden_size)
        # else:
        #     self.opt_proj = nn.Identity()
        self.qwen_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.qwen_model.config.hidden_size
        )
        # 设置最大文本长度和 prompt
        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.qwen_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

        # 其他分类与情感层
        self.voca_size = len(self.qwen_tokenizer)
        self.classifier = nn.Linear(768, self.voca_size)
        self.emo_clss_proj = nn.Linear(2560, 1408)  # 适配情感投影层
        self.sub_proj = nn.Linear(1408, 768)

        # 加载情感词数据
        with open('./emo_cap_dataloader/all_emo_words.txt', 'r') as emo_file:
            self.lines = emo_file.readlines()
        self.emo_words_string = ' '.join([line.strip() for line in self.lines])
        self.flag = False

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
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        # 情感嵌入
        emo_tokens = self.qwen_tokenizer(
            [self.emo_words_string],
            return_tensors="pt",
        ).to(image_embeds.device)

        emo_tokens.input_ids = emo_tokens.input_ids[:, 1:]
        emo_embeds = self.qwen_model.get_input_embeddings()(emo_tokens.input_ids)
        emo_embeds = emo_embeds.repeat(image_embeds.shape[0], 1, 1)
        emo_embeds = emo_embeds.type(image_embeds.dtype)
        emo_embeds = self.emo_clss_proj(emo_embeds)
        emo_atts = torch.ones(emo_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)

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

        em_cls_tokens = self.qwen_tokenizer(
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
            ignore_index=self.qwen_tokenizer.pad_token_id,
        )

        # --------------------------------------------------------------------------------
        global_k = 5

        # sub_cap_feas = self.sub_proj(sub_cap_feas)

        sub_atts = torch.ones(sub_cap_feas.size()[:-1], dtype=torch.long).to(image.device)

        sub_cap_feas = self.Qformer.bert(
            query_embeds=sub_cap_feas,
            encoder_hidden_states=emo_embeds,
            encoder_attention_mask=emo_atts,
            return_dict=True,
        )

        d = emo_class_fea.last_hidden_state.shape[-1]
        sorce = torch.bmm(emo_class_fea.last_hidden_state, sub_cap_feas.last_hidden_state.transpose(1, 2)) / math.sqrt(
            d)
        att_weight = F.softmax(sorce, dim=-1)
        global_feats = torch.bmm(att_weight, sub_cap_feas.last_hidden_state)

        # 三个特征合并
        query_output.last_hidden_state = torch.cat(
            (global_feats, query_output.last_hidden_state, emo_class_fea.last_hidden_state), dim=1
        )

        inputs_qwen = self.qwen_proj(query_output.last_hidden_state)
        atts_qwen = torch.ones(inputs_qwen.size()[:-1], dtype=torch.long).to(image.device)

        self.qwen_tokenizer.padding_side = "right"

        text = [t + "\n" for t in samples["text_input"]]

        qwen_tokens = self.qwen_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        targets = qwen_tokens.input_ids.masked_fill(
            qwen_tokens.input_ids == self.qwen_tokenizer.pad_token_id, -100
        )
        if self.prompt:
            targets[:, : self.prompt_length] = -100  # 不对提示部分计算损失

        empty_targets = (
            torch.ones(atts_qwen.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.qwen_model.get_input_embeddings()(qwen_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_qwen, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_qwen, qwen_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.qwen_model(
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

        # 设置prompt
        # prompt = samples.get("prompt", self.prompt)
        # prompt = [prompt] * B if isinstance(prompt, str) else prompt
        # assert len(prompt) == B, "The number of prompts must be equal to the batch size."

        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        if isinstance(prompt, str):
            prompt = [prompt] * B
        else:
            assert len(prompt) == B, "The number of prompts must be equal to the batch size."

        # TextCaps：将 OCR tokens 插入 prompt
        if "ocr_tokens" in samples and "{}" in prompt[0]:
            prompt = [p.format(', '.join(samples['ocr_tokens'][i][:30])) for i, p in enumerate(prompt)]

        image = image.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))

            # 图像和情感特征提取及融合
            sub_cap_feas = torch.mean(image_embeds, dim=1)
            sub_cap_feas = sub_cap_feas.view(B, T, -1)

            image_embeds = image_embeds.view(B, -1, image_embeds.shape[-1])
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            # 情感嵌入
            emo_tokens = self.qwen_tokenizer([self.emo_words_string], return_tensors="pt").to(image_embeds.device)
            emo_tokens.input_ids = emo_tokens.input_ids[:, 1:]
            emo_embeds = self.qwen_model.model.decoder.embed_tokens(emo_tokens.input_ids).repeat(image_embeds.shape[0],
                                                                                                 1, 1)
            emo_embeds = self.emo_clss_proj(emo_embeds).type(image_embeds.dtype)
            emo_atts = torch.ones(emo_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)

            # 查询 tokens 和图像、情感的融合
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            emo_class_fea = self.Qformer.bert(query_embeds=query_tokens, encoder_hidden_states=emo_embeds,
                                              encoder_attention_mask=emo_atts, return_dict=True)
            query_output = self.Qformer.bert(query_embeds=query_tokens, encoder_hidden_states=image_embeds,
                                             encoder_attention_mask=image_atts, return_dict=True)
            global_k=5
            # 图像和情感的加权融合
            sub_cap_feas = self.sub_proj(sub_cap_feas)
            sub_atts = torch.ones(sub_cap_feas.size()[:-1], dtype=torch.long).to(
                image.device
            )
            sub_cap_feas = self.Qformer.bert(query_embeds=sub_cap_feas, encoder_hidden_states=emo_embeds,
                                             encoder_attention_mask=emo_atts, return_dict=True)
            d = emo_class_fea.last_hidden_state.shape[-1]
            att_weight = F.softmax(
                torch.bmm(emo_class_fea.last_hidden_state, sub_cap_feas.last_hidden_state.transpose(1, 2)) / math.sqrt(
                    d), dim=-1)
            global_feats = torch.bmm(att_weight, sub_cap_feas.last_hidden_state)

            # 最终嵌入拼接
            query_output.last_hidden_state = torch.cat(
                (global_feats, query_output.last_hidden_state, emo_class_fea.last_hidden_state), dim=1)
            inputs_qwen = self.qwen_proj(query_output.last_hidden_state)
            atts_qwen = torch.ones(inputs_qwen.size()[:-1], dtype=torch.long).to(image.device)

            # 获取 Qwen-2 的输入
            qwen_tokens = self.qwen_tokenizer(prompt, return_tensors="pt", padding="longest", truncation=True,
                                              max_length=self.max_txt_len).to(image.device)
            attention_mask = torch.cat([atts_qwen, qwen_tokens.attention_mask], dim=1)

            # 嵌入输入到 Qwen-2
            inputs_embeds = self.qwen_model.get_input_embeddings()(qwen_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_qwen, inputs_embeds], dim=1)

            # 使用 Qwen-2 生成文本
            outputs = self.qwen_model.generate(
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

            # 解码生成的文本
            output_text = self.qwen_tokenizer.batch_decode(outputs, skip_special_tokens=True)
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
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_qwen = self.qwen_proj(query_output.last_hidden_state)
            atts_qwen = torch.ones(inputs_qwen.size()[:-1], dtype=torch.long).to(image.device)

            # 设置文本输入
            if isinstance(samples["text_input"], str):
                samples["text_input"] = [samples["text_input"]]
            text_input = [prompt.format(question) for question in samples["text_input"]] if prompt else samples[
                "text_input"]

            self.qwen_tokenizer.padding_side = "left"
            qwen_tokens = self.qwen_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)

            # 构建注意力掩码和输入嵌入
            attention_mask = torch.cat([atts_qwen, qwen_tokens.attention_mask], dim=1)
            inputs_embeds = self.qwen_model.get_input_embeddings()(qwen_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_qwen, inputs_embeds], dim=1)

            # 使用 Qwen-2 生成答案
            outputs = self.qwen_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                eos_token_id=self.eos_token_id,
                length_penalty=length_penalty,
            )
            output_text = self.qwen_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_text = [text.strip() for text in output_text]

        # 如果需要，应用词形还原
        if self._apply_lemmatizer or ("apply_lemmatizer" in samples and samples["apply_lemmatizer"]):
            output_text = self._lemmatize(output_text)

        return output_text

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

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
        # 获取配置中的各个参数
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        qwen_model = cfg.get("qwen_model", "qwen2")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")  # 精度设为 fp16，以节省内存
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        # 初始化模型类
        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            qwen_model=qwen_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
        )

        # 加载模型检查点
        model.load_checkpoint_from_config(cfg)

        return model
