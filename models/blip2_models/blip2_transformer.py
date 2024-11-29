"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
from packaging import version
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train

from transformers import AutoTokenizer, OPTForCausalLM, OPTConfig
# import modelscope
# from modelscope import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import transformers
import torch.nn.functional as F
import math

from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import torch.nn as nn
import logging


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, num_layers=6, num_heads=8, ff_hid_dim=2048):
        """
        A simple transformer decoder module.

        Args:
            vocab_size (int): The size of the input vocabulary.
            hidden_size (int): The size of the hidden representation (default 768).
            num_layers (int): The number of transformer decoder layers (default 6).
            num_heads (int): The number of attention heads in each layer (default 8).
            ff_hid_dim (int): The hidden dimension of the feedforward network (default 2048).
        """
        super(TransformerDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.positional_encoding = nn.Parameter(torch.randn(1, 512, hidden_size))  # Max 512 positions

        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=ff_hid_dim,
                dropout=0.1
            )
            for _ in range(num_layers)
        ])

        self.output_projection = nn.Linear(hidden_size, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Forward pass through the transformer decoder.

        Args:
            tgt (Tensor): The target sequence (batch_size, tgt_len).
            memory (Tensor): The memory (output from encoder) (batch_size, src_len, hidden_size).
            tgt_mask (Tensor, optional): Mask for target sequence (batch_size, tgt_len).
            memory_mask (Tensor, optional): Mask for memory sequence (batch_size, src_len).

        Returns:
            Tensor: The decoded output (batch_size, tgt_len, vocab_size).
        """
        # Embed target tokens and add positional encoding
        tgt_embed = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]

        # Apply transformer decoder layers
        for layer in self.decoder_layers:
            tgt_embed = layer(tgt_embed, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

        # Project the decoder output to vocab size
        output = self.output_projection(tgt_embed)
        return output


@registry.register_model("blip2_transformer")
class Blip2OPT(Blip2Base):
    """
    BLIP2 OPT model.
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_opt", "caption_coco_opt2.7b")
    """
    # 预训练模型导入配置
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_opt2.7b": "configs/models/blip2/blip2_pretrain_opt2.7b.yaml",
        "pretrain_opt6.7b": "configs/models/blip2/blip2_pretrain_opt6.7b.yaml",
        "caption_coco_opt2.7b": "configs/models/blip2/blip2_caption_opt2.7b.yaml",
        "caption_coco_opt6.7b": "configs/models/blip2/blip2_caption_opt6.7b.yaml",
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
            opt_model="facebook/opt-2.7b",
            prompt="",
            max_txt_len=32,
            apply_lemmatizer=False,
    ):
        """
        Initialize the model with transformer decoder instead of OPT.
        """
        # transformer版本
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.37"), "BLIP-2 OPT requires transformers>=4.27"

        self.tokenizer = self.init_tokenizer()

        # 初始化视觉编码器
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        # 冻结视觉编码器
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        # 初始化 Qformer
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )

        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        # 替换 OPT 模型为 Transformer Decoder
        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
        self.voca_size = len(self.opt_tokenizer)
        self.transformer_decoder = TransformerDecoder(
            vocab_size=self.voca_size,
            hidden_size=768,
            num_layers=6,
            num_heads=8,
            ff_hid_dim=2048
        )

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

        # 定义线性层进行词汇生成
        self.classifier = nn.Linear(768, self.voca_size)

        # 导入情感词列表（如有）
        with open('./emo_cap_dataloader/all_emo_words.txt', 'r') as emo_file:
            self.lines = emo_file.readlines()

        self.emo_words_string = ' '.join([line.strip() for line in self.lines])
        self.flag = False

        # 直接生成情感输出
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

        # 图像的 attention mask
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        # 情感嵌入
        emo_tokens = self.tokenizer([self.emo_words_string], return_tensors="pt").to(image_embeds.device)
        emo_tokens.input_ids = emo_tokens.input_ids[:, 1:]

        # 使用 Transformer 解码器生成情感嵌入
        emo_embeds = self.transformer_decoder.embed_tokens(emo_tokens.input_ids)
        emo_embeds = emo_embeds.repeat(image_embeds.shape[0], 1, 1)
        emo_embeds = emo_embeds.type(image_embeds.dtype)
        emo_embeds = self.emo_clss_proj(emo_embeds)

        emo_atts = torch.ones(emo_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)

        # Qformer 与情感嵌入融合
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        emo_class_fea = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=emo_embeds,
            encoder_attention_mask=emo_atts,
            return_dict=True,
        )

        # 图像特征和情感特征融合
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        # 获取情感文本输入
        all_em_words = []
        for t in samples["text_input"]:
            unique_words = set()
            for word in t.split():
                word = word.strip('.,').lower()
                if word in self.emo_words_string:
                    unique_words.add(word)
            all_em_words.append(' '.join([line.strip() for line in list(unique_words)]) + "\n")

        em_cls_tokens = self.tokenizer(
            all_em_words,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            add_special_tokens=False,
            max_length=10,
        ).to(image_embeds.device)

        # 情感分类损失
        em_mean_feats = emo_class_fea.last_hidden_state.mean(1)
        em_mean_feats = self.classifier(em_mean_feats)
        em_logits = torch.log_softmax(em_mean_feats, dim=1)
        cls_emo = em_logits.view(image_embeds.shape[0], -1, 1).repeat(1, 1, 10)
        cls_loss = F.nll_loss(cls_emo.permute(2, 0, 1).reshape(-1, self.voca_size),
                              em_cls_tokens.input_ids.transpose(0, 1).contiguous().view(-1),
                              ignore_index=self.tokenizer.pad_token_id)

        # Sub-captions
        global_k = 5
        sub_cap_feas = self.sub_proj(sub_cap_feas)
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

        # 融合图像、情感、全局特征
        query_output.last_hidden_state = torch.cat(
            (global_feats, query_output.last_hidden_state, emo_class_fea.last_hidden_state), dim=1)

        inputs_opt = self.sub_opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)

        # 获取文本输入并准备输入嵌入
        text = [t + "\n" for t in samples["text_input"]]
        opt_tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        targets = opt_tokens.input_ids.masked_fill(
            opt_tokens.input_ids == self.tokenizer.pad_token_id, -100
        )
        if self.prompt:
            targets[:, :self.prompt_length] = -100  # do not apply loss to the prompt

        # 创建一个空的target，跟opt出的target拼接，使得输入和目标对齐
        empty_targets = (
            torch.ones(atts_opt.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        # 使用 transformer 解码器嵌入
        inputs_embeds = self.transformer_decoder.embed_tokens(opt_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)

        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

        # 使用 Transformer 解码器进行文本生成
        with self.maybe_autocast():
            outputs = self.transformerdecoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        loss = outputs.loss
        return {"loss": loss + cls_loss, "emo_loss": cls_loss}

    @torch.no_grad()
    def generate(self,
                 samples,
                 use_nucleus_sampling=False,
                 num_beams=5,
                 max_length=30,
                 min_length=1,
                 top_p=0.9,
                 repetition_penalty=1.0,
                 length_penalty=1.0,
                 num_captions=1,
                 temperature=1):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty.
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
            image_embeds = self.visual_encoder(image)

        # 图像嵌入
        image_embeds = image_embeds.view(B, -1, image_embeds.shape[-1])
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        # 文本输入编码（prompt）
        opt_tokens = self.tokenizer(prompt, return_tensors="pt", padding="longest", truncation=True,
                                    max_length=self.max_txt_len).to(image.device)
        input_ids = opt_tokens.input_ids
        attention_mask = opt_tokens.attention_mask

        # 使用 Transformer 解码器
        tgt_mask = self.generate_square_subsequent_mask(input_ids.size(1)).to(image.device)

        # 使用自定义的 Transformer 解码器
        decoder_output = self.transformer_decoder(input_ids, image_embeds, tgt_mask=tgt_mask)

        # logits (batch_size, tgt_len, vocab_size)
        logits = decoder_output  # Decoder的输出就是logits
        next_token_logits = logits[:, -1, :]  # Get logits for the last token position

        # 根据 `temperature`, `top_p`, `num_beams` 等进行采样
        if use_nucleus_sampling:
            next_tokens = self.nucleus_sampling(next_token_logits, top_p=top_p, temperature=temperature)
        else:
            next_tokens = torch.argmax(next_token_logits, dim=-1)

        return self.tokenizer.batch_decode(next_tokens, skip_special_tokens=True)

    def generate_square_subsequent_mask(self, size):
        """Generate a mask for the target sequence to prevent attending to future tokens."""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask

    def nucleus_sampling(self, logits, top_p=0.9, temperature=1.0):
        """Nucleus sampling for text generation."""
        logits = logits / temperature
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens that are not in the top-p nucleus
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_logits[sorted_indices_to_remove] = -float("Inf")

        # Sample from the filtered logits
        probabilities = F.softmax(sorted_logits, dim=-1)
        next_token = torch.multinomial(probabilities, 1)
        next_token = sorted_indices.gather(-1, next_token)

        return next_token.squeeze(-1)

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
            # 获取视觉特征
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            # 扩展查询token
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            # 获取用于文本的输入特征
            inputs_opt = self.sub_opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)

            if isinstance(samples["text_input"], str):
                samples["text_input"] = [samples["text_input"]]
            if prompt:
                text_input = [prompt.format(question) for question in samples["text_input"]]
            else:
                text_input = samples["text_input"]

            # 加载并准备文本tokenizer
            self.opt_tokenizer.padding_side = "left"
            opt_tokens = self.opt_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)

            # 拼接opt的特征和文本tokenizer的输出
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

            # 使用transformer解码器部分
            inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)

            # 使用generate方法进行推理
            outputs = self.opt_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                eos_token_id=self.eos_token_id,
                length_penalty=length_penalty,
            )

            # 解码生成的tokens
            output_text = self.opt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_text = [text.strip() for text in output_text]

        # 如果需要词形还原处理
        if self._apply_lemmatizer or ("apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]):
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
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        opt_model = cfg.get("opt_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            opt_model=opt_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
        )
        model.load_checkpoint_from_config(cfg)

        return model