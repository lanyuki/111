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

from transformers import AutoTokenizer, OPTForCausalLM, OPTConfig
# import modelscope
# from modelscope import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import transformers
import torch.nn.functional as F
import math





@registry.register_model("blip2_opt")
class Blip2OPT(Blip2Base):
    """
    BLIP2 OPT model.
    Supported model types:
        - pretrained_opt2.7b: pretrained model with OPT2.7b
        - pretrained_opt6.7b: pretrained model with OPT6.7b
        - caption_coco_opt2.7b: fintuned image captioning model with OPT2.7b
        - caption_coco_opt6.7b: fintuned image captioning model with OPT6.7b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_opt", "caption_coco_opt2.7b")
    """
    # 预训练模型导入
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_opt2.7b": "configs/models/blip2/blip2_pretrain_opt2.7b.yaml",
        "pretrain_opt6.7b": "configs/model  s/blip2/blip2_pretrain_opt6.7b.yaml",
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
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        当 apply_lemmatizer 设置为 True 时，predict_answers() 函数返回的结果会经过词形还原处理，变成词汇的基本形式（例如，"running" 会变成 "run"，"better" 会变成 "good"）
        """
        # transformer版本
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.37"), "BLIP-2 OPT requires transformers>=4.27"

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        # 冻结vit
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        # qformer初始,不要cls语义标记,去掉词和位置embedding
        # 每层的输出不要,然后中间层也不要
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )

                                                            #self.Qformer2, self.query_tokens2 = self.init_Qformer(
                                                            #    num_query_token, self.visual_encoder.num_features
                                                            #)
                                                    #
                                                            #self.Qformer3, self.query_tokens3 = self.init_Qformer(
                                                            #    num_query_token, self.visual_encoder.num_features
                                                            #)

        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        #self.Qformer2.cls = None
        #self.Qformer2.bert.embeddings.word_embeddings = None
        #self.Qformer2.bert.embeddings.position_embeddings = None
        #for layer in self.Qformer2.bert.encoder.layer:
        #    layer.output = None
        #    layer.intermediate = None

        #加载opt模型,并且将换行符换成token_id而非结束
        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
        self.opt_model = OPTForCausalLM.from_pretrained(
            opt_model, torch_dtype=torch.float16
        )
        for name, param in self.opt_model.named_parameters():
            param.requires_grad = False
        self.eos_token_id = self.opt_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]

        self.sub_opt_proj = nn.Linear(
           self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )

        # 设最大文本长度,prompt
        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")  #prompt转tkonen,这还蛮不一样的
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

        self.voca_size = len(self.opt_tokenizer)
        # 一步步投到768
        self.classifier = nn.Linear(768,self.voca_size)

        self.emo_clss_proj = nn.Linear(768,1408)
        self.sub_proj = nn.Linear(1408,768)

        # 导入数据,但是具体文件没有看到
        with open('./emo_cap_dataloader/all_emo_words.txt', 'r') as emo_file:
            self.lines = emo_file.readlines()

        self.emo_words_string = ' '.join([line.strip() for line in self.lines])
        self.flag = False

        # 直接fc成一个值,根据情感和文本
        self.aggfc = nn.Linear(1408*2, 1408)
        self.aggfc2 = nn.Linear(1408, 1,bias=False)
        #nn.init.xavier_normal_(self.aggfc)
        #nn.init.xavier_normal_(self.aggfc2)


    def forward(self, samples):
        image = samples["image"]

        #图像特征提取
        B, C, T, H, W = image.shape
        image = image.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))


        #sub_cap_feas = torch.mean(image_embeds, dim = 1)
        #sub_cap_feas = sub_cap_feas.view(B,T,-1)

        #sub_cap_feas = torch.mean(image_embeds, dim = 1)

        #图像加入时间维度，保留通道维度
        sub_cap_feas = image_embeds.view(B,-1,image_embeds.shape[-1])

        #temp_inputs = torch.cat((image_embeds, image_embeds),2).view(-1, 1408 * 2)
#
        #o = self.aggfc2(F.tanh(self.aggfc(temp_inputs)))
#
        #e = o.view(B*T, -1)
        #alpha = F.softmax(e, dim=1)
        #sub_cap_feas = torch.bmm(alpha.unsqueeze(1), image_embeds)
        #sub_cap_feas = sub_cap_feas.view(B,T,-1)

        image_embeds = image_embeds.view(B,-1,image_embeds.shape[-1])
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        # 情感嵌入
        emo_tokens = self.opt_tokenizer(
            [self.emo_words_string],
            return_tensors="pt",
        ).to(image_embeds.device)

        emo_tokens.input_ids = emo_tokens.input_ids[:,1:]

        emo_embeds = self.opt_model.model.decoder.embed_tokens(emo_tokens.input_ids)

        emo_embeds = emo_embeds.repeat(image_embeds.shape[0],1,1)

        emo_embeds = emo_embeds.type(image_embeds.dtype)

        emo_embeds = self.emo_clss_proj(emo_embeds)
        emo_atts = torch.ones(emo_embeds.size()[:-1], dtype=torch.long).to(
            image_embeds.device
        )
        #q和情感融合
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        #query_tokens2 = self.query_tokens2.expand(image_embeds.shape[0], -1, -1)

        emo_class_fea = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=emo_embeds,
            encoder_attention_mask=emo_atts,
            return_dict=True,
        )
        #图和情感
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


        em_cls_tokens = self.opt_tokenizer(
            all_em_words,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            add_special_tokens=False,
            max_length=10,
        ).to(image_embeds.device)

        #zero_clo = Variable(torch.cuda.FloatTensor(image_embeds.shape[0],1).fill_(0))
        em_mean_feats = emo_class_fea.last_hidden_state.mean(1)
        em_mean_feats = self.classifier(em_mean_feats)
        em_logits = torch.log_softmax(em_mean_feats, dim=1)
        cls_emo = em_logits.view(image_embeds.shape[0],-1,1).repeat(1,1,10)
        cls_loss = F.nll_loss(cls_emo.permute(2,0,1).reshape(-1, self.voca_size),
                                em_cls_tokens.input_ids.transpose(0,1).contiguous().view(-1),
                                ignore_index=self.opt_tokenizer.pad_token_id)


        #--------------------------------------------------------------------------------
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

        #em_mean_feats2 = torch.mean(emo_class_fea.last_hidden_state, dim = 1,keepdim = True)

        d = emo_class_fea.last_hidden_state.shape[-1]
        sorce = torch.bmm(emo_class_fea.last_hidden_state, sub_cap_feas.last_hidden_state.transpose(1, 2)) / math.sqrt(d)
        att_weight = F.softmax(sorce, dim=-1)
        global_feats = torch.bmm(att_weight, sub_cap_feas.last_hidden_state)
        #global_feats = global_feats.repeat(1,T,1)

        #_, topk_indices = torch.topk(sim_tensor, k=global_k)
#
        #sub = []
        #for i in range(B):
        #    temp = []
        #    for j in topk_indices[i]:
        #        temp.append(sub_cap_feas.last_hidden_state[i,j,:])
        #    temp_fea = torch.stack(temp,dim=0)
        #    sub.append(temp_fea)
        #sub_cap_feas = torch.stack(sub,dim=0).view(B,global_k,-1)

        #sub_inputs_opt = self.sub_opt_proj(sub_cap_feas)
        #sub_atts_opt = torch.ones(sub_inputs_opt.size()[:-1], dtype=torch.long).to(image.device)
#
#
        #sub_empty_targets = (
        #    torch.ones(sub_atts_opt.size(), dtype=torch.long).to(image.device).fill_(-100)
        #)
#
        #targets = torch.cat([sub_empty_targets, targets], dim=1)
        #inputs_embeds = torch.cat([sub_inputs_opt, inputs_embeds], dim=1)
        #attention_mask = torch.cat([sub_atts_opt, attention_mask], dim=1)

        #--------------------------------------------------------------------------------

        #query_output.last_hidden_state = torch.cat((query_output.last_hidden_state,emo_class_fea.last_hidden_state),dim=1)
        # 三个特征合起来
        query_output.last_hidden_state = torch.cat((global_feats,query_output.last_hidden_state,emo_class_fea.last_hidden_state),dim=1)


        inputs_opt = self.sub_opt_proj(query_output.last_hidden_state)

        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)

        self.opt_tokenizer.padding_side = "right"

        text = [t + "\n" for t in samples["text_input"]]

        opt_tokens = self.opt_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        targets = opt_tokens.input_ids.masked_fill(
            opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        if self.prompt:
            targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt

        # 创建一个空的target，跟opt出的target拼接，使得输入和目标对齐，
        empty_targets = (
            torch.ones(atts_opt.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
        #TODO:RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 2560 but got size 768 for tensor number 1 in the list.
        #需要将 inputs_embeds 调整为与 inputs_opt 在维度1上的形状一致
        device = self.opt_model.device
        inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids).to(device)
        # linear_layer = nn.Linear(768, 2560).to(device)
        # inputs_embeds = torch.cat([inputs_opt, linear_layer(inputs_embeds),], dim=1)
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds ], dim=1)

        # print(inputs_embeds.shape)
        # print(inputs_embeds.shape)
        # print(inputs_embeds)
        # print(inputs_embeds)
        # 填充16896/2560除不干净，没办法啦
        # padding_size = (2560 - (inputs_embeds.size(1) % 2560)) % 2560
        # inputs_embeds = torch.cat(
        #     [inputs_embeds, torch.zeros(inputs_embeds.size(0), padding_size, inputs_embeds.size(2), device=inputs_embeds.device)], dim=1
        # )
        # inputs_embeds = inputs_embeds.view(inputs_embeds.size(0), 2560)
        # inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)

        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.opt_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss + cls_loss,"emo_loss": cls_loss}

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

            #sub_cap_feas = torch.mean(image_embeds, dim = 1)
            #sub_cap_feas = sub_cap_feas.view(B,T,-1)

            #temp_inputs = torch.cat((image_embeds, image_embeds),2).view(-1, 1408 * 2)
#
            #o = self.aggfc2(F.tanh(self.aggfc(temp_inputs)))
#
            #e = o.view(B*T, -1)
            #alpha = F.softmax(e, dim=1)
            #sub_cap_feas = torch.bmm(alpha.unsqueeze(1), image_embeds)
            #sub_cap_feas = sub_cap_feas.view(B,T,-1)

            # 图像和情感的变换，有时间要细看一下
            sub_cap_feas = image_embeds.view(B,-1,image_embeds.shape[-1])

            image_embeds = image_embeds.view(B,-1,image_embeds.shape[-1])
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            emo_tokens = self.opt_tokenizer(
                [self.emo_words_string],
                return_tensors="pt",
            ).to(image_embeds.device)

            emo_tokens.input_ids = emo_tokens.input_ids[:,1:]

            emo_embeds = self.opt_model.model.decoder.embed_tokens(emo_tokens.input_ids)

            emo_embeds = emo_embeds.repeat(image_embeds.shape[0],1,1)

            emo_embeds = emo_embeds.type(image_embeds.dtype)

            emo_embeds = self.emo_clss_proj(emo_embeds)

            emo_atts = torch.ones(emo_embeds.size()[:-1], dtype=torch.long).to(
            image_embeds.device
            )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

            #query_tokens2 = self.qujainery_tokens2.expand(image_embeds.shape[0], -1, -1)
            # 加情感的embeding融合
            emo_class_fea = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=emo_embeds,
                encoder_attention_mask=emo_atts,
                return_dict=True,
            )
            # 输出情感
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            #--------------------------------------------------------------------------------
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

            #em_mean_feats2 = torch.mean(emo_class_fea.last_hidden_state, dim = 1,keepdim = True)

            d = emo_class_fea.last_hidden_state.shape[-1]
            sorce = torch.bmm(emo_class_fea.last_hidden_state, sub_cap_feas.last_hidden_state.transpose(1, 2)) / math.sqrt(d)
            att_weight = F.softmax(sorce, dim=-1)

            global_feats = torch.bmm(att_weight, sub_cap_feas.last_hidden_state)
            #global_feats = global_feats.repeat(1,T,1)


            #_, topk_indices = torch.topk(sim_tensor, k=global_k)
#
            #sub = []
            #for i in range(B):
            #    temp = []
            #    for j in topk_indices[i]:
            #        temp.append(sub_cap_feas.last_hidden_state[i,j,:])
            #    temp_fea = torch.stack(temp,dim=0)
            #    sub.append(temp_fea)
            #sub_cap_feas = torch.stack(sub,dim=0).view(B,global_k,-1)

            #sub_inputs_opt = self.sub_opt_proj(sub_cap_feas)
            #sub_atts_opt = torch.ones(sub_inputs_opt.size()[:-1], dtype=torch.long).to(image.device)
#
#
            #inputs_embeds = torch.cat([sub_inputs_opt, inputs_embeds], dim=1)
            #attention_mask = torch.cat([sub_atts_opt, attention_mask], dim=1)

            #--------------------------------------------------------------------------------


            #query_output.last_hidden_state = torch.cat((query_output.last_hidden_state,emo_class_fea.last_hidden_state),dim=1)
            query_output.last_hidden_state = torch.cat((global_feats,query_output.last_hidden_state,emo_class_fea.last_hidden_state),dim=1)


            inputs_opt = self.sub_opt_proj(query_output.last_hidden_state)

            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                image.device
            )

            opt_tokens = self.opt_tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

            # new version for transformers>=4.27
            inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_opt,inputs_embeds],dim=1)


            outputs = self.opt_model.generate(
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
            output_text = self.opt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

            # previous version for transformers<4.27
            # if use_nucleus_sampling:
            #     query_embeds = inputs_opt.repeat_interleave(num_captions, dim=0)
            #     num_beams = 1
            # else:
            #     query_embeds = inputs_opt.repeat_interleave(num_beams, dim=0)

            # outputs = self.opt_model.generate(
            #     input_ids=input_ids,
            #     query_embeds=query_embeds,
            #     attention_mask=attention_mask,
            #     do_sample=use_nucleus_sampling,
            #     top_p=top_p,
            #     temperature=temperature,
            #     num_beams=num_beams,
            #     max_new_tokens=max_length,
            #     min_length=min_length,
            #     eos_token_id=self.eos_token_id,
            #     repetition_penalty=repetition_penalty,
            #     length_penalty=length_penalty,
            #     num_return_sequences=num_captions,
            # )

            # prompt_length = opt_tokens.input_ids.shape[1]
            # output_text = self.opt_tokenizer.batch_decode(
            #     outputs[:, prompt_length:], skip_special_tokens=True
            # )

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
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_opt = self.sub_opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                image.device
            )

            if isinstance(samples["text_input"], str):
                samples["text_input"] = [samples["text_input"]]
            if prompt:
                text_input = [prompt.format(question) for question in samples["text_input"]]
            else:
                text_input = samples["text_input"]

            self.opt_tokenizer.padding_side = "left"
            opt_tokens = self.opt_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)

            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

            # require transformers>=4.27
            inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_opt,inputs_embeds],dim=1)
            # cap输出
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
            output_text = self.opt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
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