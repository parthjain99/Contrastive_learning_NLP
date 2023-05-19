import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import T5ForConditionalGeneration
from fastNLP import seq_len_to_mask

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.generator = T5ForConditionalGeneration.from_pretrained(self.args['model_name'])
        
        self.encoder = self.generator.get_encoder()
        self.decoder = self.generator.get_decoder()
        self.nll = CrossEntropyLoss(ignore_index = self.args['ignore_index'])
        self.vocab_size = self.generator.config.vocab_size
        self.hidden_size = self.generator.config.hidden_size
        self.linear_layer = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.xavier_uniform_(self.linear_layer.weight)
    
    def get_ngram(self, input_tensor, n=2):
        batch_size, cand_num, seq_len = input_tensor.size(0), input_tensor.size(1), input_tensor.size(2)
        clipped_seq_len = seq_len - n + 1
        input_tensor_repeated = input_tensor[:, :, None, :].repeat(1, 1, clipped_seq_len, 1)
        temp_matrix_1 = torch.triu(torch.ones(seq_len, seq_len))
        temp_matrix_2 = torch.triu(torch.ones(seq_len, seq_len), diagonal=n)
        temp_matrix_final = (temp_matrix_1 - temp_matrix_2)[:clipped_seq_len].bool()[None, None, :, :]
        returned = torch.masked_select(input_tensor_repeated, temp_matrix_final.to(input_tensor.device))
        return returned.view(batch_size, cand_num, clipped_seq_len, n)
    
    def pad2_len(self, input_tensor, max_len):
        pad_size = max_len - input_tensor.shape[-1]
        pad_tensor = torch.full([input_tensor.shape[0], input_tensor.shape[1], pad_size], self.args['pad_id'],
                                device=input_tensor.device).long()
        return torch.cat([input_tensor, pad_tensor], dim=-1)
        
    def oracle_function(self, references, predicted):
        
        n = min(min(self.args['n_gram'], references.size(-1)), predicted.size(-1))
        sample_num = predicted.size(1)
        
        ## padded references
        references_padding = (~(references == self.args['pad_id'])).float()
        references_mask = torch.arange(0, references_padding.size(1), device=references_padding.device) * torch.ones_like(references_padding)
        references_mask = torch.where(
            references_mask < (torch.sum(references_padding, dim=-1, keepdims=True) - n + 1),
            references_padding, torch.zeros_like(references_padding)
        )[:, :references_mask.size(-1) - n + 1]
        
        ## padded predicted
        predicted_padding = (~(predicted == self.args['pad_id'])).float()
        predicted_mask = torch.arange(0, predicted_padding.size(-1), device=predicted_padding.device) * torch.ones_like(predicted_padding)
        predicted_mask = torch.where(
            predicted_mask < (torch.sum(predicted_padding, dim=-1, keepdims=True) - n + 1),
            predicted_padding, torch.zeros_like(predicted_padding)
        )[:, :, :predicted_mask.size(-1) - n + 1]
        
        
        references = references * references_padding
        predicted = predicted * predicted_padding
        references = references[:, None, :].repeat(1, sample_num, 1)
        
        ## get n-gram tensors
        references_ngram = self.get_ngram(references, n).float()
        predicted_ngram = self.get_ngram(predicted, n).float()
        
        ## get similarity between references and predicted n-gram
        similarities_matrix = (torch.norm(predicted_ngram.unsqueeze(3) - references_ngram.unsqueeze(2),
                                 p=2, dim=-1) == 0.0).to(torch.float)
        
        similarities_matrix *= predicted_mask.unsqueeze(3) * references_mask.unsqueeze(1).unsqueeze(2)
        similarities_matrix = torch.sum(torch.max(similarities_matrix, dim=-1).values, dim=-1)
        
        
        ## Calculate brevity penalty
        ref_len = torch.sum(references_padding, dim=-1, keepdims=True)
        pre_len = torch.sum(predicted_padding, dim=-1)
        brevity = torch.exp(1 - (ref_len / pre_len))
        brevity = torch.where(ref_len >= pre_len, brevity, torch.ones_like(brevity))

        return similarities_matrix / torch.sum(predicted_mask, dim=-1) * brevity
    
    def ranking_loss(self, cos_distance, oracle_distance):
        margin = 0.01
        ones = torch.ones(cos_distance.size(), device=cos_distance.device)
        loss_func = torch.nn.MarginRankingLoss(0.0)
        total_loss = loss_func(cos_distance, cos_distance, ones)

        # candidate loss
        n = cos_distance.size(1)
        for i in range(1, n):
            pos_score = cos_distance[:, :-i]
            neg_score = cos_distance[:, i:]
            same_mask = (torch.abs(oracle_distance[:, :-i] - oracle_distance[:, i:]) > margin).float()
            ones = torch.ones(pos_score.size(), device=cos_distance.device)
            loss_func = torch.nn.MarginRankingLoss(margin * i, reduction='none')
            marginal_loss = loss_func(pos_score, neg_score, ones)
            if same_mask.sum() > 0:
                total_loss += (marginal_loss * same_mask).sum() / same_mask.sum()

        return total_loss
    
    def affine_transformation(self, input_features, padding_mask, axis=1):
        length = torch.sum(padding_mask, axis=1) - 1
        padding_mask = seq_len_to_mask(length, max_len=padding_mask.shape[-1])
        trans_tmp = F.relu(self.linear_layer(input_features))  # batch
        trans_tmp = trans_tmp * padding_mask.unsqueeze(-1).float()
        trans_emb = torch.sum(trans_tmp, axis=axis)
        return trans_emb * (1 / length.unsqueeze(-1))
        
    @torch.no_grad()
    def beam_search(self, src_inp, src_mask):
        candidates = self.generator.generate(input_ids = src_inp,
                                attention_mask = src_mask,
                                num_return_sequences=self.args['train_beam_size'],
                                num_beam_groups=self.args['train_beam_size'],
                                diversity_penalty=self.args['train_diversity_pen'],
                                num_beams=self.args['train_beam_size'],
                                max_length=self.args['max_length'] + 2,
                                min_length=self.args['min_length'] + 1,
                                no_repeat_ngram_size=self.args['no_repeat_ngram'],
                                length_penalty=self.args['length_pen'],
                                early_stopping=self.args['early_stop'])
        return candidates.view(self.args['batch_size'], self.args['train_beam_size'], -1)
    
    @torch.no_grad()
    def eval_beam_search(self, input_ids, attention_mask):
        self.generator.eval()
        ret_dict = self.generator.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_return_sequences=self.args['eval_beam_size'],
            num_beams=self.args['eval_beam_size'],
            max_length=self.args['max_length'] + 2,
            min_length=self.args['min_length'] + 1,
            no_repeat_ngram_size=self.args['no_repeat_ngram'],
            length_penalty=self.args['length_pen'],
            early_stopping=self.args['early_stop'],
            output_scores=True,
            return_dict_in_generate=True,
            output_hidden_states=True
        )
        candidates = ret_dict["sequences"]
        candidates_mask = (candidates != self.args['pad_id']).long()
        
        candidates_len = torch.sum(candidates_mask, dim=-1)
        max_len = torch.max(candidates_len).item()
        candidates = candidates[:, :max_len]
        beam_indices = ret_dict['beam_indices']
        beam_indices = torch.where(beam_indices > 0, beam_indices, 0)
        decoder_hidden_states = ret_dict["decoder_hidden_states"]
        hidden_states_from_output = torch.cat(
            [decoder_hidden_states[i][-1] for i in range(len(decoder_hidden_states))],dim=1)
        
        h = hidden_states_from_output.shape[-1]
        decoder_hidden_states = torch.gather(hidden_states_from_output, 0,
                                             beam_indices[:, :-1].unsqueeze(-1).repeat(1, 1, h))

        encoder_hidden_states = ret_dict["encoder_hidden_states"][-1]  
        encoder_feature = self.affine_transformation(encoder_hidden_states, attention_mask)  
        decoder_feature = self.affine_transformation(decoder_hidden_states, candidates_mask[:, :-1])
        decoder_feature = decoder_feature.view(input_ids.size(0), self.args['eval_beam_size'], -1) 
        cos_distance = torch.cosine_similarity(encoder_feature.unsqueeze(1), decoder_feature,dim=-1)
        scores = ret_dict["sequences_scores"].view(input_ids.size(0), -1)
        normalize = torch.sum(0 - scores, keepdim=True, dim=-1)
        score = (1 - self.args['alpha']) * (scores / normalize) + self.args['alpha'] * cos_distance
        candidates = candidates.view(input_ids.size(0), self.args['eval_beam_size'], -1)
        max_indices = torch.argmax(score, dim=-1)[:, None, None]
        dummy = max_indices.repeat(1, 1, candidates.size(2))
        return torch.gather(candidates, 1, dummy).squeeze(1)
    
    @torch.no_grad()
    def infer_beam_search(self, input_ids, attention_mask, args):
        self.generator.eval()
        ret_dict = self.generator.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_return_sequences=args['beam_size'],
            num_beams=args['beam_size'],
            max_length=args['max_length'] + 2,
            min_length=args['min_length'] + 1,
            no_repeat_ngram_size=args['no_repeat_ngram'],
            length_penalty=args['length_pen'],
            early_stopping=args['early_stop'],
            output_scores=True,
            return_dict_in_generate=True,
            output_hidden_states=True
        )
        candidates = ret_dict["sequences"]
        candidates_mask = (candidates != self.args['pad_id']).long()
        
        candidates_len = torch.sum(candidates_mask, dim=-1)
        max_len = torch.max(candidates_len).item()
        candidates = candidates[:, :max_len]
        beam_indices = ret_dict['beam_indices']
        beam_indices = torch.where(beam_indices > 0, beam_indices, 0)
        decoder_hidden_states = ret_dict["decoder_hidden_states"]
        hidden_states_from_output = torch.cat(
            [decoder_hidden_states[i][-1] for i in range(len(decoder_hidden_states))],dim=1)
        
        h = hidden_states_from_output.shape[-1]
        decoder_hidden_states = torch.gather(hidden_states_from_output, 0,
                                             beam_indices[:, :-1].unsqueeze(-1).repeat(1, 1, h))

        encoder_hidden_states = ret_dict["encoder_hidden_states"][-1]  
        encoder_feature = self.affine_transformation(encoder_hidden_states, attention_mask)  
        decoder_feature = self.affine_transformation(decoder_hidden_states, candidates_mask[:, :-1])
        decoder_feature = decoder_feature.view(input_ids.size(0), args['beam_size'], -1) 
        cos_distance = torch.cosine_similarity(encoder_feature.unsqueeze(1), decoder_feature,dim=-1)
        scores = ret_dict["sequences_scores"].view(input_ids.size(0), -1)
        normalize = torch.sum(0 - scores, keepdim=True, dim=-1)
        score = (1 - args['alpha']) * (scores / normalize) + args['alpha'] * cos_distance
        candidates = candidates.view(input_ids.size(0), args['beam_size'], -1)
        max_indices = torch.argmax(score, dim=-1)[:, None, None]
        dummy = max_indices.repeat(1, 1, candidates.size(2))
        return torch.gather(candidates, 1, dummy).squeeze(1)
    
    
    def evaluate_step(self, src_inp, target_inp, target_outp):
        src_mask = (src_inp != self.args['pad_id'])
        candidate_ids = self.eval_beam_search(src_inp, src_mask)
        return {"score": self.oracle_function(target_inp, candidate_ids.unsqueeze(1)).mean()}
    
    def forward(self, src_inp, target_inp, target_outp):
        
        ## masking
        src_mask = ~(src_inp == self.args['pad_id'])
        
        tgt_mask = ~(target_inp == self.args['pad_id'])
        tgt_mask[:,0] = 1
        
        ## get encoder and decoder states
        encoder_state = self.encoder(src_inp, src_mask)['last_hidden_state']
        decoder_state = self.decoder(input_ids=target_inp, attention_mask=tgt_mask,
                                     encoder_hidden_states=encoder_state,encoder_attention_mask=src_mask)
        if self.args['model_name'] == "t5-small":
            decoder_last_layer = decoder_state[0] * (self.generator.model_dim ** -0.5)
        else:
            decoder_last_layer = decoder_state[0]
            
        lm_logits = self.generator.lm_head(decoder_last_layer)
        nll_loss = self.nll(lm_logits.view(-1, self.vocab_size), target_outp.view(-1))
        
        ## contrastive loss calculation
        
        ## generate beam size candidates from model
        candidates = self.beam_search(src_inp, src_mask)
        cand_len = candidates.size(2)
        
        in_batch_samples = target_inp[None,:,:].repeat(self.args['batch_size'], 1 , 1)
        sample_len = in_batch_samples.size(2)
        
        if sample_len < cand_len:
            in_batch_samples = self.pad2_len(in_batch_samples, cand_len)
        else:
            in_batch_samples = in_batch_samples[:,:,:cand_len]
        
        all_samples = torch.cat([candidates, in_batch_samples], dim = 1)
        oracle_distances = self.oracle_function(target_inp, all_samples)
        
        distance_mask = (oracle_distances < 0.99)
        oracle_distance_masked = oracle_distances * distance_mask.float()
        
        oracle_distances, actual_indices = torch.sort(oracle_distance_masked, dim=-1, descending=True)
        
        if self.args['max_sample_num']:
            sample_num = min(self.args['max_sample_num'] - 1, oracle_distance_masked.size(1) - 1)
        else:
            sample_num = oracle_distance_masked.size(1) - 1
            
        
        sampled_oracle_distance = oracle_distances[:, :sample_num]
        sampled_actual_indices = actual_indices[:, :sample_num]
        
        gold_indices = torch.arange(0, self.args['batch_size']).reshape(self.args['batch_size'], 1).to(
            sampled_oracle_distance.device) + candidates.size(1)  
        sampled_indices = torch.cat([gold_indices, sampled_actual_indices], dim=-1)
        
        gold_distance = torch.full([self.args['batch_size'], 1], 1.0, device=sampled_oracle_distance.device)
        sampled_bleu_distance = torch.cat([gold_distance, sampled_oracle_distance], dim=-1)
        temp_indices = sampled_indices.unsqueeze(-1).repeat(1, 1, all_samples.size(2))
        sampled_input = torch.gather(all_samples, 1, temp_indices)

        decoder_hidden_states = []
        for sample_idx in range(sampled_indices.size(-1)):
            sampled_input_dec = sampled_input[:, sample_idx, :]

            sample_pad_mask = ~(sampled_input_dec == self.args['pad_id'])
            sample_pad_mask[:, 0] = 1

            decoder_out = self.decoder(input_ids=sampled_input_dec, attention_mask=sample_pad_mask,
                                  encoder_hidden_states=encoder_state,
                                  encoder_attention_mask=src_mask)  
            decoder_feature = decoder_out[0]
            decoder_feature = self.affine_transformation(decoder_feature, sample_pad_mask)  
            decoder_hidden_states.append(decoder_feature.unsqueeze(1))

        encoder_feature = self.affine_transformation(encoder_state, src_mask)  
        decoder_feature = torch.cat(decoder_hidden_states, dim=1)  
        cos_distance = torch.cosine_similarity(encoder_feature.unsqueeze(1), decoder_feature,
                                               dim=-1)  
        cl_loss = self.ranking_loss(cos_distance, sampled_bleu_distance)
        
        return {'loss': nll_loss + cl_loss, "cl_loss": cl_loss}
