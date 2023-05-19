from transformers import T5ForConditionalGeneration, AutoTokenizer
from fastNLP.io import JsonLoader
from extras import CoNTMetric, CoNTCallback
from torch_optimizer import Adafactor
from fastNLP import Trainer
from fastNLP import prepare_dataloader
from model import Model

args = {}
args['task'] = "common_gen"
args['batch_size'] = 8
args['max_tgt_len'] = 64
args['max_src_len'] = 64
args['ignore_index'] = -100

## Model Params
args['model_name'] = "mrm8488/t5-base-finetuned-common_gen" # 
args['n_gram'] = 2
args['train_beam_size'] = 12
args['train_diversity_pen'] = 2.0

args['alpha'] = 0.2
args['eval_beam_size'] = 8
args['eval_diversity_pen'] = 0.0

args['max_length'] = 70
args['min_length'] = 5
args['no_repeat_ngram'] = 4
args['length_pen'] = 2.0
args['early_stop'] = True
args['max_sample_num'] = 16

## trainer params
args['validate_every'] = 100
args['accum_steps'] = 1
args['epochs'] = 1
args['learning_rate'] = 2e-5
args['save_path'] = "checkpoints/" + args['task'] + "/" + args["model_name"]
args['train_data'] = '/common/home/av730/533/NLP_Project/tokenized/train.jsonl'
args['val_data'] = '/common/home/av730/533/NLP_Project/tokenized/val.jsonl'

tokenizer = AutoTokenizer.from_pretrained(args["model_name"])
args['pad_id'] = tokenizer.pad_token_id
args['eos_id'] = tokenizer.eos_token_id
args['bos_id'] = tokenizer.bos_token_id
init_bundle = JsonLoader().load({'train': args['train_data'], 'val': args['val_data']})
dls = prepare_dataloader(init_bundle, batch_size=args['batch_size'], num_workers=1)

for dl in dls.values():
    dl.set_pad('src_inp', pad_val=args['pad_id'])
    dl.set_pad('target_inp', pad_val=args['pad_id'])
    dl.set_pad('target_outp', pad_val=args['ignore_index'])

model = Model(args)

callback = [CoNTCallback(args, metric="torch_ngram#ngram-overlap", topk=3)]
metric = CoNTMetric()
optimizer = Adafactor(model.parameters(), lr = args['learning_rate'], relative_step=False,scale_parameter=False,warmup_init=False)

model_trainer = Trainer(model=model, train_dataloader=dls['train'], device=[0,1,2,3], optimizers=optimizer, accumulation_steps=args['accum_steps'],
                       evaluate_dataloaders=dls['val'], metrics={"ngram-overlap": metric}, driver="torch",
                       n_epochs=args['epochs'], callbacks=callback, fp16=False, evaluate_every=args['validate_every'], 
                       torch_kwargs={'ddp_kwargs': {'find_unused_parameters': True}})

model_trainer.run()
