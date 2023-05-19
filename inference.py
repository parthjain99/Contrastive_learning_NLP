import json
from multiprocessing import Pool
import os
import warnings
import torch
from tqdm import tqdm
from transformers import T5Tokenizer

warnings.filterwarnings("ignore")

COMMONGEN = "generate a sentence with: "
XSUM = "summarize: "

args = {}
args["output_dir_path"] = "/common/home/av730/533/NLP_Project/testresults"
args["model_path"] = "/common/home/av730/533/NLP_Project/checkpoints/common_gen/mrm8488/t5-base-finetuned-common_gen/2023-05-01-17_21_18_979143/epoch-0_step-300.pt"
args["dataset"] = "common_gen"
args["dataset_path"] = "/common/home/av730/533/CoNT/jsonl_files/common_gen/test.jsonl"
args["gpus"] = "0,1,2,3"
args["model_name"] = "t5-base"
args["batch_size"] = 16
args["max_src_len"] = 64
args["PTM"] = "t5"
args["beam_size"] = 4
args["early_stop"] = False
args["no_repeat_ngram"] = 4
args["alpha"] = 0.2
args["min_length"] = 5
args["max_length"] = 70
args["diversity_pen"] = 0.0
args["length_pen"] = 2.0

gpus = args["gpus"].split(",")
test_file = args["dataset_path"]

tokenizer = T5Tokenizer.from_pretrained("t5-small")

write_dir = args["output_dir_path"]
os.makedirs(write_dir, exist_ok=True)


def generate_batch(sources, targets, article_ids, model, device, sys_file, ref_file):
    dct = tokenizer.batch_encode_plus(sources, max_length=args["max_src_len"], return_tensors="pt", truncation=True,
                                      padding=True)
    text_id = dct["input_ids"]
    batch_size, seq_len = text_id.size(0), text_id.size(1)
    for p in model.parameters():
        p.requires_grad = False
    cand_id = model.infer_beam_search(
        input_ids=text_id.to(device),
        attention_mask=dct["attention_mask"].to(device),
        args=args
    )
    sys_outs = []
    ref_outs = []
    for i in range(batch_size):
        dec = tokenizer.decode(cand_id[i], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        sys_outs.append(dec)
        ref_outs.append(targets[i])

    with open(sys_file, "a") as f:
        for i in range(len(article_ids)):
            inst = {"article_id": article_ids[i], "sys_out": sys_outs[i]}
            print(json.dumps(inst), file=f)
    with open(ref_file, "a") as f:
        for i in range(len(article_ids)):
            inst = {"article_id": article_ids[i], "ref_out": ref_outs[i]}
            print(json.dumps(inst), file=f)


def load_jsonl(path):
    inst_list = []
    begin_idx = 0
    with open(path) as f:
        for line in f:
            inst = json.loads(line)
            inst["article_id"] = str(begin_idx)
            begin_idx += 1
            inst_list.append(inst)
    return inst_list


def split_dataset_into(num_parts=8):
    insts = load_jsonl(test_file)
    insts_every_ds = len(insts) // num_parts
    new_insts = []
    for i in range(num_parts + 1):
        new_insts.append(insts[i * insts_every_ds:(i + 1) * insts_every_ds])
    last_inst = new_insts.pop()
    new_insts[-1].extend(last_inst)
    assert len(new_insts) == num_parts
    return new_insts


def gen_sys_out(inp_args):
    insts_index, ckpt_path, output_dir = inp_args
    insts = insts_split[insts_index]
    insts_every_ds = args["batch_size"]
    num_batches = len(insts) // insts_every_ds + 1
    new_insts = []
    for i in range(num_batches):
        insts_batch = insts[i * insts_every_ds:(i + 1) * insts_every_ds]
        new_insts.append(insts_batch)
    device = f'cuda:{gpus[insts_index]}'
    model = torch.load(ckpt_path, map_location=torch.device('cpu'))
    sys_file = os.path.join(output_dir, f".{args['dataset']}.sys")
    ref_file = os.path.join(output_dir, f".{args['dataset']}.ref")
    model.eval()
    model.to(device)
    
    if args["dataset"] == "common_gen":
        prompt = COMMONGEN
    else:
        prompt = XSUM
    with tqdm(total=len(new_insts)) as pbar:
        for insts_batch in new_insts:
            
            sources = [prompt+ inst_batch["source"] for inst_batch in insts_batch]
            if len(sources) == 0:
                break
            targets = [inst_batch["target"] for inst_batch in insts_batch]
            article_ids = [inst_batch["article_id"] for inst_batch in insts_batch]
            generate_batch(sources, targets, article_ids, model, device, sys_file, ref_file)
            pbar.update(1)

insts_split = split_dataset_into(len(gpus))
with Pool(len(gpus)) as p:
    p.map(gen_sys_out, zip(range(len(gpus)), [args["model_path"]] * len(gpus), [write_dir] * len(gpus)))