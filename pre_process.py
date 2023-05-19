import json
import glob
import os
from transformers import AutoTokenizer

COMMONGEN_PROMPT = "generate a sentence with: "
XSUM_PROMPT = "summarize: "

def process_data(model: str, dataset_path: str, output_dir_path: str, dataset_name: str, tgt_len: int, src_len: int):
    if model not in ["t5-small", "t5-base"] and not "common_gen" in model and not "xsum" in model:
        print("ERROR: model not supported. Only t5-small or t5-base allowed for model")
        return False
    
    tokenizer = AutoTokenizer.from_pretrained(model)
    if dataset_name == "common_gen":
        prompt = COMMONGEN_PROMPT
    else:
        prompt = XSUM_PROMPT
    
    files = glob.glob(os.path.join(dataset_path, "*.jsonl"))

    for file in files:
        output_file_path = os.path.join(output_dir_path, os.path.basename(file))
        print(output_file_path)
        input_file = open(file, 'r')
        output_file = open(output_file_path, 'w')

        for line in input_file:
            line_json = json.loads(line)
            line_json["src_id"] = tokenizer.encode(prompt + line_json["source"])
            line_json["tgt_id"] = tokenizer.encode(line_json["target"])

            line_json["target_inp"] = line_json["tgt_id"] if len(line_json["tgt_id"]) <= tgt_len else line_json["tgt_id"][:tgt_len-1]
            
            line_json["src_inp"] =  line_json["src_id"] if len(line_json["src_id"]) <= src_len else line_json["src_id"][:src_len-1]+[tokenizer.eos_token_id]
            
            line_json["target_outp"] = line_json["target_inp"]
            line_json["target_inp"] = [tokenizer.pad_token_id] + line_json["target_inp"][:-1]
            line_json["seq_len"] = len(line_json["target_outp"])

            output_file.write(json.dumps(line_json, ensure_ascii=False))
            output_file.write("\n")

model_name="mrm8488/t5-base-finetuned-common_gen"
dataset_jsonl_path = "/common/home/av730/533/CoNT/jsonl_files/common_gen"
output_dir_path = "/common/home/av730/533/NLP_Project/tokenized"
dataset_name = "common_gen"
source_len = 64
target_len = 64

process_data(model_name, dataset_jsonl_path, output_dir_path, dataset_name, source_len, target_len)
