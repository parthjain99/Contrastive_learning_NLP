import itertools
import os
import torch
from sortedcontainers import SortedList

from fastNLP import Callback, rank_zero_call
from fastNLP import logger
from fastNLP.envs import FASTNLP_LAUNCH_TIME
from fastNLP import Metric


class CoNTMetric(Metric):
    def __init__(self):
        super(CoNTMetric, self).__init__()
        self.torch_ngram = []

    def update(self, score):
        self.torch_ngram.append(score)

    def get_metric(self, reset=True):
        torch_ngrams = self.all_gather_object(self.torch_ngram)
        torch_ngram = list(itertools.chain(*torch_ngrams))
        eval_result = sum(torch_ngram) / len(torch_ngram)
        if reset:
            self.torch_ngram = []
        return {"torch_ngram": eval_result.item()}


class CoNTCallback(Callback):
    def __init__(self, args, metric, topk):
        super(CoNTCallback, self).__init__()
        self.args = args
        self.metric = metric
        self.dev_results = SortedList([])
        self.cl_loss_list = []
        self.nll_loss_list = []
        self.topk = topk
        self.timestamp =os.environ[FASTNLP_LAUNCH_TIME]
        logger.info(f"The checkpoint will be saved in this folder for this time: {self.timestamp}")

    def _save_this_model(self, trainer, model_path):
        try:
            torch.save(trainer.model, model_path)
            if len(self.dev_results) > self.topk:
                del_model = self.dev_results.pop(0)[1]
                os.remove(del_model)
            print(f" ============= save model at {model_path} ============= ")
        except Exception as e:
            print(f"The following exception:{e} happens when save {model_path}.")

    @rank_zero_call
    def on_evaluate_end(self, trainer, results):
        trainer.driver.barrier()

        score = results[self.metric]
        save_dir = os.path.join(self.args['save_path'], self.timestamp)
        name = "epoch-{}_step-{}.pt".format(trainer.cur_epoch_idx,
                                            trainer.global_forward_batches // self.args["accum_steps"])

        model_path = os.path.join(save_dir, name)
        os.makedirs(save_dir, exist_ok=True)
        if len(self.dev_results) < self.topk:
            self.dev_results.add((score, model_path))
            self._save_this_model(trainer, model_path)
        else:
            if score >= self.dev_results[0][0]:
                self.dev_results.add((score, model_path))
                self._save_this_model(trainer, model_path)
                print("save hyperparams in ", os.path.join(save_dir, "hyperparams.txt"))
                with open(os.path.join(save_dir, "hyperparams.txt"), "w") as f:
                    for key, value in self.args.items():
                        print(key, ":", value, file=f)
        trainer.driver.barrier()

    @rank_zero_call
    def on_before_backward(self, trainer, outputs):
        cl_loss = outputs["cl_loss"].detach().cpu().item()
        nll_loss = outputs["loss"].detach().cpu().item() - cl_loss
        self.cl_loss_list.append(cl_loss)
        self.nll_loss_list.append(nll_loss)
        if trainer.global_forward_batches % max(self.args['validate_every'] // 4, 1) == 0:
            logger.info(
                f'Contrastive loss is {sum(self.cl_loss_list) / len(self.cl_loss_list)}, '
                f'nll_loss is {sum(self.nll_loss_list) / len(self.nll_loss_list)}')
            self.cl_loss_list = []
            self.nll_loss_list = []