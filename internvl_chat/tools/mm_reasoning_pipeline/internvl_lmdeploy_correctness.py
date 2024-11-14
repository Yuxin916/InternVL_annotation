import io
import os
import time
import json
import socket
import argparse
import datetime
import subprocess

import torch
# import deepspeed

from PIL import Image
from collections import defaultdict
from lmdeploy import pipeline, TurbomindEngineConfig, VisionConfig
from lmdeploy.vl.constants import IMAGE_TOKEN

try:
    from petrel_client.client import Client
    client = Client()
except:
    import warnings
    warnings.warn(
        'Fail to import petrel_client! '
        'You can ignore this warning if you do not need to load image from ceph.'
    )


IMG_PLACEHOLDER = '<image>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'


INSTRUCTION_EN = (
    "Your task is to answer the question below. "
    "Give step by step reasoning before you answer, and when you're ready to answer, "
    "please use the format \"Final answer: ..\""
    "\n\n"
    "Question:"
    "\n\n"
    "{question}"
)

INSTRUCTION_ZH = (
    "你的任务是回答以下问题。在回答之前，请逐步推理说明您的思路。当你准备好给出答案时，请使用以下格式：\"答案: ...\""
    "\n\n"
    "问题:"
    "\n\n"
    "{question}"
)

VALID_INSTRUCTIONS = [
    'Answer the question using a single word or phrase.',
    "Answer with the option's letter from the given choices directly.",
]
VALID_INSTRUCTIONS = set(VALID_INSTRUCTIONS)


def init_distributed_mode():
    rank = int(os.environ['SLURM_PROCID'])
    local_rank = rank % torch.cuda.device_count()

    world_size = int(os.environ["SLURM_NTASKS"])
    local_size = int(os.environ["SLURM_NTASKS_PER_NODE"])

    if "MASTER_PORT" not in os.environ:
        port = 22222
        # for i in range(22222, 65535):
        #     cmd = f'netstat -aon|grep {i}'
        #     with os.popen(cmd, 'r') as file:
        #         if '' == file.read():
        #             port = i
        #             break

        print(f'MASTER_PORT = {port}')
        os.environ["MASTER_PORT"] = str(port)

        time.sleep(3)

    node_list = os.environ["SLURM_NODELIST"]
    addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = addr

    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['LOCAL_WORLD_SIZE'] = str(local_size)
    os.environ['WORLD_SIZE'] = str(world_size)

    torch.cuda.set_device(local_rank)


def localtime():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


def collate_fn(batches):
    items = []
    inputs = []
    for batch in batches:
        items.append(batch['item'])
        inputs.append((batch['question'], batch['image']))

    return inputs, items


class VQADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        sample_max_num=None,
    ):
        with open(data) as file:
            self.data = file.readlines()

        if sample_max_num is not None and len(self.data) > sample_max_num:
            print(f'Truncate data lines. {len(self.data)} => {sample_max_num}')
            step = len(self.data) // sample_max_num
            self.data = self.data[::step]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = json.loads(self.data[idx])
        question = item['question']
        images = item['image']
        if not isinstance(images, (list, tuple)):
            images = [images]

        images_new = []
        for image in images:
            if 's3://' in image:
                image = io.BytesIO(client.get(image))
            image = Image.open(image).convert('RGB')
            images_new.append(image)
        images = images_new

        for instruction in VALID_INSTRUCTIONS:
            if question.endswith(instruction):
                question = question[:-len(instruction)].strip()

        return {
            'question': question.replace(IMG_PLACEHOLDER, IMAGE_TOKEN),
            'image': images,
            'item': item.copy(),
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def evaluate_chat_model():
    dataset = VQADataset(
        data=args.prompt_path,
        sample_max_num=args.sample_max_num,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        sampler=InferenceSampler(len(dataset)),
    )

    generation_config = dict(
        request_output_len=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    item2num = defaultdict(int)
    results_file = os.path.basename(args.prompt_path)
    results_file = os.path.join(args.out_dir, results_file)
    if os.path.exists(results_file):
        with open(results_file) as file:
            lines = file.readlines()
        for line in lines:
            item = json.loads(line)
            if isinstance(item['image'], list):
                item2num[(tuple(item['image']), item['question'])] += 1
            else:
                item2num[(item['image'], item['question'])] += 1

    if args.ref_dir is not None:
        ref_items = set()
        ref_file = os.path.join(args.ref_dir, os.path.basename(results_file))

        with open(ref_file) as file:
            lines = file.readlines()
        for line in lines:
            item = json.loads(line)
            ref_items.add((item['image'], item['question']))

        print(f'{len(ref_items)=}')

    print(
        f'[Rank {torch.distributed.get_rank()}] '
        f'Begin to answer {len(dataloader)} batches '
        f'(about {len(dataloader) * args.batch_size} samples), '
        f'{args.prompt_path=}, '
        f'{len(item2num)=}'
    )

    log_freq = max(len(dataloader) // 100, 1)
    print_freq = max(len(dataloader) // 100, 1)
    outputs = []
    for idx, (inputs, items) in enumerate(dataloader):
        assert len(inputs) == len(items)
        assert len(inputs) == 1

        if args.ref_dir is not None and (items[0]['image'], items[0]['question']) not in ref_items:
            continue

        if isinstance(items[0]['image'], list):
            cnt = args.num_return_sequences - item2num[(tuple(items[0]['image']), items[0]['question'])]
        else:
            cnt = args.num_return_sequences - item2num[(items[0]['image'], items[0]['question'])]

        if cnt <= 0:
            continue

        response_list = []
        for _ in range(0, cnt, args.batch_size):
            curr_response_list = pipe([inputs[0]] * args.batch_size, **generation_config)
            response_list.extend([response.text for response in curr_response_list])
        assert len(response_list) == cnt

        response_key = 'response'
        query_list = [inputs[0][0]] * cnt
        for item_idx, item in enumerate(items):
            n = cnt
            for r in response_list[item_idx * n: item_idx * n + n]:
                item = item.copy()
                item[response_key] = r
                outputs.append(item)

        if idx % log_freq == 0:
            print(
                f'[{localtime()}] '
                f'[Rank {torch.distributed.get_rank()}] '
                f'[Progress {idx}/{len(dataloader)}] '
            )

        if idx % print_freq == 0 and torch.distributed.get_rank() == 0:
            print(
                f'[Prompt]\n{query_list[-1]}\n'
                f'[Image]\n{outputs[-1]["image"]}\n'
                f'[Input]\n{outputs[-1]["question"]}\n'
                f'[Output]\n{outputs[-1][response_key]}\n'
                f'[Answer]\n{outputs[-1]["answer"]}\n'
                f'[End]\n'
            )

        if idx % print_freq == 0 and torch.distributed.get_rank() == 0 and cnt > 2:
            print(
                f'[Prompt]\n{query_list[-2]}\n'
                f'[Image]\n{outputs[-2]["image"]}\n'
                f'[Input]\n{outputs[-2]["question"]}\n'
                f'[Output]\n{outputs[-2][response_key]}\n'
                f'[Answer]\n{outputs[-2]["answer"]}\n'
                f'[End]\n'
            )

        if torch.distributed.get_rank() == 0:
            print(f'[Item {idx} {localtime()}] Finish to log')

    print(f'[{localtime()}] [Rank {torch.distributed.get_rank()}] Finish')

    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_outputs, outputs)

    merged_outputs = sum(merged_outputs, start=[])

    if torch.distributed.get_rank() == 0:
        with open(results_file, 'a') as file:
            for output in merged_outputs:
                file.write(json.dumps(output) + '\n')

        print(f'[{localtime()}] Results ({len(merged_outputs)=}) saved to {results_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--prompt-path', type=str, default='')
    parser.add_argument('--out-dir', type=str, default='sampled_outputs')
    parser.add_argument('--ref-dir', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--min-new-tokens', type=int, default=1)
    parser.add_argument('--num-return-sequences', type=int, default=8)
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--sample-max-num', type=int, default=None)
    parser.add_argument('--prompt-version', type=str, default='en', choices=['en', 'zh'])
    args = parser.parse_args()

    global INSTRUCTION
    if args.prompt_version == 'zh':
        INSTRUCTION = INSTRUCTION_ZH
    elif args.prompt_version == 'en':
        INSTRUCTION = INSTRUCTION_EN
    else:
        assert False, f'Unsupported prompt version {args.prompt_version}'

    assert args.num_return_sequences % args.batch_size == 0
    assert args.temperature > 0

    init_distributed_mode()

    model_name = '_'.join(args.checkpoint.split('/')[-2:])
    args.out_dir = os.path.join(args.out_dir, model_name)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    if int(os.getenv('RANK', '0')) % args.tp != 0:
        print(f"[SLURM_PROCID {int(os.environ['SLURM_PROCID'])}] Exit early")
        exit(0)

    if args.tp > 1:
        os.environ['RANK'] = str(int(os.environ['RANK']) // args.tp)
        os.environ['LOCAL_RANK'] = str(int(os.environ['LOCAL_RANK']) // args.tp)
        os.environ['WORLD_SIZE'] = str(int(os.environ['WORLD_SIZE']) // args.tp)
        # different rank should use different gpu, otherwise the all gather operation will be blocked
        torch.cuda.set_device(int(os.environ['RANK']) % torch.cuda.device_count())

    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
        timeout=datetime.timedelta(days=10),
    )
    torch.distributed.barrier()
    print(f'world_size={torch.distributed.get_world_size()}, ip={socket.gethostbyname(socket.gethostname())}')

    vision_config = VisionConfig(max_batch_size=25)
    pipe = pipeline(
        args.checkpoint,
        vision_config=vision_config,
        backend_config=TurbomindEngineConfig(session_len=8192, tp=args.tp)
    )
    pipe.vl_encoder.model.config.max_dynamic_patch = args.max_num
    pipe.vl_encoder.model.config.dynamic_image_size = args.dynamic

    # lmdeploy will update the current_device
    torch.cuda.set_device(int(os.environ['RANK']) % torch.cuda.device_count())

    print(
        f'Begin to sample data from model {args.checkpoint}, '
        f'dynamic: {pipe.vl_encoder.model.config.dynamic_image_size}, '
        f'max_num: {pipe.vl_encoder.model.config.max_dynamic_patch}, '
    )
    evaluate_chat_model()