#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成仅含英文/数字/空格的 JSONL 数据集（GSM8K格式，answer 为空字符串）

核心逻辑（支持变长且控制命中率）：
1) 先生成 num_prefixes 条“公共序列”（长度统一为 max_common_len）
2) 每条样本先确定真实长度 real_len（定长/高斯/区间）
3) 计算 common_len = round(real_len * prefix_ratio)
4) 从所属公共序列中截取前 common_len 作为样本前缀
5) 后面补充随机后缀到 real_len

这样在 token 维度下，每条样本命中率约为 common_len / real_len ≈ prefix_ratio（仅有取整误差）。
"""

import argparse
import json
import math
import os
import random
import re
from typing import List, Tuple, Optional

from transformers import AutoTokenizer

ALLOWED_RE = re.compile(r'^[A-Za-z0-9 ]+$')


# ------------------------- 基础工具 -------------------------

def ensure_dir(path: str):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def basename_of(path: str) -> str:
    try:
        name = os.path.basename(os.path.normpath(path))
        return name or "tokenizer"
    except Exception:
        return "tokenizer"


def decode_ids(tokenizer, ids: List[int]) -> str:
    try:
        return tokenizer.decode(ids, clean_up_tokenization_spaces=False)
    except Exception:
        return ""


def encode_ids(tokenizer, text: str) -> List[int]:
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except Exception:
        return []


def is_allowed_text(s: str) -> bool:
    if not s:
        return False
    if any(c in s for c in ("\n", "\r", "\t")):
        return False
    return ALLOWED_RE.fullmatch(s) is not None


def filter_allowed(s: str) -> str:
    s2 = "".join(ch for ch in s if (ch.isalnum() or ch == " "))
    s2 = s2.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return s2


def build_safe_token_pools(tokenizer) -> Tuple[List[int], List[int], int]:
    nospace_ids, space_ids = [], []
    try:
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    except Exception:
        special_ids = set()
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)

    for tid in range(vocab_size):
        if tid in special_ids:
            continue
        piece = decode_ids(tokenizer, [tid])
        if not piece:
            continue
        if not is_allowed_text(piece):
            continue
        if piece.startswith(" "):
            space_ids.append(tid)
        else:
            nospace_ids.append(tid)

    filler_id = None
    if space_ids:
        filler_id = space_ids[0]
    else:
        for cand in [" 0", " 1", " a", " A", " B", " 2"]:
            ids = encode_ids(tokenizer, cand)
            if ids:
                t = ids[-1]
                if t not in special_ids:
                    filler_id = t
                    break
        if filler_id is None and nospace_ids:
            filler_id = nospace_ids[0]

    if not space_ids and filler_id is not None:
        space_ids.append(filler_id)
    if not nospace_ids and filler_id is not None:
        nospace_ids.append(filler_id)
    if filler_id is None:
        filler_id = 0

    return nospace_ids, space_ids, filler_id


def fix_to_target_token_len_by_ids(tokenizer, ids: List[int], target_len: int, add_token_id: int) -> List[int]:
    ids = list(ids)
    for _ in range(4096):
        text = decode_ids(tokenizer, ids)
        if not is_allowed_text(text):
            text = filter_allowed(text)
        cur_ids = encode_ids(tokenizer, text)
        cur_len = len(cur_ids)
        if cur_len == target_len:
            return cur_ids
        if cur_len < target_len:
            ids.append(add_token_id)
        else:
            if ids:
                ids.pop()
            else:
                ids.append(add_token_id)

    if not ids:
        ids = [add_token_id]
    while len(ids) < target_len:
        ids.append(add_token_id)
    ids = ids[:target_len]
    return ids


def parse_prefix_ratio(r: str) -> float:
    r = str(r).strip()
    if r.endswith("%"):
        v = float(r[:-1]) / 100.0
    else:
        v = float(r)
    if not (0.0 <= v <= 1.0):
        raise ValueError("prefix-ratio 必须在 [0,1] 或 [0%,100%]")
    return v


# ------------------------- 长度分布 -------------------------

def sample_target_length(
    rng: random.Random,
    fixed_length: int,
    length_mean: Optional[int] = None,
    length_std: Optional[float] = None,
    length_min: Optional[int] = None,
    length_max: Optional[int] = None,
) -> int:
    fixed_length = max(1, int(fixed_length))
    has_gauss = (length_mean is not None) and (length_std is not None)
    has_range = (length_min is not None) and (length_max is not None)

    lo = 1 if length_min is None else max(1, int(length_min))
    hi = None if length_max is None else max(1, int(length_max))
    if hi is not None and lo > hi:
        lo, hi = hi, lo

    if has_gauss:
        mu = max(1, int(length_mean))
        sigma = max(0.0, float(length_std))
        val = mu if sigma == 0 else int(round(rng.gauss(mu, sigma)))
        if hi is not None:
            val = min(val, hi)
        val = max(lo, val)
        return max(1, val)

    if has_range:
        return rng.randint(lo, hi)

    return fixed_length


def build_length_tag(
    fixed_length: int,
    length_mean: Optional[int],
    length_std: Optional[float],
    length_min: Optional[int],
    length_max: Optional[int],
) -> str:
    if (length_mean is not None) and (length_std is not None):
        tag = f"G{int(length_mean)}_{str(length_std).replace('.', 'd')}"
        if (length_min is not None) and (length_max is not None):
            tag += f"_C{int(length_min)}_{int(length_max)}"
        return tag
    if (length_min is not None) and (length_max is not None):
        return f"U{int(length_min)}_{int(length_max)}"
    return f"L{int(fixed_length)}"


# ------------------------- 主流程 -------------------------

def create_multi_prefix_dataset(
    data_num,
    prefix_num,
    length,
    ratio,
    model_path,
    seeds,
    dataset_path,
    length_mean: Optional[int] = None,
    length_std: Optional[float] = None,
    length_min: Optional[int] = None,
    length_max: Optional[int] = None,
):
    total = max(1, int(data_num))
    num_prefixes = max(1, int(prefix_num))
    default_len = max(1, int(length))
    hit_ratio = parse_prefix_ratio(ratio)
    seed = int(seeds)
    rng = random.Random(seed)

    # tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    nospace_ids, space_ids, filler_id = build_safe_token_pools(tokenizer)
    append_id = space_ids[0] if space_ids else filler_id
    first_id = nospace_ids[0] if nospace_ids else filler_id

    # 预采样所有 real_len（这样能先确定 max_common_len）
    real_lens: List[int] = []
    for _ in range(total):
        rl = sample_target_length(
            rng=rng,
            fixed_length=default_len,
            length_mean=length_mean,
            length_std=length_std,
            length_min=length_min,
            length_max=length_max,
        )
        real_lens.append(max(1, int(rl)))

    common_lens = [max(0, min(rl, int(round(rl * hit_ratio)))) for rl in real_lens]
    max_common_len = max(common_lens) if common_lens else 0

    # 生成公共序列池（统一长度 max_common_len）
    prefix_pool_ids: List[List[int]] = []
    for _ in range(num_prefixes):
        if max_common_len <= 0:
            pids = []
        else:
            pids = [first_id]
            for _j in range(max_common_len - 1):
                tid = space_ids[rng.randrange(len(space_ids))] if space_ids else filler_id
                pids.append(tid)
            pids = fix_to_target_token_len_by_ids(tokenizer, pids, max_common_len, add_token_id=append_id)
        prefix_pool_ids.append(pids)

    # 命名与输出
    tok_name = basename_of(model_path)
    pct_label = f"{int(round(hit_ratio * 100))}p"
    length_tag = build_length_tag(default_len, length_mean, length_std, length_min, length_max)
    base_tag = f"seed{seed}_{length_tag}_{pct_label}_num{total}_{tok_name}"

    ensure_dir(dataset_path)
    prefix_dir = os.path.join(dataset_path, f"{base_tag}_prefix")
    ensure_dir(prefix_dir)

    prefix_jsonl_path = os.path.join(prefix_dir, "prefix.jsonl")
    dataset_jsonl_path = os.path.join(dataset_path, f"{base_tag}.jsonl")

    # 写 prefix.jsonl（保存完整公共序列）
    with open(prefix_jsonl_path, "w", encoding="utf-8") as pf:
        for pids in prefix_pool_ids:
            ptxt = decode_ids(tokenizer, pids) if pids else ""
            if not is_allowed_text(ptxt):
                ptxt = filter_allowed(ptxt)
                pids2 = encode_ids(tokenizer, ptxt)
                pids2 = fix_to_target_token_len_by_ids(tokenizer, pids2, max_common_len, add_token_id=append_id)
                ptxt = decode_ids(tokenizer, pids2)
            pf.write(json.dumps({"question": ptxt, "answer": ""}, ensure_ascii=True))
            pf.write("\n")

    # 生成 dataset.jsonl
    # 分组策略沿用：逐组 ceil 配额
    remaining = total
    groups_left = num_prefixes
    sample_idx = 0

    with open(dataset_jsonl_path, "w", encoding="utf-8") as df:
        for g_idx in range(num_prefixes):
            if remaining <= 0 or sample_idx >= total:
                break

            group_target = int(math.ceil(remaining / groups_left))
            groups_left -= 1

            base_prefix_ids = prefix_pool_ids[g_idx] if g_idx < len(prefix_pool_ids) else []

            for _ in range(group_target):
                if sample_idx >= total:
                    break

                real_len = real_lens[sample_idx]
                common_len = common_lens[sample_idx]

                # 前缀截取
                prefix_part = base_prefix_ids[:common_len] if common_len > 0 else []

                # 补后缀
                cur_ids = list(prefix_part)
                suffix_need = max(0, real_len - len(cur_ids))
                for _k in range(suffix_need):
                    tid = space_ids[rng.randrange(len(space_ids))] if space_ids else filler_id
                    cur_ids.append(tid)

                # 严格长度对齐（只在末尾增删，不破坏前缀语义）
                final_ids = fix_to_target_token_len_by_ids(
                    tokenizer,
                    cur_ids,
                    real_len,
                    add_token_id=append_id
                )

                q = decode_ids(tokenizer, final_ids)
                if not is_allowed_text(q):
                    q = filter_allowed(q)
                    q_ids = encode_ids(tokenizer, q)
                    q_ids = fix_to_target_token_len_by_ids(tokenizer, q_ids, real_len, add_token_id=append_id)
                    q = decode_ids(tokenizer, q_ids)

                df.write(json.dumps({"question": q, "answer": ""}, ensure_ascii=True))
                df.write("\n")

                sample_idx += 1

            remaining -= group_target

    return {
        "prefix_jsonl": prefix_jsonl_path,
        "dataset_jsonl": dataset_jsonl_path,
        "max_common_len": max_common_len,
        "avg_real_len": (sum(real_lens) / len(real_lens)) if real_lens else 0.0,
        "avg_common_len": (sum(common_lens) / len(common_lens)) if common_lens else 0.0,
        "avg_hit_ratio": (sum((c / r) for c, r in zip(common_lens, real_lens)) / len(real_lens)) if real_lens else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description="生成含公共序列前缀的数据集（支持变长且按样本控制命中率）")
    ap.add_argument("--total", type=int, required=True, help="数据集总条数（>=1）")
    ap.add_argument("--num-prefixes", type=int, required=True, help="总公共序列数（>=1）")
    ap.add_argument("--length", type=int, required=True, help="默认长度（定长模式或分布回退）")
    ap.add_argument("--prefix-ratio", type=str, required=True, help='命中率/公共前缀比例，如 "50%%" 或 "0.5"')
    ap.add_argument("--tokenizer-dir", type=str, required=True, help="tokenizer 路径")
    ap.add_argument("--seed", type=int, required=True, help="随机种子")
    ap.add_argument("--dataset-path", type=str, default=".", help="输出目录（默认当前目录）")

    # 变长分布参数（可选）
    ap.add_argument("--length-mean", type=int, default=None, help="高斯分布均值（与 --length-std 配对）")
    ap.add_argument("--length-std", type=float, default=None, help="高斯分布标准差（与 --length-mean 配对）")
    ap.add_argument("--length-min", type=int, default=None, help="最小长度（区间模式或高斯截断）")
    ap.add_argument("--length-max", type=int, default=None, help="最大长度（区间模式或高斯截断）")

    args = ap.parse_args()

    # 参数校验
    if (args.length_mean is None) ^ (args.length_std is None):
        raise ValueError("length-mean 和 length-std 必须同时提供，或同时不提供。")
    if (args.length_min is None) ^ (args.length_max is None):
        raise ValueError("length-min 和 length-max 必须同时提供，或同时不提供。")
    if args.length_mean is not None and args.length_mean < 1:
        raise ValueError("length-mean 必须 >= 1")
    if args.length_std is not None and args.length_std < 0:
        raise ValueError("length-std 必须 >= 0")
    if args.length_min is not None and args.length_min < 1:
        raise ValueError("length-min 必须 >= 1")
    if args.length_max is not None and args.length_max < 1:
        raise ValueError("length-max 必须 >= 1")

    ret = create_multi_prefix_dataset(
        data_num=args.total,
        prefix_num=args.num_prefixes,
        length=args.length,
        ratio=args.prefix_ratio,
        model_path=args.tokenizer_dir,
        seeds=args.seed,
        dataset_path=args.dataset_path,
        length_mean=args.length_mean,
        length_std=args.length_std,
        length_min=args.length_min,
        length_max=args.length_max,
    )

    print("[完成] 数据集已生成：")
    print(f"  - 公共前缀：{ret['prefix_jsonl']}  (行数={args.num_prefixes})")
    print(f"  - 数据集：  {ret['dataset_jsonl']} (行数={args.total})")
    print("[统计]（token 维度）")
    print(f"  - max_common_len={ret['max_common_len']}")
    print(f"  - avg_real_len={ret['avg_real_len']:.4f}")
    print(f"  - avg_common_len={ret['avg_common_len']:.4f}")
    print(f"  - avg_hit_ratio={ret['avg_hit_ratio']:.6f}")


if __name__ == "__main__":
    main()