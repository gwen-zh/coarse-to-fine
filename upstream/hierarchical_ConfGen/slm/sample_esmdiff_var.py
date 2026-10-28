import os
import json
import random
from pathlib import Path
import argparse
from functools import partial
from time import time, strftime
import tempfile

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import rootutils
from esm.sdk.api import ESMProtein, GenerationConfig
from esm.models.esm3 import ESM3, ESMOutput
from esm.utils.decoding import decode_structure
from esm.utils.constants import esm3 as C
from esm.sdk.api import ESMProtein, ESMProteinTensor
from esm.utils.generation import iterative_sampling_raw

from omegaconf import OmegaConf
import hydra

from slm.models.model import MaskedDiffusionLanguageModeling
from slm.geometry_feedback import geometry_feedback
from slm.models.utils import protseq_to_data
from slm.utils.checkpoint_utils import load_state_dict_from_lightning_ckpt
from slm.utils.eval_utils import (
    load_seq_from_pdb,
    merge_pdbfiles,
    timer,
)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Loading pre-trianed esm3 model...")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
_model_esm3 = ESM3.from_pretrained("esm3_sm_open_v1").to(device) 


@torch.no_grad()
def decode(structure_tokens, sequence_tokens, esm3_model, save_to=None):
    assert len(structure_tokens) == len(sequence_tokens), f"{len(structure_tokens)} != {len(sequence_tokens)}"
    sequence_tokens = torch.cat(
        [torch.LongTensor([C.SEQUENCE_BOS_TOKEN]), 
        sequence_tokens.cpu(), 
        torch.LongTensor([C.SEQUENCE_EOS_TOKEN])]
    )
    structure_tokens = torch.cat(
        [torch.LongTensor([C.FINE_STRUCTURE_BOS_TOKEN]), 
        structure_tokens.cpu(), 
        torch.LongTensor([C.FINE_STRUCTURE_EOS_TOKEN])]
    )
    
    prot = ESMProteinTensor(sequence=sequence_tokens, structure=structure_tokens)
    prot = prot.to(device)
    esm3_model = esm3_model.to(device)
    raw_protein = esm3_model.decode(prot)
    if save_to is not None:
        raw_protein.to_pdb(save_to)
    return raw_protein


@timer
@torch.no_grad()
def minibatch_gibbs_by_esm(
    protseq,
    esm3_model,
    output_dir: Path,
    sample_basename: str, 
    num_samples: int = 10,
    num_steps: int = 16,
    temperature: float = 1.4,
    top_p: float = 0.9,
    n_max_residue_square: int = 200*200*105,
    coordinates = None,
    mask_ids=None,
):

    str_time = strftime("%Y%m%d-%H%M%S")
    output_dir = output_dir / f"T{temperature}_step{num_steps}_topp{top_p}_N{num_samples}_{str_time}"
    save_to = output_dir / f"{sample_basename}.pdb"
    print(f"Results will save to {save_to}")
    if save_to.exists():
        print(f"Skip existing {save_to}")
        return None

    if mask_ids is not None:
        print(f"Masking {len(mask_ids)} residues and inpainting...")
        assert coordinates is not None, "Need to provide coordinates for masking"
        protseq = list(protseq)
        for idx in mask_ids:
            assert 0 <= idx < len(sequence), f"Invalid mask index {idx} for sequence of length {len(sequence)}"
            protseq[idx] = '_'
            coordinates[idx] = float('Inf')
        protseq = ''.join(protseq)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    with tempfile.TemporaryDirectory() as tmpdirname:
        out_list = []
        batchs = []
        bsz = []
        target_size = len(protseq) * len(protseq) * num_samples
        n_batch = target_size // n_max_residue_square
        residual_size = target_size % n_max_residue_square
        batch_size = n_max_residue_square // int(len(protseq) * len(protseq))
        for i in range(n_batch):
            bsz.append(batch_size)
        if residual_size > 0:
            bsz.append(num_samples-sum(bsz))
        assert sum(bsz) == num_samples, f"{sum(bsz)} != {num_samples}"
        for bs in bsz:
            prot_list = [ESMProtein(sequence=protseq, coordinates=coordinates) for _ in range(bs)]
            print(f"Generating {len(prot_list)} samples for {protseq}...")
            cfg_list = [
                GenerationConfig(track="structure", num_steps=num_steps, temperature=temperature, top_p=top_p)
                for _ in range(bs)
            ]
            out_list += iterative_sampling_raw(
                esm3_model, proteins=prot_list, configs=cfg_list,
            )   # input/output have the same type
        out_list = [out for out in out_list if isinstance(out, ESMProtein)]
        for i, prot in enumerate(out_list):
            base = sample_basename + '.' + f"{i}.pdb"
            tmp = Path(tmpdirname) / base
            prot.to_pdb(tmp)
            saved.append(tmp)
        merge_pdbfiles(saved, save_to, verbose=False)
    return out_list


@timer
@torch.no_grad()
def ddpm_sample_by_esm(
    sequence,
    pl_model,
    output_dir: Path,
    sample_basename: str,
    esm3_model: ESM3 = _model_esm3,
    num_samples: int = 5,
    num_steps: int = 10,
    eps: float = 1e-5,
    n_max_residue_square: int = 200*200*105,
    coordinates = None,
    mask_ids = None,
    filled_ids = None,
    total_size = None,
    sample_max_t: float = 1.0,
    feedback_rounds: int = 0,
    feedback_steps: int = 5,
    feedback_radius: int = 2,
    feedback_policy: str = "geometry",
    feedback_fixed_budget: bool = False,
    seed: int = 0,
):
    model = pl_model
    if feedback_rounds and "|" in sequence:
        raise ValueError("Geometry feedback currently supports continuous single chains")
    str_time = strftime("%Y%m%d-%H%M%S")
    output_dir = output_dir / f"step{num_steps}_eps{eps}_N{num_samples}_{str_time}"
    if feedback_rounds:
        output_dir = output_dir / f"repair_{feedback_policy}_r{feedback_rounds}_s{feedback_steps}_fixed{int(feedback_fixed_budget)}_seed{seed}"
    save_to = output_dir / f"{sample_basename}.pdb"
    print(f"Results will save to {save_to}")
    if save_to.exists():
        print(f"Skip existing {save_to}")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    with tempfile.TemporaryDirectory() as tmpdirname:
        prot = protseq_to_data(
            sequence,
            esm3_model,
            encode_only=True,
            coordinates=coordinates,
            mask_ids=mask_ids,
            filled_ids=filled_ids,
            total_size=total_size,
        )
        sequence_tokens_singleton = prot["sequence_tokens"]
        esm3_model = esm3_model.to('cpu')
        start_t = time()    
        structure_tokens = []
        repair_trace = []
        batchs = []
        bsz = []
        target_size = sequence_tokens_singleton.size(0) * sequence_tokens_singleton.size(0) * num_samples
        n_batch = target_size // n_max_residue_square
        residual_size = target_size % n_max_residue_square
        batch_size = n_max_residue_square // int(sequence_tokens_singleton.size(0) * sequence_tokens_singleton.size(0))
        for i in range(n_batch):
            batchs.append(sequence_tokens_singleton[None, :].repeat(batch_size, 1))
            bsz.append(batch_size)
        
        if residual_size > 0:
            batchs.append(sequence_tokens_singleton[None, :].repeat(num_samples-sum(bsz), 1))
            bsz.append(num_samples-sum(bsz))
        
        assert sum(bsz) == num_samples, f"{sum(bsz)} != {num_samples}"
        print(f"Total {num_samples} samples will be generated in batchs {bsz}...")
        
        for batch in batchs:
            if mask_ids is not None:
                input_prior = prot['structure_tokens'][None, :].repeat(batch.size(0), 1)
                for idx in mask_ids:
                    input_prior[:, idx] = C.VAR_STRUCTURE_MASK_TOKEN
            elif filled_ids is not None:
                input_prior = prot['structure_tokens'][None, :].repeat(batch.size(0), 1)
                for idx in range(total_size):
                    if idx not in filled_ids:
                        input_prior[:, idx] = C.VAR_STRUCTURE_MASK_TOKEN
            else:
                input_prior = None
            lengths = torch.tensor([batch.shape[1]-2]*batch.size(0),dtype=torch.long,device=device)
            sampled = model.ddpm_sample(
                num_steps=num_steps, 
                sequence_tokens=batch, 
                eps=eps, 
                input_prior=input_prior, 
                sample_max_t=sample_max_t,
                lengths=lengths,
                return_coarse=bool(feedback_rounds),
            )
            if feedback_rounds:
                fine, coarse = sampled

                def decode_ca(tokens, batch_index):
                    protein = decode(tokens, batch[batch_index, 1:-1], esm3_model)
                    return protein.coordinates[:, 1, :]  # atom37 CA

                # Keep later baseline batches aligned with the same initial RNG
                # stream, even though feedback consumes additional randomness.
                with torch.random.fork_rng():
                    fine, coarse, trace = geometry_feedback(
                        model, batch, fine, coarse, lengths, decode_ca,
                        rounds=feedback_rounds, num_steps=feedback_steps,
                        radius=feedback_radius, policy=feedback_policy,
                        fixed_budget=feedback_fixed_budget)
                offset = sum(t.shape[0] for t in structure_tokens)
                for event in trace:
                    event["sample"] += offset
                repair_trace.extend(trace)
                sampled = fine
            structure_tokens.append(sampled)
        structure_tokens = torch.cat(structure_tokens, dim=0)
        
        structure_tokens = structure_tokens[:, 1:-1]
        sequence_tokens_singleton = sequence_tokens_singleton[1:-1]

        print(f"Sampling token time: {time() - start_t:.2f}s")
        out_list = []
        for i in range(len(structure_tokens)):
            base = sample_basename + '.' + f"{i}.pdb"
            tmp = Path(tmpdirname) / base
            st_i = structure_tokens[i]
            decode(structure_tokens=st_i, sequence_tokens=sequence_tokens_singleton, esm3_model=esm3_model, save_to=tmp)
            saved.append(tmp)
        merge_pdbfiles(saved, save_to, verbose=False)
        with (output_dir / "sampling_audit.json").open("w") as handle:
            json.dump(dict(seed=seed, num_steps=num_steps, num_samples=num_samples,
                           feedback_rounds=feedback_rounds, feedback_steps=feedback_steps,
                           feedback_radius=feedback_radius, feedback_policy=feedback_policy,
                           feedback_fixed_budget=feedback_fixed_budget,
                           elapsed_seconds=time() - start_t, trace=repair_trace), handle, indent=2)
        print(f"Total time: {time() - start_t:.2f}s")
    return out_list


def get_argparser():
    parser = argparse.ArgumentParser(description="Evaluate the ensemble of protein structures.")
    parser.add_argument("--input", type=str, default="data/targets/bpti", help="Path to the data directory.")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to the model checkpoint. If None, use the pre-trained ESM3 model.")
    parser.add_argument("--output", type=str, default="output/inference_esmdiff")
    parser.add_argument("--mode", type=str, default="ddpm", choices=["gibbs", "ddpm"])
    parser.add_argument("--num_steps", type=int, default=3, help="Number of denoising steps.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate.")
    parser.add_argument("--mask_ids", type=str, default=None, help="Comma-separated list of masked indices.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--feedback_rounds", type=int, default=0)
    parser.add_argument("--feedback_steps", type=int, default=5)
    parser.add_argument("--feedback_radius", type=int, default=2)
    parser.add_argument("--feedback_policy", choices=["geometry", "random", "global"], default="geometry")
    parser.add_argument("--feedback_fixed_budget", action="store_true", help="Use all repair rounds, including on clean chains, for compute controls.")
    args = parser.parse_args()
    if args.feedback_rounds < 0 or args.feedback_steps < 1 or args.feedback_radius < 0:
        parser.error("Invalid feedback budget")
    if args.feedback_rounds and (args.mode != "ddpm" or args.mask_ids is not None):
        parser.error("Feedback currently supports unconditional DDPM sampling of single chains")
    return args


if __name__ == '__main__':
    args = get_argparser()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.ckpt is None:
        assert args.mode == "gibbs", "Only Gibbs sampling is supported for the pre-trained ESM3 model."
        net = _model_esm3
        model = None
    else:
        model = load_state_dict_from_lightning_ckpt(args.ckpt, device=device)
        net = model.net
    
    assert args.input is not None, "Please provide the path to the data directory."
    data_path = Path(args.input)
    assert data_path.is_dir(), f"Invalid directory {data_path} (Currently we only support pdb files in a folder as input)."

    sample_fn = {
        "gibbs": partial(minibatch_gibbs_by_esm, esm3_model=net),
        "ddpm": partial(ddpm_sample_by_esm, pl_model=model,
                        feedback_rounds=args.feedback_rounds, feedback_steps=args.feedback_steps,
                        feedback_radius=args.feedback_radius, feedback_policy=args.feedback_policy,
                        feedback_fixed_budget=args.feedback_fixed_budget,
                        seed=args.seed),
    }[args.mode]

    print(f">>> Sampling mode = {args.mode} ...")
    ##############################################################################

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_paths = list((data_path).iterdir())
    target_paths = sorted(p for p in target_paths if p.suffix == '.pdb')
    for p in tqdm(target_paths, desc="Loop over target..."):
        prot = ESMProtein.from_pdb(p)
        sequence = prot.sequence  # throw away other entities
        coordinates = None
        if args.mask_ids is not None:
            mask_ids = [int(idx) for idx in args.mask_ids.split(",")]    # 0-based index
            coordinates = prot.coordinates
        else:
            mask_ids = None
        sample_fn(
            sequence, 
            output_dir=output_dir, 
            sample_basename=p.stem, 
            num_samples=args.num_samples, 
            num_steps=args.num_steps,
            coordinates=coordinates,   #None
            mask_ids=mask_ids,         #None
        )
