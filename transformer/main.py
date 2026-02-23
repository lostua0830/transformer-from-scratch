import torch
from transformer.model.transformer_model import Transformer
import transformer.train as T
import transformer.data.shakespeare as S
from transformer.model.build_optimizer import build_optimizer
from transformer.generate import generate
import time
import yaml
from pathlib import Path

@torch.no_grad()
def benchmark_cache_vs_nocache(model, src_ids, bos_id, eos_id, gen_cfg, warmup=2, runs=10):
    device = src_ids.device

    def run_once(use_cache: bool):
        _ = generate(
            model=model,
            src_ids=src_ids,
            bos_id=bos_id,
            eos_id=eos_id,
            use_cache=use_cache,
            **gen_cfg,
        )

    # warmup
    for _ in range(warmup):
        run_once(use_cache=False)
        run_once(use_cache=True)

    def timed(use_cache: bool):
        t0 = time.perf_counter()
        for _ in range(runs):
            run_once(use_cache=use_cache)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        return (t1 - t0) / runs

    t_no = timed(False)
    t_yes = timed(True)

    print(f"[benchmark] no_cache: {t_no*1000:.2f} ms/run")
    print(f"[benchmark] use_cache: {t_yes*1000:.2f} ms/run")
    print(f"[benchmark] speedup: {t_no / max(t_yes, 1e-12):.2f}x")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    block_size = 128
    prefix_len = 32
    batch_size = 64
    data_path = Path(__file__).resolve().parent / "data" / "tinyshakespeare.txt"
    stoi,itos,vocab_size,pad_id,bos_id,eos_id,text_ids = S.prepare_vocab(str(data_path))


    data_cfg = {
    "stoi": stoi,
    "itos": itos,
    "vocab_size": vocab_size,
    "pad_id": pad_id,
    "bos_id": bos_id,
    "eos_id": eos_id,
    "text_ids": text_ids,
    "block_size": block_size,
    "prefix_len": prefix_len,
    "batch_size": batch_size,
    "val_ratio": 0.1,
    "num_workers": 0,
    "pin_memory": device.startswith("cuda"),
    }
    cfg_path = Path(__file__).resolve().parent / "config" / "base.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)


    model_cfg = cfg["model_cfg"]
    optimizer_cfg = cfg["optimizer_cfg"]
    train_cfg = cfg["train_cfg"]
    sample_cfg = cfg["sample_cfg"]
    gen_cfg = cfg["gen_cfg"]
    ben_gen_cfg = cfg["ben_gen_cfg"]
    run_cfg     = cfg["run_cfg"]

    model_cfg["src_vocab"] = vocab_size
    model_cfg["tgt_vocab"] = vocab_size
    model_cfg["pad_id"] = pad_id

    sample_cfg["bos_id"] = bos_id
    sample_cfg["eos_id"] = eos_id
    sample_cfg["pad_id"] = pad_id
    sample_cfg["itos"]   = itos

    mode = run_cfg["mode"]
    resume = run_cfg["resume"]
    root = Path(__file__).resolve().parent
    re_ckpt_path = root/run_cfg["latest_ckpt_path"]
    ckpt_path = root/run_cfg["best_ckpt_path"]

    model = Transformer(**model_cfg).to(device)
    if mode == "train":
        train_loader,val_loader = S.make_shake_dataloader(data_cfg)
        
        optimizer = build_optimizer(model,**optimizer_cfg)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id)
        global_step = train_cfg["global_step"]
        start_epoch = 0

        if resume:
            ckpt = T.load_checkpoint(re_ckpt_path, device=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            global_step = ckpt.get("global_step", 0)
            start_epoch = ckpt.get("epoch", -1) + 1
            print(f"resumed from {ckpt_path}, epoch={start_epoch}, global_step={global_step}")

        avg_train_loss, global_step = T.fit(
        epochs=train_cfg["epochs"],
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        global_step=global_step,
        start_epoch=start_epoch,
        total_epochs=start_epoch + train_cfg["epochs"],
        config={
            "data_cfg": data_cfg,
            "model_cfg": model_cfg,
            "optimizer_cfg": optimizer_cfg,
            "train_cfg": train_cfg,
        },
        sample_cfg=sample_cfg,root=root
    )

        print("done", avg_train_loss, global_step)
    elif mode == "benchmark":
        ckpt = T.load_checkpoint(ckpt_path, device=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        prompt = "To be, or not to be"
        src = S.encode(prompt,stoi)[:prefix_len]
        src_ids = src.unsqueeze(0).to(device)
        ben_gen_cfg["prefix_ids"] = src_ids
        for i in range(11):
            print("beam",i+1)
            ben_gen_cfg["beam_size"]=i+1
            benchmark_cache_vs_nocache(model, src_ids, bos_id, eos_id, ben_gen_cfg, warmup=2, runs=20)
    elif mode == "inference":
        ckpt = T.load_checkpoint(ckpt_path,device=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        prompt = "To be, or not to be"
        src = S.encode(prompt,stoi)[:prefix_len]
        src_ids = src.unsqueeze(0).to(device)
        gen_cfg["prefix_ids"] = src_ids
        gen = generate(model,src_ids,bos_id,eos_id,**gen_cfg)
        src_clean = T.strip_special(src_ids[0].tolist(), pad_id, bos_id, eos_id)
        gen_clean = T.strip_special(gen[0].tolist(), pad_id, bos_id, eos_id)
        prefix_n = len(src_clean)
        gen_completion = gen_clean[prefix_n:]

        print("\n================= INFERENCE =================")
        print("PROMPT:")
        print(S.decode(src_clean, itos))
        print("\nGEN completion:")
        print(S.decode(gen_completion, itos)[:500])
        print("============================================\n")





if __name__ == "__main__":
    main()