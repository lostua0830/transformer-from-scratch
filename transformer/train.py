from tqdm import tqdm
from transformer.generate import generate
import transformer.data.shakespeare as shake
import torch
import os
def train_one_epoch(train_loader,model,optimizer,criterion,device,global_step,sample_cfg=None,epoch_index=None,epochs=None):
    model.train()
    running_loss = 0.0
    total_tokens = 0
    desc = f"Epoch {epoch_index+1}/{epochs} | Train" if epoch_index is not None else "Train"
    pbar = tqdm(train_loader, desc=desc, leave=False,dynamic_ncols=False, ncols=100)
    for batch in pbar:
        optimizer.zero_grad(set_to_none=True)
        src_ids = batch["src_ids"]
        tgt_ids = batch["tgt_ids"]
        src_ids,tgt_ids = src_ids.to(device),tgt_ids.to(device)
        tgt_in_ids = tgt_ids[:,:-1]
        logits = model(src_ids,tgt_in_ids)
        
        labels = tgt_ids[:,1:]
        B,T,V = logits.shape
        loss= criterion(logits.reshape(B*T,V),labels.reshape(B*T))
        loss.backward()
        optimizer.step()

        num_tokens = (labels != criterion.ignore_index).sum().item()
        running_loss += loss.item() * num_tokens
        total_tokens += num_tokens

        pbar.set_postfix_str(f"loss={loss.item():7.4f} step={global_step:7d}")
        global_step += 1
        if sample_cfg is not None:
            gen_steps = sample_cfg.get("gen_steps", 500)


            if global_step % gen_steps == 0:
                src0 = src_ids[:1]
                sample_and_gen(model,src_ids,tgt_ids,prefix_ids=src0,max_length_tokens=256,sample_cfg=sample_cfg)
                model.train()
            


    train_loss = running_loss / max(total_tokens, 1)
    return train_loss,global_step

@torch.no_grad()
def eval_one_epoch(val_loader,model,criterion,device,epoch_index=None,epochs=None):
    model.eval()
    running_loss =0.0
    total_tokens = 0
    desc = f"Epoch {epoch_index+1}/{epochs} | Eval" if epoch_index is not None else "Eval"
    pbar = tqdm(val_loader, desc=desc, leave=False,dynamic_ncols=False, ncols=100)
    for batch in pbar:
        src_ids = batch["src_ids"]
        tgt_ids = batch["tgt_ids"]
        src_ids,tgt_ids = src_ids.to(device),tgt_ids.to(device)
        tgt_in_ids = tgt_ids[:,:-1]
        logits = model(src_ids,tgt_in_ids)
        labels = tgt_ids[:,1:]
        B,T,V = logits.shape
        loss= criterion(logits.reshape(B*T,V),labels.reshape(B*T))

        num_tokens = (labels != criterion.ignore_index).sum().item()
        running_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        pbar.set_postfix_str(f"loss={loss.item():7.4f} ")
    val_loss = running_loss/max(total_tokens,1)
    return val_loss

def fit(
    epochs,
    train_loader,
    val_loader,
    model,
    optimizer,
    criterion,
    device,
    global_step=0,
    start_epoch=0,
    total_epochs=None,
    config=None,
    sample_cfg=None,
    root=None
):
    total_loss = 0
    best_val = float("inf")
    if total_epochs is None:
        total_epochs = start_epoch + epochs
    for epoch_idx in range(start_epoch, start_epoch + epochs):
        train_loss,global_step = train_one_epoch(
            train_loader,model,optimizer,criterion,device,global_step,
            sample_cfg=sample_cfg,epoch_index=epoch_idx,epochs=total_epochs
        )
        val_loss = eval_one_epoch(
            val_loader,model,criterion,device,epoch_index=epoch_idx,epochs=total_epochs
        )
        total_loss +=train_loss
        print(f"epoch={epoch_idx+1} train={train_loss:.4f} val={val_loss:.4f}")
        save_checkpoint(root/"checkpoints/latest.pt",model,optimizer,epoch_idx,global_step,config)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(root/"checkpoints/best.pt",model,optimizer,epoch_idx,global_step,config)

            
    return total_loss / max(epochs, 1), global_step

def save_checkpoint(path, model, optimizer, epoch, global_step, config):
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath,exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
    }, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    return ckpt



@torch.no_grad()
def sample_and_gen(model,src_ids,tgt_ids,
                   prefix_ids=None,max_length_tokens=256,sample_cfg=None):
    src0 = src_ids[:1]
    tgt0 = tgt_ids[:1]
    bos_id = sample_cfg["bos_id"]
    eos_id = sample_cfg["eos_id"]
    pad_id = sample_cfg["pad_id"]
    temperature = sample_cfg["temperature"]
    decode_mode = sample_cfg["decode_mode"]
    topp = sample_cfg["topp"]
    topk = sample_cfg["topk"]
    itos = sample_cfg["itos"]
    gen = generate( model,src0,bos_id,eos_id,prefix_ids=prefix_ids,max_length_tokens=max_length_tokens,temperature=temperature,decode_mode= decode_mode,topp=topp,topk = topk)
    src_clean = strip_special(src0[0].tolist(), pad_id, bos_id, eos_id)
    gt_clean = strip_special(tgt0[0].tolist(), pad_id, bos_id, eos_id)
    gen_clean = strip_special(gen[0].tolist(), pad_id, bos_id, eos_id)
    prefix_n = len(src_clean)
    gt_completion = gt_clean[prefix_n:]
    gen_completion = gen_clean[prefix_n:]

    print("\n================= PREFIX -> COMPLETION =================")
    print("PREFIX:")
    print(shake.decode(src_clean, itos))

    print("\nGT completion:")
    print(shake.decode(gt_completion, itos)[:500])

    print("\nGEN completion:")
    print(shake.decode(gen_completion, itos)[:500])
    print("========================================================\n")
    
    
def strip_special(ids, pad_id, bos_id, eos_id):
    ids = [x for x in ids if x != pad_id]
    if len(ids) > 0 and ids[0] == bos_id:
        ids = ids[1:]
    if eos_id in ids:
        ids = ids[: ids.index(eos_id)]
    return ids


