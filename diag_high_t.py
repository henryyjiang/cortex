"""
Measure how CortexGPT loss scales with T on real Pile data.
Hypothesis: without curriculum (default --curriculum_steps 0), the model
immediately samples T~8-16 from LN-Poisson(8), and pretrained weights at
high T give loss >> random, explaining the observed 17.6 at step 10.
"""
import sys, math, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from model import build_cortex_gpt
from data import build_dataloader, load_tokenizer

DEV   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = "EleutherAI/pythia-160m"
DTYPE = torch.bfloat16
N_BATCHES = 3

tok = load_tokenizer(MODEL)
loader = build_dataloader(tok, seq_len=2048, batch_size=4, num_workers=0, seed=42)
batches = [next(iter(loader)) for _ in range(N_BATCHES)]
batches = [{"input_ids": b["input_ids"].to(DEV), "labels": b["labels"].to(DEV)} for b in batches]
print(f"Loaded {N_BATCHES} Pile batches of shape {batches[0]['input_ids'].shape}")

m, _ = build_cortex_gpt(MODEL, memory_slots=0, from_scratch=False,
                         scalable_init=False, h0_init="z0", torch_dtype=DTYPE)
m = m.to(DEV).eval()

print("\nT  | pn=True  | pn=False")
print("-" * 30)
for T in [1, 2, 4, 8, 12, 16, 24, 32]:
    losses_pn, losses_nopn = [], []
    for b in batches:
        with torch.no_grad():
            loss_pn = m(b["input_ids"], labels=b["labels"],
                        num_steps=torch.tensor([0, T]))["loss"].item()
        losses_pn.append(loss_pn)
    avg_pn = sum(losses_pn) / len(losses_pn)

    m.ln_prelude = None
    for b in batches:
        with torch.no_grad():
            loss_nopn = m(b["input_ids"], labels=b["labels"],
                          num_steps=torch.tensor([0, T]))["loss"].item()
        losses_nopn.append(loss_nopn)
    avg_nopn = sum(losses_nopn) / len(losses_nopn)

    # Restore prelude norm for next iteration
    import torch.nn as nn
    m.ln_prelude = nn.LayerNorm(m.config.hidden_size).to(DEV).to(DTYPE)
    # Reset to identity (default init)
    m.ln_prelude.weight.data.fill_(1.0)
    m.ln_prelude.bias.data.fill_(0.0)

    print(f"{T:2d} | {avg_pn:.4f}   | {avg_nopn:.4f}")

print(f"\nRandom baseline: {math.log(50304):.4f}")
print("Training step 10 (no curriculum): ~17.6")
