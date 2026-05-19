"""
Directly measure CortexGPT loss on real Pile data at seq_len=2048 (training conditions).
This rules out whether the high training loss (17.6) is data-specific or model-specific.
"""
import sys, math, torch, torch.nn.functional as F
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from model import build_cortex_gpt
from data import build_dataloader, load_tokenizer
from transformers import AutoModelForCausalLM

DEV   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = "EleutherAI/pythia-160m"
DTYPE = torch.bfloat16
N_BATCHES = 5   # average over 5 batches of 4 seqs x 2048 tokens

print(f"Device: {DEV}")

tok = load_tokenizer(MODEL)
loader = build_dataloader(tok, seq_len=2048, batch_size=4, num_workers=0, seed=42)
data_iter = iter(loader)

def collect_batches(n):
    batches = []
    for _ in range(n):
        b = next(data_iter)
        batches.append({
            "input_ids": b["input_ids"].to(DEV),
            "labels":    b["labels"].to(DEV),
        })
    return batches

print(f"Loading {N_BATCHES} Pile batches ...")
batches = collect_batches(N_BATCHES)
print(f"Loaded. Batch shape: {batches[0]['input_ids'].shape}")

def avg_loss(loss_fn):
    losses = [loss_fn(b) for b in batches]
    return sum(losses) / len(losses)

# ── 1. Pythia-160m ─────────────────────────────────────────────────────────────
print("\n[1] Pythia-160m on real Pile 2048-tok ...")
pythia = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE).to(DEV).eval()
def py_loss(b):
    with torch.no_grad():
        logits = pythia(b["input_ids"]).logits.float()
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           b["labels"].reshape(-1)).item()
loss_py = avg_loss(py_loss)
print(f"  Pythia loss = {loss_py:.4f}")
del pythia; torch.cuda.empty_cache()

# ── 2. CortexGPT T=1, prelude_norm=True ───────────────────────────────────────
print("\n[2] CortexGPT T=1, pn=True ...")
m, _ = build_cortex_gpt(MODEL, memory_slots=0, from_scratch=False,
                         scalable_init=False, h0_init="z0", torch_dtype=DTYPE)
m = m.to(DEV).eval()
def cx_loss_t1(b):
    with torch.no_grad():
        return m(b["input_ids"], labels=b["labels"],
                 num_steps=torch.tensor([0, 1]))["loss"].item()
loss_cx_t1 = avg_loss(cx_loss_t1)
print(f"  CortexGPT T=1 pn=True  loss = {loss_cx_t1:.4f}  (+{loss_cx_t1-loss_py:.4f} vs Pythia)")

# ── 3. CortexGPT T=1, prelude_norm=False ──────────────────────────────────────
m.ln_prelude = None
def cx_loss_t1_nopn(b):
    with torch.no_grad():
        return m(b["input_ids"], labels=b["labels"],
                 num_steps=torch.tensor([0, 1]))["loss"].item()
loss_cx_nopn = avg_loss(cx_loss_t1_nopn)
print(f"  CortexGPT T=1 pn=False loss = {loss_cx_nopn:.4f}  (+{loss_cx_nopn-loss_py:.4f} vs Pythia)")

# ── 4. CortexGPT T=8, prelude_norm=False ──────────────────────────────────────
def cx_loss_t8_nopn(b):
    with torch.no_grad():
        return m(b["input_ids"], labels=b["labels"],
                 num_steps=torch.tensor([0, 8]))["loss"].item()
loss_cx_t8_nopn = avg_loss(cx_loss_t8_nopn)
print(f"  CortexGPT T=8 pn=False loss = {loss_cx_t8_nopn:.4f}  (+{loss_cx_t8_nopn-loss_py:.4f} vs Pythia)")
del m; torch.cuda.empty_cache()

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"  Random baseline       : {math.log(50304):.4f}")
print(f"  Pythia (oracle)       : {loss_py:.4f}")
print(f"  CortexGPT T=1 pn=True : {loss_cx_t1:.4f}")
print(f"  CortexGPT T=1 pn=False: {loss_cx_nopn:.4f}")
print(f"  CortexGPT T=8 pn=False: {loss_cx_t8_nopn:.4f}")
print(f"  Training (step 10)    : ~17.6  <- still unexplained")
