"""
Diagnostic: compare Pythia-160m forward pass vs CortexGPT retrofit at T=1.

Key results from Run 1:
  Pythia baseline: 2.28
  CortexGPT T=1 pn=True: 4.18 (+1.9)  <- prelude_norm is the main issue
  CortexGPT T=1 pn=False: 2.41 (+0.14) <- nearly Pythia-level!

But training shows 17.6 at step 10. This script investigates why.
"""
import sys, math, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from model import build_cortex_gpt, CortexGPT
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = "EleutherAI/pythia-160m"
DTYPE = torch.bfloat16
torch.manual_seed(42)
RANDOM_BASELINE = math.log(50304)

TEXTS = [
    "The transformer architecture has become the dominant approach in natural language processing. "
    "It relies on self-attention mechanisms to capture long-range dependencies. "
    "Recurrent neural networks maintain hidden state across time steps. "
    "Pre-training on large corpora and fine-tuning has proven very effective. "
    "Backpropagation computes gradients by applying the chain rule through layers. "
    "Attention is all you need according to the seminal 2017 paper. "
    "The softmax function converts logits into a probability distribution. "
    "Weight initialisation affects the convergence speed of training significantly. "
    "Batch normalisation and layer normalisation stabilise training by normalising. "
    "The BERT model introduced bidirectional pretraining for language understanding. " * 8,

    "In 1969, Neil Armstrong became the first human to walk on the Moon. "
    "The Second World War ended in 1945 with the surrender of Germany and Japan. "
    "The French Revolution began in 1789 with the storming of the Bastille. "
    "Isaac Newton formulated the laws of motion and universal gravitation. "
    "Albert Einstein published the theory of special relativity in 1905. "
    "Charles Darwin proposed the theory of evolution by natural selection. "
    "Ludwig van Beethoven composed nine symphonies despite going deaf. "
    "William Shakespeare wrote Hamlet Macbeth Othello and King Lear. "
    "The Great Wall of China stretches thousands of miles across northern China. "
    "The Roman Empire fell in 476 AD when Odoacer deposed the last emperor. " * 8,

    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n"
    "class Node:\n    def __init__(self, val):\n        self.val = val\n        self.next = None\n"
    "import numpy as np\nimport pandas as pd\nfrom sklearn.model_selection import train_test_split\n"
    "SELECT * FROM users WHERE age > 18 ORDER BY created_at DESC LIMIT 100;\n"
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. The five boxing wizards jump quickly. "
    "Sphinx of black quartz judge my vow. Jackdaws love my big sphinx of quartz. " * 8,

    "The human genome contains approximately 3 billion base pairs of DNA. "
    "Proteins are synthesised from amino acids according to instructions in mRNA. "
    "The mitochondria is known as the powerhouse of the cell. "
    "Photosynthesis converts light energy into chemical energy in plants. "
    "Quantum mechanics describes the behaviour of particles at subatomic scales. "
    "The speed of light in a vacuum is approximately 299792458 metres per second. "
    "General relativity describes gravity as a curvature of spacetime. "
    "The periodic table organises chemical elements by atomic number. "
    "DNA replication occurs during the S phase of the cell cycle. "
    "Neurons communicate via electrochemical signals across synapses. " * 8,
]

def make_batch(tok, seq_len=512):
    all_ids = []
    for text in TEXTS:
        ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        while len(ids) < seq_len + 1:
            ids = torch.cat([ids, ids])
        all_ids.append(ids[:seq_len + 1])
    batch = torch.stack(all_ids).to(DEV)
    return batch[:, :seq_len].contiguous(), batch[:, 1:].contiguous()

def pythia_loss(model, input_ids, labels):
    """Correct: extract logits, compute CE manually (avoid HF's internal shift)."""
    with torch.no_grad():
        logits = model(input_ids).logits.float()
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)).item()

def cortex_loss(model, input_ids, labels, **kw):
    with torch.no_grad():
        return model(input_ids, labels=labels, **kw)["loss"].item()

def cortex_loss_train(model, input_ids, labels, **kw):
    """Loss with model in TRAIN mode (activates dropout if any)."""
    model.train()
    with torch.no_grad():
        out = model(input_ids, labels=labels, **kw)
    model.eval()
    return out["loss"].item()

print("Loading tokenizer ...")
tok = AutoTokenizer.from_pretrained(MODEL)
input_ids, labels = make_batch(tok, seq_len=512)
print(f"Batch: {input_ids.shape}")

# ── 1. Pythia baseline ─────────────────────────────────────────────────────────
print("\n[1] Pythia-160m baseline (eval)")
pythia = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE).to(DEV).eval()
loss_py = pythia_loss(pythia, input_ids, labels)
print(f"  loss = {loss_py:.4f}")
del pythia; torch.cuda.empty_cache()

# ── 2. CortexGPT current training setup — eval, batched path ─────────────────
print("\n[2] CortexGPT retrofit T=1, prelude_norm=True, eval, BATCHED (our diag shows 4.18)")
m, _ = build_cortex_gpt(MODEL, memory_slots=0, from_scratch=False,
                         scalable_init=False, h0_init="z0", torch_dtype=DTYPE)
m = m.to(DEV).eval()
loss_t1_batch = cortex_loss(m, input_ids, labels, num_steps=torch.tensor([0, 1]))
print(f"  loss = {loss_t1_batch:.4f}   delta = {loss_t1_batch - loss_py:+.4f}")

# Same model but per-sequence path (list of same tuples -- forces _loop_per_sequence)
# Force use_per_seq by passing mixed T values
per_seq_same = [(0, 1)] * 4
loss_t1_perseq_same = cortex_loss(m, input_ids, labels,
                                   num_steps=[(0,1),(0,1),(0,2),(0,1)])
print(f"  loss (per-seq, mixed T1/T2) = {loss_t1_perseq_same:.4f}   delta = {loss_t1_perseq_same - loss_py:+.4f}")

# ── 3. CortexGPT T=1, train mode ──────────────────────────────────────────────
print("\n[3] CortexGPT T=1, pn=True, TRAIN MODE (as in actual training)")
loss_t1_train = cortex_loss_train(m, input_ids, labels, num_steps=torch.tensor([0, 1]))
print(f"  loss = {loss_t1_train:.4f}   delta = {loss_t1_train - loss_py:+.4f}")
del m; torch.cuda.empty_cache()

# ── 4. CortexGPT T=1, prelude_norm=False, eval ───────────────────────────────
print("\n[4] CortexGPT T=1, prelude_norm=False, eval")
m, _ = build_cortex_gpt(MODEL, memory_slots=0, from_scratch=False,
                         scalable_init=False, h0_init="z0", torch_dtype=DTYPE)
m.ln_prelude = None
m = m.to(DEV).eval()
loss_t1_nopn = cortex_loss(m, input_ids, labels, num_steps=torch.tensor([0, 1]))
print(f"  loss = {loss_t1_nopn:.4f}   delta = {loss_t1_nopn - loss_py:+.4f}")
del m; torch.cuda.empty_cache()

# ── 5. CortexGPT seq_len=2048 (match training) ───────────────────────────────
print("\n[5] CortexGPT T=1, pn=True, seq_len=2048 (match training)")
input_ids_2k, labels_2k = make_batch(tok, seq_len=2048)
print(f"  Batch: {input_ids_2k.shape}")
m, _ = build_cortex_gpt(MODEL, memory_slots=0, from_scratch=False,
                         scalable_init=False, h0_init="z0", torch_dtype=DTYPE)
m = m.to(DEV).eval()
loss_t1_2k = cortex_loss(m, input_ids_2k, labels_2k, num_steps=torch.tensor([0, 1]))
print(f"  loss = {loss_t1_2k:.4f}   delta vs Pythia on 512-tok = {loss_t1_2k - loss_py:+.4f}")

# Pythia on 2048 tokens too
pythia = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE).to(DEV).eval()
loss_py_2k = pythia_loss(pythia, input_ids_2k, labels_2k)
print(f"  Pythia 2048-tok loss = {loss_py_2k:.4f}")
del pythia, m; torch.cuda.empty_cache()

# ── 6. CortexGPT T=8, prelude_norm=False (what well-trained should look like) ─
print("\n[6] CortexGPT T=8, prelude_norm=False (target at end of curriculum)")
m, _ = build_cortex_gpt(MODEL, memory_slots=0, from_scratch=False,
                         scalable_init=False, h0_init="z0", torch_dtype=DTYPE)
m.ln_prelude = None
m = m.to(DEV).eval()
loss_t8_nopn = cortex_loss(m, input_ids, labels, num_steps=torch.tensor([0, 8]))
print(f"  loss = {loss_t8_nopn:.4f}   delta = {loss_t8_nopn - loss_py:+.4f}")
del m; torch.cuda.empty_cache()

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Summary")
print("="*60)
print(f"  Random baseline             : {RANDOM_BASELINE:.4f}")
print(f"  Pythia-160m (oracle)        : {loss_py:.4f}")
print()
print(f"  CortexGPT T=1 pn=T eval     : {loss_t1_batch:.4f}  (+{loss_t1_batch-loss_py:.2f} vs Pythia)")
print(f"  CortexGPT T=1 pn=T train    : {loss_t1_train:.4f}  (+{loss_t1_train-loss_py:.2f} vs Pythia)")
print(f"  CortexGPT T=1 pn=T mix-T    : {loss_t1_perseq_same:.4f}  (per-seq path)")
print(f"  CortexGPT T=1 pn=F eval     : {loss_t1_nopn:.4f}  (+{loss_t1_nopn-loss_py:.2f} vs Pythia)")
print(f"  CortexGPT T=1 pn=T 2048-tok : {loss_t1_2k:.4f}  (+{loss_t1_2k-loss_py:.2f} vs Pythia)")
print(f"  CortexGPT T=8 pn=F eval     : {loss_t8_nopn:.4f}  (+{loss_t8_nopn-loss_py:.2f} vs Pythia)")
print()
print("  Observed training loss at step 10: ~17.6 (needs explaining)")
print("  Primary issue found: prelude_norm adds ~+2 nats at T=1")
print("  Fix: disable prelude_norm for retrofit/combined modes")
