from __future__ import annotations

import pathlib
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from backend.torch_wrapper import TorchSNNext


def build_dataset(sp, data_path: pathlib.Path, seq_len: int = 32, max_samples: int = 32) -> List[torch.Tensor]:
  """
  Tokenize lines into fixed-length id tensors (pad with EOS).
  """
  eos_id = sp.eos_id()
  samples = []
  with data_path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      ids = sp.encode_as_ids(line)
      ids = ids[: seq_len + 1]  # leave room for next-token target
      if len(ids) < seq_len + 1:
        ids += [eos_id] * (seq_len + 1 - len(ids))
      samples.append(torch.tensor(ids, dtype=torch.long))
      if len(samples) >= max_samples:
        break
  return samples


class TextSeqDataset(Dataset):
  def __init__(self, sp, data_path: pathlib.Path, seq_len: int = 32, max_samples: int | None = None):
    self.samples = build_dataset(sp, data_path, seq_len=seq_len, max_samples=max_samples or 10_000)

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    return self.samples[idx]


def collate_batch(batch: List[torch.Tensor]) -> torch.Tensor:
  return torch.stack(batch, dim=0)


def train_one_batch(
    sp,
    data_path: pathlib.Path,
    device: torch.device = torch.device("cuda"),
    seq_len: int = 32,
    hidden_size: int = 64,
):
  """
  Minimal one-batch training loop to sanity-check gradients.
  - Embedding -> SNN (no synapses) -> linear head for next-token prediction.
  """
  vocab_size = sp.GetPieceSize()
  samples = build_dataset(sp, data_path, seq_len=seq_len, max_samples=1)
  if not samples:
    print("[train test] dataset empty")
    return
  seq = samples[0].to(device)
  inp, tgt = seq[:-1], seq[1:]

  embed = torch.nn.Embedding(vocab_size, hidden_size, device=device)
  snn = TorchSNNext(num_neurons=hidden_size, dt=1e-4, surrogate_slope=4.0, batch_size=1).to(device)
  head = torch.nn.Linear(hidden_size, vocab_size, device=device)
  opt = torch.optim.Adam(list(embed.parameters()) + list(snn.parameters()) + list(head.parameters()), lr=1e-3)

  snn.train()
  opt.zero_grad()
  loss = 0.0
  for token_id in inp:
    current = embed(token_id.unsqueeze(0)).to(device)  # shape (1, hidden_size)
    spikes, _, _ = snn(current)
    logits = head(spikes.unsqueeze(0))  # shape (1, vocab_size)
    loss = loss + torch.nn.functional.cross_entropy(logits, token_id.unsqueeze(0))
  loss.backward()
  torch.nn.utils.clip_grad_norm_(list(embed.parameters()) + list(snn.parameters()) + list(head.parameters()), 1.0)
  opt.step()
  print(f"[train test] one-batch loss: {loss.item():.4f}")


def build_batch(
    sp,
    data_path: pathlib.Path,
    seq_len: int,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
  samples = build_dataset(sp, data_path, seq_len=seq_len, max_samples=batch_size)
  if len(samples) < batch_size:
    raise RuntimeError("Not enough samples; increase max_samples or data size.")
  batch = torch.stack(samples, dim=0)
  inp = batch[:, :-1]
  tgt = batch[:, 1:]
  return inp, tgt


class SNNStack(torch.nn.Module):
  def __init__(self, vocab_size: int, hidden1: int, hidden2: int, batch_size: int = 1, dt: float = 1e-4):
    super().__init__()
    self.batch_size = batch_size
    self.embed = torch.nn.Embedding(vocab_size, hidden1)
    self.snn1 = TorchSNNext(num_neurons=hidden1, dt=dt, surrogate_slope=4.0, batch_size=batch_size)
    # Inter-layer synaptic weights (learned)
    self.syn_w12 = torch.nn.Parameter(torch.randn(hidden2, hidden1) * 0.1)
    self.snn2 = TorchSNNext(num_neurons=hidden2, dt=dt, surrogate_slope=4.0, batch_size=batch_size)
    self.head = torch.nn.Linear(hidden2, vocab_size)

  def forward(self, token_id: torch.Tensor):
    # token_id shape: (batch,)
    x = self.embed(token_id)  # (batch, hidden1)
    spikes1_flat, _, _ = self.snn1(x)
    spikes1 = spikes1_flat.view(token_id.shape[0], -1)
    current2 = torch.matmul(spikes1, self.syn_w12.t())
    spikes2_flat, _, _ = self.snn2(current2)
    spikes2 = spikes2_flat.view(token_id.shape[0], -1)
    logits = self.head(spikes2)
    return logits


def reset_state_layer(layer: TorchSNNext):
  with torch.no_grad():
    layer.V_soma.copy_(layer.soma_E_L)
    layer.V_ais.copy_(layer.ais_E_L)
    layer.refrac_until.fill_(-1e9)
    layer.I_syn_total.zero_()
    layer.spiked_now.zero_()
    layer.buffer_idx.zero_()
    layer.g_incr_buffer.zero_()
    layer.syn_g.zero_()
    layer.syn_pre_trace.zero_()
    layer.syn_post_trace.zero_()
    layer.syn_r_pre.zero_()
    layer.syn_r_post.zero_()


def train_epoch(
    sp,
    data_path: pathlib.Path,
    device: torch.device = torch.device("cuda"),
    seq_len: int = 32,
    batch_size: int = 1,
    hidden1: int = 64,
    hidden2: int = 64,
    lr: float = 1e-3,
    tbptt_len: int = 16,
    log_every: int = 1,
    epochs: int = 1,
    save_path: pathlib.Path | None = None,
):
  vocab_size = sp.GetPieceSize()
  dataset = TextSeqDataset(sp, data_path, seq_len=seq_len, max_samples=batch_size)
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

  model = SNNStack(vocab_size, hidden1, hidden2, batch_size=batch_size).to(device)
  opt = torch.optim.Adam(model.parameters(), lr=lr)
  model.train()

  global_step = 0
  for epoch in range(epochs):
    for batch in loader:
      batch = batch.to(device)
      inp = batch[:, :-1]
      tgt = batch[:, 1:]

      reset_state_layer(model.snn1)
      reset_state_layer(model.snn2)

      opt.zero_grad()
      loss = 0.0
      # TBPTT: process in chunks of tbptt_len
      for start in range(0, seq_len, tbptt_len):
        end = min(seq_len, start + tbptt_len)
        for t in range(start, end):
          token_t = inp[:, t]
          logits = model(token_t)
          loss = loss + torch.nn.functional.cross_entropy(logits, tgt[:, t])
        loss.backward(retain_graph=(end < seq_len))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        global_step += 1
        if global_step % log_every == 0:
          print(f"[train] epoch {epoch+1} step {global_step} chunk {start}-{end} done")
        loss = 0.0

  if save_path is not None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"[train] saved checkpoint to {save_path}")
