"""Unified autoregressive sampler with qubit-count validity masks and optional chunked decoding."""

from typing import Optional

import torch
import torch.nn.functional as F

from ..data.utils import Vocab
from ..layers.utils import mask_with_top_k, mask_with_top_p


def token_to_sequence(tokens: torch.Tensor, vocab: Vocab):
    """Convert generated token IDs [B, K, L] into space-separated gate strings."""
    mask = (tokens != vocab.pad_id).long()
    seqlen = mask.sum(-1).cpu().numpy()
    tokens_np = tokens.cpu().numpy()

    bsz, nsamples = tokens_np.shape[:2]
    sequences = []
    for b in range(bsz):
        sample_seqs = []
        for i in range(nsamples):
            slen = int(seqlen[b, i])
            if slen <= 1:
                sample_seqs.append("")
                continue
            last = tokens_np[b, i, slen - 1]
            end = slen - 1 if last == vocab.eos_id else slen
            token_ids = tokens_np[b, i, 1:end]
            sample_seqs.append(" ".join(vocab.detokenize(token_ids)))
        sequences.append(sample_seqs)
    return sequences


def build_qubit_token_mask(vocab: Vocab, max_qubits: int) -> torch.Tensor:
    """Precompute validity mask table V_n of shape [max_qubits, |V|].

    Row (n - 1) has 1 for <eos> and every gate token whose qubit indices are all < n.
    """
    vocab_size = len(vocab)
    token_mask = torch.zeros(max_qubits, vocab_size)
    token_mask[:, vocab.eos_id] = 1.0

    for idx in range(4, vocab_size):
        token = vocab.id2token[idx]
        parts = token.split("_")
        if len(parts) == 2:
            max_q = int(parts[-1])
        elif len(parts) == 3:
            max_q = max(int(parts[1]), int(parts[2]))
        else:
            raise ValueError(f"Unexpected gate token format in vocabulary: {token!r}")

        for q in range(max_qubits):
            if q >= max_q:
                token_mask[q, idx] = 1.0

    return token_mask


class Sampler:
    """Autoregressive decoder with KV caching, qubit-range masking, and optional chunking.

    Sharing the encoder forward pass across sample chunks (via ``max_chunk_size``)
    keeps peak VRAM bounded for both RL rollouts and high-budget (large-K) inference.
    """

    def __init__(self, model, vocab_path: str, max_qubits: int = 20):
        self.max_qubits = max_qubits
        self.vocab = Vocab(vocab_path)
        self.bos_token_id = self.vocab.bos_id
        self.eos_token_id = self.vocab.eos_id
        self.pad_token_id = self.vocab.pad_id

        self.model = model
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        self.token_mask = build_qubit_token_mask(self.vocab, self.max_qubits)

    @torch.inference_mode()
    def evaluate(self, src, src_len, graph, tgt, model=None, reduce: bool = True):
        """Compute logits or mean token log-likelihood for target sequences."""
        assert src.shape[0] == tgt.shape[0]
        assert src.dim() == 2

        if model is None:
            model = self.model
        model.eval()

        if tgt.dim() == 2:
            tgt = tgt[:, None]

        num_samples = tgt.shape[1]
        src, src_mask = model.encoder(src, src_len, graph)
        _, seqlen, d_model = src.shape
        device = src.device

        src_expand = src[:, None, :, :].tile(1, num_samples, 1, 1).reshape(-1, seqlen, d_model)
        src_mask_expand = src_mask[:, None].tile(1, num_samples, 1, 1, 1).reshape(-1, 1, 1, seqlen)
        tgt_expand = tgt.reshape(-1, tgt.shape[-1])

        tgt_sub_mask = model._create_subsequent_mask(tgt_expand.size(1) - 1, device)
        tgt_mask = tgt_sub_mask.unsqueeze(1)

        logits, _ = model.decoder(
            tgt=tgt_expand[:, :-1],
            memory=src_expand,
            tgt_mask=tgt_mask,
            memory_mask=src_mask_expand,
        )
        if not reduce:
            return logits

        log_probs = F.log_softmax(logits, dim=-1)
        log_prob = torch.gather(log_probs, 2, tgt_expand[:, 1:, None]).squeeze(-1)

        mask = (tgt_expand[:, 1:] != self.pad_token_id).float()
        log_likelihood = (log_prob * mask).sum(-1) / mask.sum(-1).clamp(1e-6)
        if tgt.dim() == 3:
            log_likelihood = log_likelihood.reshape(tgt.shape[0], tgt.shape[1])
        return log_likelihood

    def decode_sequence(self, tokens: torch.Tensor):
        return token_to_sequence(tokens, self.vocab)

    def _decode_chunk(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        logit_mask: torch.Tensor,
        num_samples: int,
        temperature: float,
        max_length: int,
        top_k: int,
        top_p: float,
        use_cache: bool,
    ):
        bsz, seqlen, d_model = src.shape
        device = src.device

        src_expand = src[:, None, :, :].expand(-1, num_samples, -1, -1).reshape(-1, seqlen, d_model)
        src_mask_expand = src_mask[:, None].expand(-1, num_samples, -1, -1, -1).reshape(-1, 1, 1, seqlen)
        logit_mask_expand = logit_mask[:, None].expand(-1, num_samples, -1).reshape(-1, logit_mask.shape[-1])

        num_seqs = src_expand.shape[0]
        tokens_buf = torch.full(
            (num_seqs, max_length + 1), self.pad_token_id, dtype=torch.long, device=device
        )
        tokens_buf[:, 0] = self.bos_token_id

        seq_logits_buf = torch.empty(
            (num_seqs, max_length, len(self.vocab)), dtype=src_expand.dtype, device=device
        )
        seq_log_probs_buf = torch.empty(
            (num_seqs, max_length), dtype=src_expand.dtype, device=device
        )

        finished = torch.zeros(num_seqs, dtype=torch.bool, device=device)
        pad_mask_buf = torch.zeros((num_seqs, max_length + 1), dtype=torch.float, device=device)
        pad_mask_buf[:, 0] = finished.float()

        num_layers = len(self.decoder.layers)
        num_heads = self.decoder.layers[0].self_attn.num_heads
        d_k = self.decoder.layers[0].self_attn.d_k
        kv_cache = [
            {
                "k": torch.empty(num_seqs, num_heads, max_length, d_k, device=device, dtype=src_expand.dtype),
                "v": torch.empty(num_seqs, num_heads, max_length, d_k, device=device, dtype=src_expand.dtype),
                "len": 0,
            }
            for _ in range(num_layers)
        ]
        cross_kv_cache = self.decoder.precompute_cross_kv(src_expand)

        gen_len = 0
        for step in range(max_length):
            if not use_cache:
                raise NotImplementedError("Only KV-cached decoding is supported.")
            token = tokens_buf[:, step : step + 1]
            logits, kv_cache = self.decoder(
                tgt=token,
                memory=src_expand,
                tgt_mask=None,
                memory_mask=src_mask_expand,
                cache=kv_cache,
                cross_kv_cache=cross_kv_cache,
            )

            logits = logits[:, -1, :] - (1.0 - logit_mask_expand) * 1e9
            seq_logits_buf[:, step, :] = logits

            if temperature <= 0:
                log_probs = F.log_softmax(logits, dim=-1)
                next_token = logits.argmax(-1).unsqueeze(-1)
            else:
                tempered_logits = logits / temperature
                log_probs = F.log_softmax(tempered_logits, dim=-1)
                sample_logits = mask_with_top_k(
                    mask_with_top_p(tempered_logits, top_p=top_p), top_k=top_k
                )
                probs = F.softmax(sample_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            tokens_buf[:, step + 1 : step + 2] = next_token
            token_log_prob = torch.gather(log_probs, 1, next_token).squeeze(-1)
            seq_log_probs_buf[:, step] = token_log_prob

            finished = finished | (next_token.squeeze(-1) == self.eos_token_id)
            pad_mask_buf[:, step + 1] = finished.float()
            gen_len = step + 1

            if finished.all():
                break

        seq_logits = seq_logits_buf[:, :gen_len, :].reshape(bsz, num_samples, gen_len, -1)
        seq_log_probs = seq_log_probs_buf[:, :gen_len].reshape(bsz, num_samples, gen_len)

        pad_mask_ = pad_mask_buf[:, : gen_len + 1]
        pad_mask = torch.zeros_like(pad_mask_)
        pad_mask[:, 1:] = pad_mask_[:, :-1]

        tokens = tokens_buf[:, : gen_len + 1]
        tokens = (tokens * (1 - pad_mask) + pad_mask * self.pad_token_id).long()
        tokens = tokens.reshape(bsz, num_samples, -1)

        sequences = self.decode_sequence(tokens)
        return tokens, sequences, seq_logits, seq_log_probs

    @torch.no_grad()
    def __call__(
        self,
        batch: dict,
        temperature: float = 1.0,
        max_length: int = 512,
        num_samples: int = 1,
        top_k: int = 0,
        top_p: float = 1.0,
        use_cache: bool = True,
        max_chunk_size: Optional[int] = None,
    ):
        """Autoregressively sample ``num_samples`` candidate circuits per batch item.

        Returns:
            tokens:        LongTensor [B, K, L + 1] (<bos> ... <eos> <pad>...)
            sequences:     List[List[str]] of shape [B][K]
            seq_logits:    Tensor [B, K, L, |V|] (masked pre-temperature logits)
            seq_log_probs: Tensor [B, K, L] (behavior-policy token log-probabilities)
            logit_mask:    Tensor [B, |V|] (qubit validity mask V_n)
        """
        if top_k > 0 and top_p < 1.0:
            raise ValueError("Cannot use both top-k and top-p sampling simultaneously.")

        self.model.eval()
        device = batch["src"].device
        if next(self.model.parameters()).device != device:
            self.model.to(device)

        dtype = next(self.model.parameters()).dtype
        if "graph" in batch and hasattr(batch["graph"], "x") and batch["graph"].x.dtype != dtype:
            batch["graph"].x = batch["graph"].x.to(dtype)

        src, src_mask = self.encoder(batch["src"], batch["src_len"], batch["graph"])
        bsz = src.shape[0]
        logit_mask = self.token_mask.to(src.device)[batch["num_qubits"] - 1]

        if max_chunk_size is None or num_samples <= max_chunk_size:
            tokens, seqs, seq_logits, seq_log_probs = self._decode_chunk(
                src, src_mask, logit_mask, num_samples, temperature, max_length, top_k, top_p, use_cache
            )
            return tokens, seqs, seq_logits, seq_log_probs, logit_mask

        all_tokens, all_seqs, all_logits, all_log_probs = [], [[] for _ in range(bsz)], [], []
        for chunk_start in range(0, num_samples, max_chunk_size):
            chunk_size = min(max_chunk_size, num_samples - chunk_start)
            c_tokens, c_seqs, c_logits, c_log_probs = self._decode_chunk(
                src, src_mask, logit_mask, chunk_size, temperature, max_length, top_k, top_p, use_cache
            )
            all_tokens.append(c_tokens)
            all_logits.append(c_logits)
            all_log_probs.append(c_log_probs)
            for sid, s_list in enumerate(c_seqs):
                all_seqs[sid].extend(s_list)

        max_tok_len = max(t.shape[2] for t in all_tokens)
        max_step_len = max_tok_len - 1
        vocab_size = all_logits[0].shape[-1]

        tokens = torch.full(
            (bsz, num_samples, max_tok_len), self.pad_token_id, dtype=all_tokens[0].dtype, device=src.device
        )
        seq_logits = torch.zeros(
            (bsz, num_samples, max_step_len, vocab_size), dtype=all_logits[0].dtype, device=src.device
        )
        seq_log_probs = torch.zeros(
            (bsz, num_samples, max_step_len), dtype=all_log_probs[0].dtype, device=src.device
        )

        offset = 0
        for t, lg, lp in zip(all_tokens, all_logits, all_log_probs):
            k = t.shape[1]
            tlen = t.shape[2]
            tokens[:, offset : offset + k, :tlen] = t
            seq_logits[:, offset : offset + k, : tlen - 1, :] = lg
            seq_log_probs[:, offset : offset + k, : tlen - 1] = lp
            offset += k

        return tokens, all_seqs, seq_logits, seq_log_probs, logit_mask
