"""Local instruction-tuned LLM wrapper (HuggingFace transformers) with batched generation."""
import time
import torch


class LocalLLM:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str | None = None,
                 max_context: int = 24000):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(self.device)
        self.model.eval()
        self.max_context = max_context

    # -- tokens --------------------------------------------------------------
    def count_tokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def truncate_to_tokens(self, text: str, budget: int, keep: str = "tail") -> str:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= budget:
            return text
        ids = ids[-budget:] if keep == "tail" else ids[:budget]
        return self.tok.decode(ids)

    def _render(self, system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    # -- generation ----------------------------------------------------------
    long_prompt_tokens: int = 2048   # prompts longer than this are run one at a time with chunked prefill
    prefill_chunk: int = 1024

    @torch.inference_mode()
    def _generate_long(self, ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Chunked prefill so that a 16k-token prompt never materialises a full attention matrix
        (needed on a 4 GB GPU); then greedy decoding continues from the filled KV cache."""
        from transformers import DynamicCache
        cache = DynamicCache()
        n = ids.shape[1]
        for s in range(0, n - 1, self.prefill_chunk):
            e = min(s + self.prefill_chunk, n - 1)
            self.model(input_ids=ids[:, s:e], past_key_values=cache, use_cache=True)
        return self.model.generate(input_ids=ids, past_key_values=cache, max_new_tokens=max_new_tokens,
                                   do_sample=False, pad_token_id=self.tok.pad_token_id)

    @torch.inference_mode()
    def chat_batch(self, system: str, users: list[str], max_new_tokens: int = 32,
                   batch_size: int = 8) -> list[dict]:
        """Greedy generation for a list of user prompts sharing one system prompt.
        Returns dicts with answer, prompt_tokens, latency (seconds, amortised over the batch)."""
        results: list[dict] = [None] * len(users)
        rendered = [self._render(system, u) for u in users]
        lengths = [len(self.tok(r, add_special_tokens=False)["input_ids"]) for r in rendered]
        short = [i for i in range(len(users)) if lengths[i] <= self.long_prompt_tokens]
        long = [i for i in range(len(users)) if lengths[i] > self.long_prompt_tokens]
        # sort by length so that padding inside a batch is small
        short.sort(key=lambda i: lengths[i])
        for start in range(0, len(short), batch_size):
            idxs = short[start:start + batch_size]
            texts = [rendered[i] for i in idxs]
            enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                           max_length=self.max_context).to(self.device)
            t0 = time.time()
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
            dt = (time.time() - t0) / len(idxs)
            n = enc["input_ids"].shape[1]
            for j, i in enumerate(idxs):
                ans = self.tok.decode(out[j][n:], skip_special_tokens=True).strip()
                ptoks = int(enc["attention_mask"][j].sum().item())
                results[i] = {"answer": ans, "prompt_tokens": ptoks, "latency": dt}
        for i in long:
            enc = self.tok(rendered[i], return_tensors="pt", truncation=True,
                           max_length=self.max_context).to(self.device)
            t0 = time.time()
            out = self._generate_long(enc["input_ids"], max_new_tokens)
            dt = time.time() - t0
            n = enc["input_ids"].shape[1]
            ans = self.tok.decode(out[0][n:], skip_special_tokens=True).strip()
            results[i] = {"answer": ans, "prompt_tokens": n, "latency": dt}
            if self.device == "cuda":
                torch.cuda.empty_cache()
        return results

    def chat(self, system: str, user: str, max_new_tokens: int = 32) -> dict:
        return self.chat_batch(system, [user], max_new_tokens=max_new_tokens, batch_size=1)[0]
