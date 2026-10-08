# Notices

This repository implements the tokenizer described in
[SupraTok: Cross-Boundary Tokenization for Enhanced Language Model Performance](docs/tacl.a.774.pdf)
(TACL vol. 14, pp. 1787–1802, 2026). The paper PDF is included under its CC BY 4.0 licence.

The implementation is MIT-licensed; see [LICENSE](LICENSE).

No corpora are redistributed. See [datasets/README.md](datasets/README.md) for how to obtain
Nemotron-CC, WikiText-103, Wikipedia and FLORES-200 from their original sources under their
own licences.

BPE merges are learned with [HuggingFace `tokenizers`](https://github.com/huggingface/tokenizers)
(Apache 2.0).
