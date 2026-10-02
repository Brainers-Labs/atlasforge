# Glossary

**Adapter.** A small set of extra weights (here, LoRA) that sits on top of a base model. Fine-tuning with AtlasForge produces an adapter, never a copy of N-ATLaS.

**ASR.** Automatic speech recognition: turning audio into text.

**Backend.** The thing that answers prompts and transcribes audio: an OpenAI-compatible server (`openai`) or the model loaded in-process (`local`).

**Base model.** The original model before fine-tuning, here `NCAIR1/N-ATLaS`.

**Bootstrap.** Estimating how uncertain a result is by repeatedly resampling the data. AtlasForge uses a *paired* bootstrap over examples.

**CER.** Character error rate: the fraction of characters wrong in a transcript. Lower is better.

**chrF.** A score based on overlapping character sequences, on a 0-100 scale. Higher is better.

**Circuit breaker.** The rule that stops a run after too many failures in a row, so a dead server is not retried for hours.

**Confidence interval (CI).** The range a true value plausibly lies in. AtlasForge reports 95% intervals.

**Diacritics.** Marks added to letters: tone marks, dots and hooks. In these languages they can change a word's meaning.

**Fingerprint.** A SHA-256 hash of a file. AtlasForge records the dataset's fingerprint in every run so runs from different data are never compared.

**Gated model.** A model whose files you can download only after accepting its licence on Hugging Face.

**GGUF.** A file format for running quantised models with llama.cpp and Ollama.

**Held-out set.** Data kept aside and never trained on, used to test a model fairly.

**JSONL.** JSON Lines: one JSON object per line. The dataset format.

**Leakage.** Examples appearing in both training and test data, which makes an evaluation measure memorisation.

**LoRA.** Low-rank adaptation: a way to fine-tune a model by training small extra matrices instead of all its weights.

**McNemar test.** An exact test for whether two models, scored right or wrong on the same examples, differ.

**Macro-F1.** F1 score averaged over classes, so rare classes count as much as common ones.

**NFC.** A standard Unicode form in which a letter with marks is stored as few characters as possible. AtlasForge converts text to NFC before comparing it, so the same text stored two ways compares equal.

**Pooled.** A metric computed over the whole corpus at once, as opposed to averaged per example.

**QLoRA.** LoRA on a model compressed to 4-bit precision so it fits in less GPU memory. Needs an NVIDIA GPU.

**Quantisation.** Storing a model's numbers at lower precision (such as 4-bit) to save memory.

**Regression.** A place where the new model is worse than the old one, even if it is better overall.

**Run directory.** The folder `eval` creates for one model's pass over one dataset.

**Slice.** A subset of a dataset (a language, a length, a domain) whose results are reported separately.

**Tone marks.** Marks (grave, acute, macron and similar) that show tone. Yoruba and Igbo use them heavily. Different from the dot-below, which makes a different letter.

**Tone-aware / tone-insensitive.** The two scoring views: tone marks kept, or ignored.

**Underdot.** The dot below a letter in Yoruba and Igbo (`ẹ`, `ọ`, `ṣ`, `ị`, `ụ`). It makes a different letter and is never stripped.

**vLLM, llama.cpp, Ollama.** Programs that serve a language model over a network, with an OpenAI-compatible API.

**WER.** Word error rate: the fraction of words wrong in a transcript. Lower is better. Can exceed 100%.
