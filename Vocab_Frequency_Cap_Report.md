## Vocabulary Frequency Cap Report

**Diagnosis:**
The previous dynamic tokenization scheme recorded every unique word variation (including typos and punctuation-attached words), leading to a ballooned vocabulary size of 238,972. This resulted in an oversized 2 GB logits tensor, drastically decreasing training speed and risking severe VRAM OOM faults on an RTX 4060.

**Fix Execution:**
1. **Two-Pass Stream Tokenization:** I redesigned 	ools/preprocess.cpp to use a robust two-pass methodology:
   * **Pass 1 (Frequency Analysis):** It performs a high-speed iteration over the 2.2 GB corpus, utilizing an std::unordered_map to calculate absolute word frequencies.
   * **Pass 2 (Capped Encoding):** After sorting the frequencies in descending order, the token dictionary is locked to exactly the Top 8,188 frequent words (plus 4 special tokens). During the second stream iteration, any word falling outside this elite tier is forcefully re-mapped to <unk> (1).
2. **Architecture Conformity:** main.cu natively inherits this optimized V = 8192 dimension from data/vocab.txt, mathematically shrinking the final logits tensor and drastically reducing the PCIe bandwidth requirements and VRAM footprint.

**Current Status:**
The two-pass preprocessor has successfully executed across the TinyStories_train.txt corpus. It built the dense 8,192-word vocabulary and mapped all out-of-bounds variations to the unknown token. I have recompiled cuda_engine.exe and initiated the optimized eager training loop.
