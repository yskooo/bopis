## Same Dolly prompt, two configurations

This is a single-prompt illustration using the same `/completion` endpoint,
study template, and no system prompt. The only selected-configuration
difference shown here is the generation limit: 256 tokens for Random Search and
1024 tokens for BOPIS.

| Condition | Configuration | Energy | Speed | BERTScore F1 | Answer length |
| --- | --- | ---: | ---: | ---: | ---: |
| Random Search pick | `qwen2.5-0.5b_t256_b1_Q4_K_M_g14_c4` | ~225.8 J | 3.6 tok/s | **0.017** | 256 tok |
| BOPIS x* | `qwen2.5-0.5b_t1024_b1_Q4_K_M_g14_c4` | ~1020.0 J | **4.8 tok/s** | -0.174 | 1024 tok |

## Comparison analysis

### What the numbers show

- **Energy: Random Search leads.** BOPIS uses about **794.2 J more**, or
	approximately **4.5x** the total energy of the Random Search pick. In this
	example, the longer 1024-token generation budget dominates the energy cost.
- **Token-generation speed: BOPIS leads.** BOPIS produces **4.8 tok/s versus
	3.6 tok/s**, a gain of about **33.3%**. This means the selected configuration
	has better throughput while it is generating tokens.
- **Total response time: Random Search is still faster for this prompt.** The
	approximate generation time is **71.1 s** for Random Search
	($256 / 3.6$) versus **213.3 s** for BOPIS ($1024 / 4.8$). Thus, a higher
	tok/s rate does not compensate for producing four times as many tokens.
- **Energy efficiency per generated token: Random Search also leads.** Random
	Search uses approximately **0.88 J/token**, while BOPIS uses approximately
	**1.00 J/token**. BOPIS is therefore about **12.9% less efficient per token**
	in this illustration.
- **Quality: Random Search leads on BERTScore F1.** Its score is **0.017**,
	compared with **-0.174** for BOPIS, a difference of **0.191 BERTScore
	points** in favor of Random Search. Because both values are low and this is
	only one prompt, this should be described as an observed sample difference,
	not a general conclusion about answer quality.

### Interpretation

For this particular prompt, **Random Search is the better overall choice when
the priority is lower energy, shorter completion time, and the observed
reference similarity**. **BOPIS is better only on raw token-generation speed**
in this side-by-side example. Its selected configuration allows more output,
but that extra output increases total energy and latency and does not produce a
higher BERTScore on this prompt.

This does not mean that Random Search is generally superior to BOPIS. The two
rows use different token limits, so the comparison is a trade-off between a
shorter 256-token configuration and a longer 1024-token configuration. BOPIS
may select a longer generation budget because its objective and constraints
consider a multi-objective balance across the study workload. The appropriate
question for the final experiment is therefore not simply which row has the
largest single metric, but whether BOPIS achieves a better **energy-speed-
quality trade-off on average across the same 500 prompts**.

### Thesis-ready takeaway

> In this illustrative Dolly prompt, Random Search produced a shorter and more
> energy-efficient response, while BOPIS achieved higher token throughput. The
> BOPIS configuration consumed approximately 4.5 times more total energy and
> had a lower observed BERTScore F1, although it generated tokens 33.3% faster.
> Therefore, this example demonstrates a measurable trade-off rather than a
> definitive winner. The 500-prompt validation is required to determine whether
> BOPIS provides a statistically reliable improvement in the combined
> energy-speed-quality objective.

### Important limitation

This is **one prompt only**, so it is useful for explaining the comparison but
not for proving that either method wins. The final claim should use the matched
500-prompt validation, report mean or median differences, and include the
planned statistical comparison. Also label the energy values as estimated when
they come from the project's estimated energy mode.