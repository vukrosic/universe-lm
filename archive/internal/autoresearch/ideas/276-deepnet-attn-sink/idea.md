---
id: 276-deepnet-attn-sink
author: claude-opus-4-8
status: done
round: 1
updated: 2026-06-16T05:12:54Z
transfer-risk: low
plain: Stack on new champion: alibi + deepnet + use_attn_sink (learned attention bias toward a fixed token, sinks attention probability mass). Step-0-active.
---

# 276 — deepnet + use_attn_sink on alibi

Stack on the new champion. Attention sink: each attention head has a learned scalar bias toward a specific token position, absorbing probability mass from the rest of the distribution. The mechanism is similar to the "attention sink" pattern observed in streaming LMs.

A/B vs new champion val 6.2367, band 0.04, WIN < 6.1967. Single seed (42).

See `_arq_276-deepnet-attn-sink.py` for inline config.
