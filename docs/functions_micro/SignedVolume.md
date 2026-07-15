---
name: SignedVolume
title: Signed Volume
implementation_family: micro
topics:
- microstructure
tags:
- signed volume
- order flow
- flow
- microstructure
short: Aggressor-signed volume, sign * volume.
inputs: 2
outputs: 1
parameters: []
nan_policy: ignore
---

# `SignedVolume`

## Description

`SignedVolume` multiplies a trade sign by volume to give aggressor-signed order
flow. Pair a sign source (`TickRuleSign`, or an aggressor flag) with volume, then
feed the result to `RollingKyleLambda` for price impact.
