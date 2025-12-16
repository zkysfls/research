# Benchmarking the Impact of Active Space Selection on the VQE Pipeline for Quantum Drug Discovery

> All kinds of discussions and suggestions are welcome.


  Quantum computers promise scalable treatments of electronic structure, yet applying variational
quantum eigensolvers (VQE) on realistic drug-like molecules remains constrained by the performance
 limitations of near-term quantum hardwares. A key strategy for addressing this challenge
which effectively leverages current Noisy Intermediate-Scale Quantum (NISQ) hardwares yet remains
 under-benchmarked is active space selection. We introduce a benchmark that heuristically
proposes criteria based on chemically grounded metrics to classify the suitability of a molecule for
using quantum computing and then quantiffes the impact of active space choices across the VQE
pipeline for quantum drug discovery. The suite covers several representative drug-like molecules
(e.g., lovastatin, oseltamivir, morphine) and uses chemically motivated active spaces. Our VQE
evaluations employ both simulation and quantum processing unit (QPU) execution using unitary
coupled-cluster with singles and doubles (UCCSD) and hardware-efffcient ansatz (HEA). We adopt
a more comprehensive evaluation, including chemistry metrics and architecture-centric metrics.
For accuracy, we compare them with classical quantum chemistry methods. This work establishes
the ffrst systematic benchmark for active space driven VQE and lays the groundwork for future
hardware-algorithm co-design studies in quantum drug discovery.

