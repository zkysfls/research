# SMILES-Inspired Transfer Learning for Quantum Operators in Generative Quantum Eigensolver

> Ref: https://arxiv.org/abs/2509.19715

  Given the inherent limitations of traditional Variational Quantum Eigensolver(VQE) algorithms, the integration of deep generative models into hybrid quantum-classical frameworks, specifically the Generative Quantum Eigensolver(GQE), represents a promising innovative approach. However, taking the Unitary Coupled Cluster with Singles and Doubles(UCCSD) ansatz which is widely used in quantum chemistry as an example, different molecular systems require constructions of distinct quantum operators. Considering the similarity of different molecules, the construction of quantum operators utilizing the similarity can reduce the computational cost significantly. Inspired by the SMILES representation method in computational chemistry, we developed a text-based representation approach for UCCSD quantum operators by leveraging the inherent representational similarities between different molecular systems. This framework explores text pattern similarities in quantum operators and employs text similarity metrics to establish a transfer learning framework. Our approach with a naive baseline setting demonstrates knowledge transfer between different molecular systems for ground-state energy calculations within the GQE paradigm. This discovery offers significant benefits for hybrid quantum-classical computation of molecular ground-state energies, substantially reducing computational resource requirements.


## How to use the code

### 1.train
- Change the molecule in `molecule_data = generate_molecule_data("H2", use_ucc=True)` in `training/train_gptqe.py`.

- Molecules can be from https://pennylane.ai/datasets/collection/qchem

- run `python main.py`

- ckpt files will be saved at `checkpoints/*`


### 2.predict
- `python predict_gptqe.py --model_path ./checkpoints/model.pt --molecule H2`


### 3.transfer

- without fine-tune: `python transfer_gptqe.py --model_path ./checkpoints/model.pt --source_molecule H2 --target_molecule H4 --n_sequences 200`

- with fine-tune: `python transfer_gptqe.py --model_path ./checkpoints/model.pt --source_molecule H2 --target_molecule H4 --n_sequences 200 --fine_tune --fine_tune_epochs 200 --save_model`
