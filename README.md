# DIScriminator DisAgreement INtrinsic Reward (DISDAIN), a self-contained JAX implementation

This is a simplified version of the code used in [Learning more skills through optimistic exploration][ICLR2022] (appearing at ICLR 2022).

This Colab trains an agent with a tabular Q function and a tabular discriminator ensemble on a scaled down version of the Four Rooms environment. It will parallelize across all available devices. We recommend training on a GPU backend or Colab Pro TPU backend.

The environment has 24 states. With 8 transitions, all but one state is reachable from the initial state in the top left corner. This means that at most 23 distinguishable skills can be learned.

With the default hyperparameters on a single accelerator, skill learning with DISDAIN achieves approximately 15 effective skills in 500,000 steps and 21 effective skills in 1,000,000 steps, while a matched hyperparameter baseline (with discriminator ensemble disabled) attains approximately 11 effective skills through the course of training (approximately 12 if deriving rewards from an ensemble average, without the DISDAIN bonus). Each agent trains in approximately 12 minutes on the default GPU backend.

This implementation broadly matches the setting of the Four Rooms experiments from the paper, with the following differences:

* the Four Rooms grid world has been scaled down;
* trajectories are generated online, rather than placed in and sampled from a replay buffer;
* the learning rate and bonus weight have been re-tuned in light of the above.


## Installation

Simply open the file in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepmind/disdain/blob/master/disdain.ipynb)
and run the cells in order.

## Citing this work

BibTeX for citing the DISDAIN paper:

```bibtex
@inproceedings{
  strouse2022learning,
  title={Learning more skills through optimistic exploration},
  author={DJ Strouse and Kate Baumli and David Warde-Farley and Volodymyr Mnih and Steven Stenberg Hansen},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=cU8rknuhxc}
}
```

## License and disclaimer

Copyright 2022 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
