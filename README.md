# 4DTopOpt: Source code for 4D topology optimization 

(Last update: May 23, 2024)

This code is for simultaneous optimization of the structure and self-actuation of soft robots using 4D topology optimization.

# Demo

https://youtu.be/sPY2jcAsNYs?si=muQL6DyJmcPshzLe

# Requirement

| Software  | Version |
| :---: | :---: |
| python  | 3.9.17  |
| taichi  | 1.6.0 |
| numpy  | 1.25.2 |
| numpy-ml  | 0.1.2 |
| matplotlib  | 3.7.2 |
| pyevtk  | 1.6.0 |
| pyyaml  | 6.0.1 |

Environment under [Anaconda for Windows](https://www.anaconda.com/distribution/) is tested.
Below is an example of the installation commands.

```bash
conda create -n 4dto_env python=3.9
activate 4dto_env
python -m pip install taichi==1.6.0
python -m pip install jupyter notebook==6.4.12
python -m pip install matplotlib
python -m pip install numpy-ml==0.1.2
python -m pip install gym
python -m pip install pyevtk
python -m pip install pyyaml
python -m pip install vtk
```

# Usage

Run "4dtopopt.py"

```bash
python 4dtopopt.py
```

# Citation

```
C. Yuhn, Y. Sato, H. Kobayashi, A. Kawamoto, T. Nomura, 4D topology optimization: Integrated optimization of the structure and self-actuation of soft bodies for dynamic motions, Computer Methods in Applied Mechanics and Engineering 414 (2023) 116187, https://doi.org/10.1016/j.cma.2023.116187.
```

In bibtex format:

```
@article{yuhn20234d,
  title={4D topology optimization: Integrated optimization of the structure and self-actuation of soft bodies for dynamic motions},
  author={Yuhn, Changyoung and Sato, Yuki and Kobayashi, Hiroki and Kawamoto, Atsushi and Nomura, Tsuyoshi},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={414},
  pages={116187},
  year={2023},
  publisher={Elsevier},
  doi={10.1016/j.cma.2023.116187}
}
```

# License

This project is licensed under the [Apache License Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

The portion of this code pertaining to the physical simulation of soft bodies using the material point method has been adapted from the Taichi sample code "[mpm3d.py](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/mpm3d.py)," which is provided under the Apache 2.0 license.

The original code has been modified for this project, including the replacement of the material model with a neo-Hookean hyperelastic solid model, as well as various minor additions and modifications.
