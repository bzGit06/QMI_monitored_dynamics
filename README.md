# QMI_monitored_dynamics
This repository contains the official Python implementation of [*Scaling Laws of Quantum Information Lifetime in Monitored Quantum Dynamics*](https://arxiv.org/abs/2506.22755), an article by [Bingzhi Zhang](https://sites.google.com/view/bingzhi-zhang/home), [Fangjun Hu](https://scholar.google.com/citations?user=81QGlDQAAAAJ&hl=en), [Runzhe Mo](https://scholar.google.com/citations?user=WYfkmaoAAAAJ&hl=en&oi=ao), Tianyang Chen, [Hakan E. Türeci](https://turecigroup.princeton.edu/) and [Quntao Zhuang](https://sites.usc.edu/zhuang).

## Citation
```
@misc{zhang2023dynamical,
      title={Scaling laws of QUantum Information Lifetime in Monitored Quantum Dynamics}, 
      author={Zhang, Bingzhi and Hu, Fangjun and Mo, Runzhe and Chen, Tianyang and Türeci, Hakan E.  and Zhuang, Quntao},
      year={2025},
      eprint={2506.22755},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```

## Prerequisite
The simulation of quantum circuits is performed via the [TensorCircuit](https://tensorcircuit.readthedocs.io/en/latest/#) package with [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) backend. Use of GPU is not required, but highly recommended. 

Additionally, the package [Qiskit](https://github.com/Qiskit/qiskit) is needed for experiments on IBM Quantum device.


## File Structure
The file `memory_time.ipynb` contains most numerical results. The file `memory_witness.ipynb` contains presentation of noisy simulation and experimental results on IBM Quantum devices. The file `memory_noisy_simulation_experiment.ipynb` contains experimental results on noisy simulation and IBM Quantum device. 
