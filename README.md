# Submission to the KTC 2023

For this submission we developed a **conditional diffusion model** for EIT segmentation. The conditional diffusion model is trained on a dataset of synthetic phantoms and simulated measurements.

# Table of contents 
1. [Usage](#usage)
2. [Method](#method)
3. [Examples](#examples)
4. [Evaluation](#evaluation)
5. [Authors](#authors)

## Usage

We provide the `enviroment.yml` file to restore the conda enviroment used for the submission. You can create the enviroment using the following command:

```
conda env create -f environment.yml
```

The network weights are stored [here](https://seafile.zfn.uni-bremen.de/d/59c291e4bf7d4064a1be/). They have to be stored in *diffusion_models/level_{level}/model.pt*.  We precomputed the Jacobian for an empty watertank, as well as some other matrices (smoothness regulariser, node coordinates). This eliminates the need to install Fenics in the enviroment. All of these matrices are available [here](https://seafile.zfn.uni-bremen.de/d/9108bc95b2e84cd285f8/). and have to be stored in *data/*.


The script `main.py` can be used reconstruct phantoms: 

```
python main.py /path_to_input_folder /path_to_ouput_folder difficulty_level
```


### Enviroment


## Method

Our goal is to train a [conditional diffusion model](https://arxiv.org/abs/2111.13606), 

$$ s_\theta(\sigma, t, c) \approx \nabla_\sigma \log p_t(\sigma | c) $$

to approximate the conditional score function. Here, $\sigma$ denotes the conductivity map (interpolated to the $256 \times 256$ pixel grid), $t$ the current time step and $c$ the conditional input. Note, that we do not use the raw measurements $U$ directly as the conditional input. Instead, we use an initial reconstruction method $\mathcal{R}$ and make use the initial reconstruction $\mathcal{R}(U)$. Further, this initial reconstruction is interpolated to the $256 \times 256$ pixel grid. This has the practical advantage that we can implement the diffusion model as a convolutional neural network and are independent of the underlying mesh. 

### Initial Reconstructions

We use linearised time-difference reconstructions for the conditional input. Computing this linearised time-difference reconstruction amounts to solving a regularised least squares problem

$$ \Delta \sigma = (J_{\sigma_0} \Gamma_e^{-1} J + \alpha_1 R_\text{SM} + \alpha_2 R_\text{Laplace} + \alpha_3 R_\text{NOSER})^{-1} J_{\sigma_0}^T \Gamma_e^{-1} \Delta U, $$

where $J_{\sigma_0}$ is the Jacobian w.r.t. to a constant background conductivity, $\Gamma_e^{-1}$ is the noise precision matrix and $\Delta U = U^\delta - F(\sigma_0)$ is the difference of the measurements. For $F(\sigma_0)$ we make use of the provided measurements of the empty water tank. 

We use a combination of three different priors, where $R_\text{SM}$ denotes a [smoothness prior](https://www.fips.fi/KTC2023_Instructions_v3_Oct12.pdf), $R_\text{Laplace}= - \bigtriangleup$ the graph laplacian and $R_\text{NOSER} = diag(J_{\sigma_0} \Gamma_e^{-1} J)$ the [NOSER](https://pubmed.ncbi.nlm.nih.gov/36909677/) prior. 

In total, we use five different combinations of $\alpha_1, \alpha_2$ and $\alpha_3$ as each regularisation results in different artifacts in the reconstruction. This means, that our conditional input, interpolated to the pixel grid, is of the size $5 \times 256 \times 256$. 

### Training

### Forward Operator 

For simulation, we used the forward operator provided by the organisers with the dense mesh. For the reconstruction process, we implemented the complete electrode model in Fenics


### Synthetic Training Data


## Examples

## Evaluation


We evaluate the conditional diffusion model w.r.t. the [score function](https://www.fips.fi/KTC2023_Instructions_v3_Oct12.pdf) used in the challenge. 


| Level         |    Score       |
|---------------|----------------|
| 1            | $X.XXX$       |
| 2            | $X.XXX$       |
| 3            | $X.XXX$       |
| 4            | $X.XXX$       |
| 5            | $X.XXX$       |
| 6            | $X.XXX$       |
| 7            | $X.XXX$       |


## Authors

- Alexander Denker<sup>1</sup>, Tom Freudenberg<sup>1</sup>, Željko Kereta<sup>2</sup>, Imraj RD Singh<sup>2</sup>, Tobias Kluth<sup>1</sup>, Simon Arridge <sup>2</sup>

<sup>1</sup>Center of Industrial Mathematics (ZeTeM), University of Bremen

<sup>2</sup>Department of Computer Science, University College London