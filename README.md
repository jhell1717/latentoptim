# Enabling Sample Efficient Design Space Exploration through Machine Learning

<a href="https://colab.research.google.com/github/jhell1717/latentoptim/blob/main/examples/ga_examples/run_GA.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Abstract
In engineering, design problems are often complex with components composed of intricate parts and features that define the functionality or performance of the component. One of the key driving activities to enable engineering innovation is design exploration and optimisation, referring to the ways in which potential designs are identified that result in the most optimal performance being realised in context of an objective. To determine this, and as is common in industry, numerical tools such as Computational Fluid Dynamics (CFD) or Finite Element Analysis (FEA) are used to provide performance data associated with engineering designs (e.g., lift performance for an aerofoil or heat losses for a heat exchanger). These tools are computationally expensive and not suited to fast, iterative design exploration, attributed to the computational cost of solving physical equations to provide numerical data. They are typically used towards the end of a design process, downstream of opportune points to iterate or change designs. This computational burden generally increases with added complexity in the inputs, such as number of elements in unstructured meshes, point clouds or voxels.

In contrast, optimisation methods rely on our ability to parameterise the problem in an efficient way. While simpler shapes such as tubes or rings can be easily parametrised by properties such as radius or thickness to enable a well-defined set of search dimensions that correspond to different design configurations, more complex shapes such as turbo-machinery components, or intricate heat exchange surfaces cannot be concisely described by a finite set of parameters and reside in high-dimensional geometry representations. As part of this research, supported by a comprehensive literature review, we will investigate approaches to representing high-dimensional engineering problems in a low-dimensional space to overcome the \textit{curse of dimensionality}, therefore enabling efficient design space exploration and optimisation. Identified methods will be implemented to develop a framework for embedding such representations into a proof-of-principle geometric optimisation problem. One of the key research objective will be to investigate and asses the interpretability and controllability of the low-dimensional representations to enable targeted design modification and optimisation in this intermediate low latent space manifold.

## Project Objectives

1. **Research and Compare AI Models**: Explore and evaluate different AI approaches for encoding geometries into low-dimensional spaces, including:

   - Autoencoders (AEs)
   - Generative Adversarial Networks (GANs)
   - Diffusion Models
   - Implicit Neural Representation.

2. **Facilitate Design Space Exploration**: Assess how these learned representations can enable more efficient exploration of geometric quantities of interest for optimisation tasks.

3. **Analyse Geometry Representations**: Evaluate the structure, smoothness, and interpretability of the learned geometry representations and their suitability for:

   - Gradient-based optimisation
   - Genetic algorithms
   - Bayesian optimisation

4. **Optimisation Task**: Implement the learned representations on a 2D optimisation problem—specifically, minimising the area-to-perimeter ratio. While this serves as a simplified analogy to real-world engineering challenges (e.g., optimising lift-to-drag ratios in aerospace), the methodologies can be extended to other domains.

## Why Focus on 2D Geometries?

While industry applications typically focus on 3D geometries, the computational overhead and data availability present unnecessary challenges that obscure the core research aims. By focusing on 2D shapes, this project can systematically explore and compare AI-based encoding methods without being constrained by the computational demands of 3D simulations.

## Optimisation Problem
As part of this research project, a simple optimisation problem will be used to demonstrate the limitations and principles of design space optimisation and the benefits realised by moving the optimisation  problem to a low-dimensional latent space. 

The optimisation problem chosen will be Perimeter-to-Area ratio of 2D shapes. This is analogous to a more commonly optimised metric of **Surface-Area-to-Volume** in 3-dimensions. The **Perimeter-to-Area Ratio** is a geometric measure that describes the relationship between the perimeter and the area of a 2D shape. It is commonly used to analyze the efficiency of shapes in applications like heat transfer, where minimizing the ratio can reduce heat loss.

### Formula:

```
P/A Ratio = P / A
```

Where:
- **P** = Perimeter of the shape (e.g., in meters)
- **A** = Area of the shape (e.g., in square meters)

### Significance:

- **Higher P/A Ratio**: Greater heat loss (e.g., thin or irregular shapes lose heat faster).
- **Lower P/A Ratio**: Reduced heat loss (e.g., more compact shapes like circles are thermally efficient).

### Example Calculation (Circle):

For a circle with radius **r**:

```
P = 2πr
A = πr²
P/A Ratio = (2πr) / (πr²) = 2 / r
```

This shows that as the radius increases, the **P/A ratio** decreases, reducing the rate of heat loss.


## Significance of the Research

Generative models have become pivotal in engineering due to their ability to compress high-complexity processes into a manageable, low-dimensional search space. This research aims to:

- Identify which generative models best capture and represent complex geometries.
- Demonstrate how low-dimensional representations facilitate efficient design optimisation.
- Provide insights into interpretable and controllable geometric representations that support targeted design modification.

The outcomes of this project aim to advance the use of AI-driven geometry encoding for engineering design, offering scalable solutions for rapid and efficient design exploration.

## Interpreting the Latent Space
Explainable AI is a sub field of AI that aims to demystify typically black box models, providing human interpretable explanations for predictions and decisions made by such models. In the field of generative modelling, the latent spaces generated by models such as **Variational Auto Encoders (VAE)**, are inherently abstract and are a hierarchical combination of features that preceed them. For this reason, their meaning is not directly interpretable. However, when coupling the latent variables to an optimisation problem such as that described above, a sensitivity study can be performed to gain insights into the effect of 'adjusting' latent variables, or by interpolating through the latent 'space', on the target optimisation metric (e.g., **Perimeter-to-Area-Ratio**)

## Some Motivating Literature
- Tripp, A., Daxberger, E., & Hernández-Lobato, J. M. (2020). Sample-efficient optimization in the latent space of deep generative models via weighted retraining. Advances in Neural Information Processing Systems, 2020-December.
- Zheng, L., Karapiperis, K., Kumar, S., & Kochmann, D. M. (2023). Unifying the design space and optimizing linear and nonlinear truss metamaterials by generative modeling. Nature Communications, 14(1). https://doi.org/10.1038/s41467-023-42068-x
- Siivola, E., Paleyes, A., González, J., & Vehtari, A. (2021). Good practices for Bayesian optimization of high dimensional structured spaces. In Applied AI Letters (Vol. 2, Issue 2). https://doi.org/10.1002/ail2.24
- Danhaive, R., & Mueller, C. T. (2021). Design subspace learning: Structural design space exploration using performance-conditioned generative modeling. Automation in Construction, 127. https://doi.org/10.1016/j.autcon.2021.103664

