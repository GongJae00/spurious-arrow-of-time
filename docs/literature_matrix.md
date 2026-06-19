# Literature Matrix

This matrix records the primary literature lines that bound the current project.
The goal is not to claim that no related work exists. The goal is to identify
the precise gap:

```text
When the task itself is an irreversible inverse problem, a stronger but
non-causal irreversible process can become the wrong evidence.
```

## Arrow Of Time And Temporal-Order Self-Supervision

| Work | Question | Data/problem | Target signal | Role of irreversibility | Gap for this project |
|---|---|---|---|---|---|
| Pickup et al., "Seeing the Arrow of Time", CVPR 2014, https://openaccess.thecvf.com/content_cvpr_2014/html/Pickup_Seeing_the_Arrow_2014_CVPR_paper.html | Can a video be recognized as forward or backward? | Natural video clips | Forward vs reversed video | Irreversibility is the target cue | Does not ask whether the most visible arrow is non-causal for a separate inverse task. |
| Misra et al., "Shuffle and Learn", ECCV 2016, https://arxiv.org/abs/1603.08561 | Can temporal order verification learn useful visual features? | Video frame sequences | Correct vs shuffled order | Temporal order is a self-supervised proxy | Does not construct a competing irreversible nuisance that is predictive in train and wrong OOD. |
| Wei et al., "Learning and Using the Arrow of Time", CVPR 2018, https://openaccess.thecvf.com/content_cvpr_2018/html/Wei_Learning_and_Using_CVPR_2018_paper.html | What cues make video look forward or backward, and can they support activity analysis? | Large-scale video clips | Forward vs backward | Arrow evidence is useful supervision | Does not study hidden-cause recovery through an irreversible forward process. |
| Dave et al., "No More Shortcuts: Realizing the Potential of Temporal Self-Supervision", AAAI 2024, https://ojs.aaai.org/index.php/AAAI/article/view/27913 | Why do temporal self-supervised tasks saturate or use shortcuts? | Video SSL tasks | Temporal pretext labels | Temporal shortcuts are failure modes for representation learning | Shows temporal shortcuts exist, but not the specific setting where a nuisance irreversible mechanism competes with a task-causal irreversible inverse mechanism. |

## Thermodynamic Arrow And Entropy-Production Learning

| Work | Question | Data/problem | Target signal | Role of irreversibility | Gap for this project |
|---|---|---|---|---|---|
| Kim et al., "Learning Entropy Production via Neural Networks", PRL 2020, https://arxiv.org/abs/2003.04166 | Can entropy production be estimated from trajectories? | Stochastic thermodynamic systems | Entropy production | Irreversibility is the physical quantity to estimate | Does not study task prediction under a spurious irreversible nuisance. |
| Kim et al., PRL version, https://link.aps.org/doi/10.1103/PhysRevLett.125.140604 | Same as above, peer-reviewed publication | Stochastic processes | Entropy production | Irreversibility is the target | This project must not present learned task scores as physical entropy production. |
| Kwon and Baek, "alpha-divergence Improves the Entropy Production Estimation via Machine Learning", https://arxiv.org/abs/2303.02901 | Can ML entropy-production estimators be improved? | Stochastic trajectories | Entropy production estimator | Irreversibility is estimated | Orthogonal to causal task relevance. |

## Spurious Correlation, Invariance, And Time-Series OOD

| Work | Question | Data/problem | Target signal | Role of irreversibility | Gap for this project |
|---|---|---|---|---|---|
| Arjovsky et al., "Invariant Risk Minimization", https://arxiv.org/abs/1907.02893 | Can invariant predictors recover stable causal correlations across environments? | Synthetic and supervised OOD settings | Stable label relation | Irreversibility is not central | Provides an OOD frame, but not the irreversible inverse process question. |
| Gulrajani and Lopez-Paz, "In Search of Lost Domain Generalization", https://arxiv.org/abs/2007.01434 | How reliable are domain generalization algorithms? | DG benchmarks | OOD accuracy | Irreversibility is not central | Warns against overclaiming algorithmic wins; motivates strong benchmark gates. |
| Wu et al., "Out-of-Distribution Generalization in Time Series: A Survey", https://arxiv.org/abs/2503.13868 | What methods and settings exist for time-series OOD? | Time-series OOD literature | Generalization under distribution shift | Irreversibility is one possible dynamic property, not the focus | The project must distinguish "spurious arrow" from generic temporal covariate shift. |

## Inverse Diffusion, Inverse Heat, And Source Localization

| Work | Question | Data/problem | Target signal | Role of irreversibility | Gap for this project |
|---|---|---|---|---|---|
| Beck, Blackwell, and St. Clair, "Analysis and solution of the ill-posed inverse heat conduction problem", https://impact.ornl.gov/en/publications/analysis-and-solution-of-the-ill-posed-inverse-heat-conduction-pr/ | How can unknown boundary heat quantities be inferred from measurements? | Inverse heat conduction | Hidden causes/boundary quantities | Diffusion/heat flow causes ill-posed inverse recovery | Does not study neural shortcuts from a competing irreversible nuisance process. |
| Ling et al., "Source Localization of Graph Diffusion via VAEs for Graph Inverse Problems", https://arxiv.org/abs/2206.12327 | How can sources be inferred from diffused graph observations? | Graph diffusion source localization | Diffusion source | Diffusion makes source recovery ill-posed | Does not add a spurious irreversible mechanism whose arrow is correlated with labels only in train. |
| Huang et al., "Two-stage Denoising Diffusion Model for Source Localization in Graph Inverse Problems", NeurIPS 2023, https://proceedings.neurips.cc/paper_files/paper/2023/hash/46ab9d9645b6975b947231ddb48da1ab-Abstract-Conference.html | How can diffusion models solve graph source localization? | Graph inverse problems | Source reconstruction | Source localization is inverse to graph information spread | Method targets inverse recovery, not spurious arrow reliance. |
| Wang and Zhao, "GraphSL", https://arxiv.org/abs/2405.03724 | How should graph source localization methods and datasets be organized? | Source localization library and benchmarks | Diffusion sources | Source localization is framed as inverse graph diffusion | Useful foundation for benchmark framing; does not isolate competing task-causal vs nuisance arrows. |

## Current Gap

The defensible gap is narrow:

```text
Existing work learns time direction, estimates entropy production, studies
spurious correlations, or solves inverse diffusion/source localization.
The missing controlled question is whether a neural predictor chooses the
task-causal irreversible process or a stronger non-causal irreversible process
when both are visible and only one remains valid OOD.
```

This is not enough by itself. The benchmark must prove that:

```text
1. the true task is really an inverse problem,
2. final-only evidence is insufficient or weak,
3. full core dynamics contain recoverable source information,
4. nuisance dynamics form a stronger apparent arrow in train/IID,
5. the nuisance arrow fails under OOD reversal or randomization,
6. static leaks do not solve the task.
```
