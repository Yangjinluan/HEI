**Our contributions:**

**(a) Problem view:** a new distribution shift perspective(node's neighbor pattern gap between train and test) to promote node representation learning on heterophilic graphs. **Compared with previous works that aim to design a more effective HGNN backbone, we are the first to reconsider the node representation aggregation from data distribution, thus further integrating with existing SOTA HGNN's backbones to achieve better performance(line214-line246)**. Notably, the data distribution means that **we should compare the structure-related distribution between train and test datasets(Figure 7) on the same dataset rather than directly statistics on the full dataset like many other studies.** 

**(b) Technique view:** To address our pointed distribution shift, we make a detailed theoretical analysis to explain why previous graph-based invariant learning methods can't work well(Figure 1, line 271-281,line341, line 381). **Compared with previous works that generate extra augmented graphs by mask strategy to construct different environments(Figure 1, line 312-340), we utilize the node's inherent neighbor pattern information to infer environments without augmentation. Then a natural question arises as bold by lines 147,151, how can we ensure the effectiveness of our proposed graph-based invariant learning methods? Thus, we should provide strict theory evidence for selecting a proper matrix to estimate the node's neighbor pattern from a graph perspective(Section 4.1) and casual invariant learning perspective(Section 4.3)** rather than only from experimental results. A more detailed analysis can be found in the appendix(Figure 5, A.2).



**Rebuttal for Reviewer sSKZ**

Thanks for your questions, we would like to reclarify the contribution first and then answer your questions.

**Q1 More clarification for theoretical analysis**

Thanks for your questions. Exactly, we focus on heterophilic graph structure distribution shift and our method belongs to graph-based invariant learning methods like previous works(e.g. EERM[1]), which aim to address the distribution shift. Thus, the theoretical analysis should include the graph perspective and causal invariant learning perspective respectively. The casual invariant learning methods should explore the relationship between input random variables like Figure 5 in the appendix, and exactly we also provide a more detailed analysis in A.2. We will further strengthen this and make it more understandable in the revised paper. And we hope you can reclarify our contribution.

**Q2 More clarification for comparison experiments**

**(a)Datasets:** As stated in Table 1, we have conducted experiments on commonly used heterophilic small and large-scale datasets, which include one hundred thousand nodes and millions of edges.

**(b)Baselines:** The method INL[2] you mentioned is still a work that focuses on backbone design, as stated by our contribution, we focus on reconsidering the node representation aggregation from distribution shift, thus further proposing a framework that can be integrated with existing SOTA HGNN's backbones to achieve better performance. We also provide part of comparison experiments based on your mentioned work to verify the effectiveness of our method further. Moreover, the uploading time of your mentioned paper is later than the submission of KDD this year. We will add these discussions referring to your questions.

|  Method(Chameleon)  | Full Test| High Hom Test  | Low Hom Test  |
|  :------  | :------:  | :------:  | :------:  | 
| INL  | 71.84 ± 1.22 | 76.81 ± 1.78| 65.97 ± 3.75  | 
| INL + HEI(Ours) | 74.74 ± 1.18 |  78.33 ± 1.35 |69.05  ± 3.45 |  

|  Method(Squirrel)  | Full Test| High Hom Test  | Low Hom Test  |
|  :------  | :------:  | :------:  | :------:  | 
| INL  | 64.38 ± 0.62 | 74.29 ± 3.85| 54.02 ± 1.54  | 
| INL + HEI(Ours) | 68.14 ± 0.59 |  77.35 ± 2.81 |58.13 ± 1.58   |

|  Method(Actor)  | Full Test| High Hom Test  | Low Hom Test  |
|  :------  | :------:  | :------:  | :------:  | 
| INL  | 38.12 ± 0.36 | 41.67 ± 1.38 |35.11 ± 1.85  | 
| INL + HEI(Ours) | 39.62 ± 0.56 | 44.14 ± 1.48 |37.14 ± 1.85  |

**Q3 More clarification for computational efficiency**

Exactly, we have provided complexity analysis in A.1. To further clearly address your concern about computational efficiency on large-scale datasets, we provide efficiency experiments on large-scale datasets referring to [2]. The reported results are the time(seconds) to train the model until converge, we can conclude that the extra time cost can be acceptable compared with the base(backbone itself). 

|  Method  | Penn94| arxiv-year  | twitch-gamer |
|  :------  | :------:  | :------:  | :------:  | 
| Base   | 22.3 | 7.2| 40.5.  | 
| Renode | 23.5 | 7.6| 41.2  |  
| SR-GNN | 22.9 | 7.4| 41.0  |   
| EERM   | 24.0 | 8.0| 41.5.  |  
| BAGNN  | 24.8 | 8.6| 42.1 |   
| FLOOD  | 24.5 | 8.2| 41.8  | 
| HEI(ours)| 24.1 | 8.3| 41.9 | 

[1] Handling distribution shifts on graphs: An invariance perspective. ICLR2022.

[2] Discovering Invariant Neighborhood Patterns for Non-Homophilous Graphs. Arxiv 2024.

[3] Finding global homophily in graph neural networks when meeting heterophily. ICML2022