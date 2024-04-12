**Our contributions:**

**(a) Problem view:** a new distribution shift perspective(node's neighbor pattern gap between train and test) to promote node representation learning on heterophilic graphs. **Compared with previous works that aim to design a more effective HGNN backbone, we are the first to reconsider the node representation aggregation from data distribution, thus further integrating with existing SOTA HGNN's backbones to achieve better performance(line214-line246)**. Notably, the data distribution means that **we should compare the structure-related distribution between train and test datasets(Figure 7) on the same dataset rather than directly statistics on the full dataset like many other studies.** 

**(b) Technique view:** To address our pointed distribution shift, we make a detailed theoretical analysis to explain why previous graph-based invariant learning methods can't work well(Figure 1, line 271-281,line341, line 381). **Compared with previous works that generate extra augmented graphs by mask strategy to construct different environments(Figure 1, line 312-340), we utilize the node's inherent neighbor pattern information to infer environments without augmentation. Then a natural question arises as bold by lines 147,151, how can we ensure the effectiveness of our proposed graph-based invariant learning methods? Thus, we should provide strict theory evidence for selecting a proper matrix to estimate the node's neighbor pattern from a graph perspective(Section 4.1) and casual invariant learning perspective(Section 4.3)** rather than only from experimental results. A more detailed analysis can be found in the appendix(Figure 5, A.2).



**Rebuttal for Reviewer g2D7**

Thanks for your questions, we would like to reclarify the contribution first and then answer your questions.

**Q1 More clarification for contribution.** 

**Apart from the contribution stated above, as pointed out by the introduction(161-167) and Related work(241-246), the goal we utilize the similarity to evaluate the neighbor pattern is indeed different from previous works.** Previous works aim to explore more effective neighbor aggregation mechanisms to select similar neighbors for each node(**Previous works belong to backbone design works**). However, we utilize the neighbor pattern to infer the node's environments for invariant representation learning.**Our work is a framework considering the distribution gap between train and test, that can be integrated with previously designed backbones without difficulty. In other words, the framework is our contribution rather than the specific somewhat matrix to design an effective backbone.** As long as we can find indicators that meet the conditions stated in the method, our framework can be integrated with previously designed backbones to further improve model performance from a data distribution perspective. **The experiments in Table 4 can also verify that adopting any similarity-based indicator that meets the conditions we clarify in Methods can achieve better performance than the base(previous SOTA backbone itself).**

**Q2 More clarification for computational efficiency**

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
**Q3 More detailed clarification for experiments**

Thanks for your questions to help us polish our work. 

We first clarify our experimental targets to help you understand our work easily. Our proposed framework can be integrated with the previous SOTA HGNN backbone and the similarity is to estimate the neighbor pattern Z. Then the Z is used for the input for the environment classifier to assign the nodes into different environments, which is just our key contribution compared with previous works. The experiments in Table 4 can also verify that adopting any similarity indicators that meet the conditions we clarify in Methods can achieve better performance than the base(previous SOTA backbone itself) **This means the proposed framework is the key rather than the specific somewhat similarity.**

Then we provide more detailed experimental details reported in the paper to address your questions.

**(a)The used similarity metrics our experiments.** Exactly, all reported results in Table 2 and Table 3 adopt the SimRank as the neighbor pattern indicator. The experiments in Table 4 further answer the question, RQ3, exploring the effect of different similarity matrices. We just use the experiments on small-scall datasets to clarify it more clearly and in lines 859-869, we also provide some theoretical reasons for why adopting the SimRank achieves the best performance. The results adopting different similarity-based matrices on large-scale datasets can be found below， where we use the LINKX as thebackbone.

|  Method(Penn94)  | Full Test| High Hom Test  | Low Hom Test  |
|  :------  | :------:  | :------:  | :------:  | 
| HEI(Local_Sim)  | 85.12 ± 0.21| 88.28 ± 0.33 |82.15 ± 0.59 | 
| HEI(Agg_Sim) |85.21 ± 0.17| 88.29 ± 0.38 |82.22 ± 0.54 |  
| HEI(SimRank) | 85.52 ± 0.31| 88.44 ± 0.38 |82.62 ± 0.54 |  

|  Method(arxiv-year)  | Full Test| High Hom Test  | Low Hom Test  |
|  :------  | :------:  | :------:  | :------:  | 
| HEI(Local_Sim)  | 54.41 ± 0.21 | 64.23 ± 0.47 | 48.29 ± 0.22 | 
| HEI(Agg_Sim) | 54.45 ± 0.23 | 64.33 ± 0.49 | 48.33 ± 0.32 |  
| HEI(SimRank) | 54.65 ± 0.23 | 64.53 ± 0.63 | 48.63 ± 0.32|

|  Method(twitch-gamer)  | Full Test| High Hom Test  | Low Hom Test  |
|  :------  | :------:  | :------:  | :------:  | 
| HEI(Local_Sim)  | 66.18 ± 0.12 | 83.75 ± 0.34 | 48.12 ± 0.47 | 
| HEI(Agg_Sim) |  66.21 ± 0.15 | 83..85 ± 0.39 | 48.45 ± 0.57 |  
| HEI(SimRank) | 66.29 ± 0.15 | 84.03 ± 0.38 | 49.02 ± 0.57 |

**(b)Why does the proposed method show better performance on small-scale datasets than on large-scale datasets?** Exactly, we think it's not a proper and fair comparison between the results of small-scale datasets and large-scale datasets, because the scale of test nodes between small datasets and large datasets is indeed different. Compared with the data scale, to further clarify our problem and contribution, the homophily-related data distribution is a more important factor that influences the model performance, that's why we conduct experiments in Figure 3 and Figure 6. But in all settings, our framework can consistently achieve better performance on all small-scale and large-scale datasets compared with previous works.

[1] Finding global homophily in graph neural networks when meeting heterophily. ICML2022