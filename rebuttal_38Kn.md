**Our contributions:**

**(a) Problem view:** a new distribution shift perspective(node's neighbor pattern gap between train and test) to promote node representation learning on heterophilic graphs. **Compared with previous works that aim to design a more effective HGNN backbone, we are the first to reconsider the node representation aggregation from data distribution, thus further integrating with existing SOTA HGNN's backbones to achieve better performance(line214-line246)**. Notably, the data distribution means that **we should compare the structure-related distribution between train and test datasets(Figure 7) on the same dataset rather than directly statistics on the full dataset like many other studies.** 

**(b) Technique view:** To address our pointed distribution shift, we make a detailed theoretical analysis to explain why previous graph-based invariant learning methods can't work well(Figure 1, line 271-281,line341, line 381). **Compared with previous works that generate extra augmented graphs by mask strategy to construct different environments(Figure 1, line 312-340), we utilize the node's inherent neighbor pattern information to infer environments without augmentation. Then a natural question arises as bold by lines 147,151, how can we ensure the effectiveness of our proposed graph-based invariant learning methods? Thus, we should provide strict theory evidence for selecting a proper matrix to estimate the node's neighbor pattern from a graph perspective(Section 4.1) and casual invariant learning perspective(Section 4.3)** rather than only from experimental results. A more detailed analysis can be found in the appendix(Figure 5, A.2).



**Rebuttal for Reviewer 38Kn**

Thanks for your questions, we would like to reclarify the contribution first and then answer your questions.

**Q1 The significance of the investigated problem** 

Thanks for your questions. As we know, heterophilic graphs are composed of nodes with different levels of homophily. Many previous HGNNS works aim to propose more effective Neighbor aggregation mechanisms to select similar neighbors for each node during neighbor aggregation, thus designing better backbones to achieve good performance on heterophily graph datasets. However, as shown in Table 5, their evaluation is only based on the full test nodes neglecting the difference of test nodes with high and low homohily respectively. 

**(a)From the application view**, take anomaly detection as an example, due to various time factors and the
annotation preferences of human experts, the heterophily and homophily can change across training and testing data, this distribution gap will weaken the generalization of the trained model to detect the anomaly class and the normal classes with different levels of node homophily. To avoid the influence of homophily distribution on model predictions, we should train a model that can perform well on test nodes with high homophily and low homophily simultaneously. Thus, apart from full test accuracy, we should also focus on the performance gap between test groups with different levels of homophily, which corresponds to the results of Table 2 and Table 3. Moreover, as stated in lines 786-789, in real-world scenarios, we should consider the effect of data sampling on the model training. That's why we conduct experiments to further verify it's important to address this distribution shift.

**(b)From the theory view**, the gap between train and test distribution will influence the evaluation of the model's true performance. A model that can perform well on full test nodes may achieve a huge performance gap between the high homophily test and low homophily test, which further verifies the necessity to address this issue. Moreover, we review the data split in Figure 7 and point out this neglected distribution shift.  Based on the causal invariant learning theory, an ideal model should learn the invariant feature that's independent of the environment-related feature so as to adapt to diverse and complex environments. The unstable environments often cause the gap between train and test data distribution, which is independent of the model backbone and related to data distribution, further influencing the model performance. For node classification tasks, due to the potential unlabeled status of neighbor nodes, we can not know whether the train and test split is reasonable considering the node's neighbor pattern distribution. Thus, apart from directly using a fixed backbone to fit the training data, we should also explore the strategy to optimize the backbone to achieve good performance considering the randomness of sampling, which is also a valuable problem.

**Q2 Clarification for experiments** 

Exactly, the GIN and GCN are not backbones designed for heterophilic graphs. Previous HGNN works compared with GIN and GCN because they mainly focus on HGNN backbone designs, they should further distinguish their proposed backbone from traditional GNN backbone(e.g.GCN). But as stated by contribution, our proposed methods especially focus on the distribution shift on heterophilic graphs and our proposed framework can be integrated with existing SOTA HGNN's backbones to achieve better performance. Thus, the comparison of traditional GCN and GIN is not necessary in our experiments. 

**Q3 More detailed theoretical analysis** 

Thanks for your questions. Please see Appendix A.2 to further understand the two conditions you mentioned to re-evaluate our work.

**Q4 Code Link** 

We have provided detailed Implementation Details in A.4. The code can be accessed through the anonymous link https://anonymous.4open.science/r/HEI-AC0D