**Our contributions:**

**(a) Problem view:** a new distribution shift perspective(node's neighbor pattern gap between train and test) to promote node representation learning on heterophilic graphs. **Compared with previous works that aim to design a more effective HGNN backbone, we are the first to reconsider the node representation aggregation from data distribution, thus further integrating with existing SOTA HGNN's backbones to achieve better performance(line214-line246)**. Notably, the data distribution means that **we should compare the structure-related distribution between train and test datasets(Figure 7) on the same dataset rather than directly statistics on the full dataset like many other studies.** 

**(b) Technique view:** To address our pointed distribution shift, we make a detailed theoretical analysis to explain why previous graph-based invariant learning methods can't work well(Figure 1, line 271-281,line341, line 381). **Compared with previous works that generate extra augmented graphs by mask strategy to construct different environments(Figure 1, line 312-340), we utilize the node's inherent neighbor pattern information to infer environments without augmentation. Then a natural question arises as bold by lines 147,151, how can we ensure the effectiveness of our proposed graph-based invariant learning methods? Thus, we should provide strict theory evidence for selecting a proper matrix to estimate the node's neighbor pattern from a graph perspective(Section 4.1) and casual invariant learning perspective(Section 4.3)** rather than only from experimental results. A more detailed analysis can be found in the appendix(Figure 5, A.2).



**Rebuttal for Reviewer 8FKd** 

Thanks for your questions, we would like to reclarify the contribution first and then answer your questions.

**Q1 More explanation for the motivation**

Thanks for your questions. **First of all, your mentioned works about "many GNNs can handle such a "local homophily/heterophily difference" is indeed different from our method.** Their stated local homophily/heterophily difference is the statistic on the full graph dataset, and they aim to design a backbone that can achieve good performance on the homophilic and heterophilic graphs simultaneously. In contrast, our work focuses on the structure-related distribution shift, especially on heterophilic graphs. The homophily difference in our work exists between the train set and test set from the same dataset rather than different datasets like previous works. And we aim to propose a framework that can be integrated with previous SOTA HGNN to further improve model performance on heterophilic graphs.

**Moreover, your mentioned 3 percent is not the distribution gap.** Exactly, Figure 7 reflects how many proportions of nodes are in a certain homophily interval among all train and test nodes. We can easily find that the true distribution gap is more obvious considering the Interval spacing, such as comparing $(0,0.1)$ and $(0.3,1)$. 

**Furthermore, we can verify the necessity to address our pointed distribution shift from the application view.** Take anomaly detection as an example, due to various time factors and the annotation preferences of human experts, the heterophily and homophily can change across training and testing data, and this distribution gap will weaken the generalization of the trained model to detect the anomaly class and the normal classes with different levels of node homophily. To avoid the influence of homophily distribution on model predictions, we should train a model that can perform well on test nodes with high homophily and low homophily simultaneously. Thus, apart from full test accuracy, we should also focus on the performance gap between test groups with different levels of homophily, which corresponds to the results of Table 2 and Table 3. Moreover, as stated in lines 786-789, in real-world scenarios, we should consider the effect of data sampling on the model training. That's why we conduct experiments to further verify it's important to address this distribution shift.

**Q2 More clarification for the connection between similarity-based matrix and neighbor pattern**

**The elaboration from Line 432 to Line 452** Exactly, the clusters are formed by the node's feature(raw feature or aggregated feature) rather than the topology(edges). For node classification, the aggregated node embedding comes from its own raw feature and features aggregated from neighbors. It's the aggregated node embeddings that can decide the label, and the node belonging to the same label usually owns the same cluster centroid. Moreover, the neighbor pattern is used to estimate the label relationship between the node and its neighbors. Thus, from the view of the similarity cluster, we can treat the similarity-based matrix as the neighbor pattern indicator.

**Q3 More clarification for the experiments**

Exactly, as stated by lines 156-158, we truly include latest SOTA heterophilic GNNs' pure performance without any invariant learning solution as a baseline); it can show that current SOTA heterophilic GNNs indeed cannot handle the so-called heterophily shift, and the proposed solution is very general to benefit broad heterophilic GNNs as you said.

Moreover, the experiments on homophilic graph datasets can be found in Table 5, the performance gap between the low homophily test and the high homophily test is nearly, which can verify our statement. Or we can re-clarify that the distribution shift on homophilic graph datasets is not obvious compared with heterophilic graphs to help you understand our focused problem easily.


**Q4 Other replies to the concepts you mentioned**

(a) Line 291: The adjacency matrix of a node is a 0 1 matrix that describes its connected nodes among all nodes.

(b) Line 441: we can not understand your questions. We would be glad to address it if you could clarify it more clearly.

(c) Lines 559 and 560: it's our writing issues. In causal analysis, the separated random variable should be bold, and others are not. And the X and A should both contain two elements, it's our neglected writing issues.  We will make a more detailed statement and carefully polish the paper in the revised version.