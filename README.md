## Group-based Federated Learning with Cost-efficient Sampling Mechanism in Mobile Edge Computing Networks

The paper has been accepted by IEEE Trans. Mobile Comput. 

The part of the work was published in the IEEE International Conference  on Communications (ICC), Denver, CO, USA, 9–13 June 2024 (<a href="https://ieeexplore.ieee.org/document/10623087" target="_blank">Energy-Efficient Client Sampling for Federated Learning in Heterogeneous Mobile Edge Computing Networks</a>).

**Title:** Group-based Federated Learning with Cost-efficient Sampling Mechanism in Mobile Edge Computing Networks

**Author:**  Jian Tang, Xiuhua Li, Guozeng Xu, Penghua Li, Xiaofei Wang, Victor C. M. Leung



### 1. Background and problem

Client sampling mechanism is essential for federated learning (FL) in mobile edge computing (MEC) networks, given the system and data heterogeneity of mobile clients (MCs). 
However, most of the current research on client sampling mechanisms is for individuals.
There is literature that experimentally proves that group-based sampling is superior to individual-based sampling to some extent.
Therefore, we study group-based sampling; however, group-based sampling requires labels from MCs to form.
Moreover, group-based sampling rarely considers the costs (i.e., latency and energy consumption) incurred by the MCs.
To address these challenges, we investigate and formulate the issue of group-based FL with a sampling mechanism for reducing costs in MEC networks. 
We have formulated the problem. (The process is in the paper.)

![image-20250205111311407](./assets/image-20250205111311407.png)

Figure 1. Group-based Federated Learning Model in a MEC Network. 

### 2. Proposed GFLCSM Framework

GFLCSM contains three components to solve the proposed problem.
Specifically, GFLCSM has three core modules:

1) inferring the data distribution of MCs,
2) forming group-based FL,
3) designing a cost-efficient sampling mechanism.

### 3. Experiments

You can run through the experiment with the following code

```
python main.py --server proposed
```
### 4. Citation

Finally, I would like to say this.

If this code was helpful for you, could you please cite this paper and give a star to this project? I really appreciate that !!!

```
J. Tang, X. Li, G. Xu, P. Li, X. Wang and V. C. M. Leung, "Group-based Federated Learning with Cost-efficient Sampling Mechanism in Mobile Edge Computing Networks," in IEEE Transactions on Mobile Computing, doi: 10.1109/TMC.2025.3608024.
```

or (BibTeX)

```

@ARTICLE{GFLCSM,
    author={Tang, Jian and Li, Xiuhua and Xu, Guozeng and Li, Penghua and Wang, Xiaofei and Leung, Victor C. M.},
    journal={{IEEE} Trans. Mob. Comput.},
    title={Group-based Federated Learning with Cost-efficient Sampling Mechanism in Mobile Edge Computing Networks}, 
    year={2025},
    pages={1-18},
    note={10.1109/TMC.2025.3608024}}
```

