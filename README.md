# KHG-Aclair: Knowledge HyperGraph-based Attention with Contrastive Learning

Welcome to the official repository of KHG-Aclair, a research project focused on enhancing attention mechanisms using Knowledge HyperGraphs and Contrastive Learning.

## Overview

KHG-Aclair aims to improve attention mechanisms in machine learning models by leveraging Knowledge HyperGraphs (KHGs). This repository contains the source code and datasets used in the research paper titled "KHG-Aclair". Please note that the code is currently being organized and will soon be made available in a well-structured format.
<!-- ## Features

- **Will Be**: 
- **Update**: 
- **Soon**: 
-->
  
## Requirements
python==3.8.0  

numpy=1.21.0  

pandas==1.3.0  

scikit-learn==1.1.1  

scipy==1.7.0  

torch==1.8.1  

tqdm==4.61.2  


## Usage
+ Example of Item Knowledge Hypergraph Construction :

  python ./data/run.py --dataset=movielens --rel_min=2 --rel_max=2400

+ Example of running KHG-Aclair:

  python main.py --dataset=movielens --rel_min=2 --rel_max=2400


## Dataset
The dataset is available via the following link due to file size issues.
[Dataset Download Link](https://drive.google.com/drive/folders/1Egb3JhEQe0EDkCHW75C_feDSfHkQHOXF?usp=sharing)


## Citation

If you use KHG-Aclair in your research or find it helpful, please consider citing our paper.

@article{yourcitationdetails,
  title={KHG-Aclair: Knowledge HyperGraph-based Attention with Contrastive Learning},
  author={Hyejin Park, Taeyoon Lee, Kyungwon Kim},
  journal={Will be published in the Journal of Computing Science and Engineering(JCSE)},
  year={2024},
}

## Acknowledgments
This work was supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the government of Korea (MSIT) under grant number 2021-0-01352, titled "Development of technology for validating the autonomous driving services in perspective of laws and regulations.

## Source Code References
This project was developed with reference to the source codes from [KGCL](https://github.com/yuh-yang/KGCL-SIGIR22) and [HyperGAT](https://github.com/kaize0409/HyperGAT_TextClassification). We express our deep gratitude for their contributions and ideas.

## Contact

For questions or inquiries, please contact [Hyejin Park](mailto:h.ngc1316@gmail.com) or [Taeyoon Lee](mailto:tylee814@gmail.com).
