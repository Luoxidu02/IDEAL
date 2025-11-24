# IDEAL-MDA

**IDEAL-MDA**: A Multi-View Feature Fusion Framework with Interpretable Graph Convolution for Predicting Microbe-Drug Associations. This framework integrates multi-view features, explainability feedback, and continual learning to enhance prediction accuracy, interpretability, and model adaptability.

### Dataset

*   **MDAD**: Contains 1373 drugs and 173 microbes with 2470 known associations.
*   **aBiofilm**: Contains 1720 drugs and 140 microbes with 2884 known associations.
*   **DrugVirus**: Contains 175 drugs and 95 microbes with 933 known associations.

### Data description

This project utilizes a variety of data sources to build a comprehensive model. The key data files include:
*   **adj**: The microbe-drug interaction matrix (adjacency matrix).
*   **drug_similarity**: An integrated similarity matrix for drugs, combining chemical structure and interaction profiles.
*   **microbe_similarity**: An integrated similarity matrix for microbes, based on functional similarity.
*   **drug_features**: Multi-view feature matrices for drugs, including topological, semantic (BERT), and fingerprint attributes.
*   **microbe_features**: Multi-view feature matrices for microbes, including genomic, semantic (BERT), and metabolic pathway attributes.

### Run Step

Run `main.py --retrain` to train the model and obtain the predicted scores for microbe-drug associations.


### Requirements

*   `torch~=2.4.0+cu121`
*   `pandas~=2.3.1`
*   `numpy~=2.1.2`
*   `scikit-learn~=1.7.1`
*   `networkx~=3.2.1`
*   `matplotlib~=3.10.3`
*   `seaborn~=0.13.2`
*   `tqdm~=4.67.1`
