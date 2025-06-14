# Zero-Shot Evaluation of Single-Cell Foundation Models

*Exploring Intermediate Layer Embeddings of scGPT on Spatial Transcriptomics Data*


## Introduction

This repository contains the solutions to the exercises assigned during the Hugging Face library hands-on session. While the initial exercises involved standard applications and usages of the library — such as dataset loading, model fine-tuning, and evaluation — this README focuses exclusively on the final, personal project, where a more exploratory and original analysis was conducted.

This project investigates the capabilities of **Foundation Models** in the field of single-cell genomics, with a particular focus on their performance in *zero-shot* scenarios — that is, evaluating model outputs without any task-specific fine-tuning. Foundation Models are renowned for their ability to learn universal data representations from large datasets and generalize effectively across diverse tasks.

The idea to explore *intermediate embeddings* — rather than limiting the evaluation to the final layer — was inspired by the paper ["Perception Encoder: The best visual embeddings are not at the output of the network"](https://arxiv.org/abs/2504.13181), which suggests that the most informative representations for certain tasks may reside in the inner layers of a model rather than at its output.

Additionally, this work builds upon the study ["Zero-shot evaluation reveals limitations of single-cell foundation models"](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-025-03574-x?utm_source=bmc_etoc&utm_medium=email&utm_campaign=CONR_13059_AWA1_GL_DTEC_054CI_TOC-250419#Sec3) by Kedzierska et al. (2025), a team of Microsoft researchers, which highlighted that models such as **scGPT** and Geneformer can sometimes underperform in zero-shot scenarios — in some cases being outperformed by simpler baselines. However, their evaluation focused exclusively on embeddings from the final model layer, leaving open the question of whether intermediate representations might offer improved predictive utility.


## Data and Model

Due to the unavailability of the original datasets used in the reference study, this project uses the publicly available **TCGA\_digital\_spatial\_transcriptomics** dataset from Hugging Face Datasets. This dataset contains spatial transcriptomics data: gene expression values (RNA levels) for thousands of genes measured at spatially resolved spots in tissue samples, along with their XY coordinates.

The scGPT model, hosted on Hugging Face Hub ([tdc/scGPT](https://huggingface.co/tdc/scGPT)), serves as the foundation model. It was chosen because of its relevance in the field and its role in the reference article.

## Methodology

The core of the analysis consists of extracting embeddings from *each* of the twelve Transformer encoder layers within scGPT, without modifying the model architecture or weights. The procedure is as follows:

* The input gene expression data (gene IDs + values) is encoded into a 512-dimensional vector by the model’s gene and value encoders.
* This vector sequentially passes through the twelve TransformerEncoderLayers.
* The output embedding from each layer is saved.
* The model’s pre-trained expression decoding head is applied to project these 512-dimensional embeddings into scalar values, representing predicted gene expression.
* Predictions are compared to true gene expression using metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE), restricted to genes actively expressed (excluding padding).
* A simple baseline prediction is computed using the average expression value per cell.

This setup allows a layer-wise, zero-shot evaluation of the model’s intrinsic representational power.

## Results and Insights

The evaluation confirms the original article’s conclusion that all transformer layers perform worse than the simple average-expression baseline when predicting gene expression in this zero-shot setting.

However, a critical new finding is that **intermediate layers can outperform the final layer**: for example, the second Transformer block’s embedding yields an MSE of 30.2 compared to 381.0 for the last layer, indicating richer predictive information early in the network. This underscores the importance of multi-layer analysis to understand model behavior and representation quality.

---

The following table reports the main evaluation metrics (MSE, MAE, Pearson correlation) for each Transformer layer output compared to a simple mean baseline. Full results, including standard deviations, are available in `results/scgpt_layerwise_evaluation.csv`.

| Layer                | MSE Mean   | MAE Mean   | 
| -------------------- | ---------- | ---------- | 
| Layer                | MSE Mean   | MAE Mean   | 
| -------------------- | ---------- | ---------- | 
| Mean Baseline        | **0.1507** | **0.2592** |
| Transformer Layer 0  | 42.8374    | 6.4056     |
| Transformer Layer 1  | 30.2377    | 5.3947     |
| Transformer Layer 2  | 30.8693    | 5.4469     |
| Transformer Layer 3  | 36.0490    | 5.9255     |
| Transformer Layer 4  | 64.7740    | 8.0070     |
| Transformer Layer 5  | 108.6514   | 10.3867    |
| Transformer Layer 6  | 102.0115   | 10.0561    |
| Transformer Layer 7  | 72.9351    | 8.4405     |
| Transformer Layer 8  | 85.7650    | 9.0805     |
| Transformer Layer 9  | 195.4080   | 13.8542    |
| Transformer Layer 10 | 450.0759   | 21.1728    |
| Transformer Layer 11 | 381.0563   | 19.5018    |

**Note**:

* The lowest MSE and MAE are achieved by the simple **mean baseline**.
* Among model layers, **Transformer Layer 1 and Layer 2** provide the best performance, with MSE ≈ 30.2–30.8, far better than the final layer (Layer 11).
* Deeper layers show degradation in performance, confirming findings from the reference paper.



## Future Developments

Several directions could be explored to improve and extend this analysis:

* **Dataset Alignment**: For an optimal comparison with the original study ("Zero-shot evaluation reveals limitations of single-cell foundation models"), it would be essential to re-run this evaluation using the same datasets employed by the authors. Unfortunately, these datasets are currently unavailable to the public.

* **Model Comparison**: Extending the evaluation to other single-cell foundation models, such as Geneformer, could help generalize the findings and better assess whether the observed limitations are specific to scGPT or more broadly applicable.

* **Embedding Combination Strategies**: Investigating more sophisticated methods for combining embeddings from different layers (e.g., weighted sums, learned combinations) might improve performance. In this work, a simple mean of embeddings from multiple layers was tested but led to degraded performance compared to the best single-layer results.



