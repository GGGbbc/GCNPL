# GCNPL
### model
<p align="center">
  <img src="./figures/model.png" />
</p>

## 1. Environments

```
- python==3.8
- cuda==11.3
```

## 2. Dependencies

```
- numpy==1.18.0
- scikit-learn==0.22.1
- scipy==1.4.1
- tqdm==4.41.1
- transformers==4.0.0
- torch==1.10.0
- pandas==1.3.4
- scikit-learn==1.0.1
```

## 3. Dataset

Here we provide the processed GAD dataset

## 4. Preparation

- Getting word embeddings in text with Glove
- Biomedical texts can be analyzed using the Stanford CoreNLP tool to obtain syntactic dependency trees.

## 5. Training and Evaluate

```bash
- sh gad.sh
