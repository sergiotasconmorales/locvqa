# Localized Questions in Medical Visual Question Answering

This is the official repository of the paper "Localized Questions in Medical Visual Question Answering," (MICCAI 2023). We also have a [Project Website](https://sergiotasconmorales.github.io/conferences/miccai2023.html).

**Are you attending MICCAI 2023 in Vancouver? Let's connect! This is my [LinkedIn](https://www.linkedin.com/in/sergio-tascon/) or drop me an email at sergio.tasconmorales@unibe.ch.**


Our paper presents a method to answer questions about regions by using localized attention. In localized attention, a target region can be given to the model so that answers are focused on a user-defined region.  


🔥 Repo updates
- [ ] Data download and VQA-Introspect data preparation
- [ ] Training 
- [ ] Inference

## Installing requirements
After cloning the repo, create a new environment with Python 3.9, activate it, and then install the required packages by running:

    pip install -r requirements.txt

---

## Data
We used the VQA-Introspect and DME-VQA datasets to test our method. You can download the final versions of both datasets from [here](https://zenodo.org/record/7777878) and [here](https://zenodo.org/record/7777849), respectively. Notice that the image features of the COCO dataset (used by VQA-Introspect) must be downloaded separately for [train](https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/train2014_obj36.zip) and [val](https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/val2014_obj36.zip). For simplicity, you can organize your data as follows, after unzipping:

**📂data**\
 ┣ **📂lxmert**\
 ┃&nbsp; ┗ **📂data**\
 ┃ &nbsp; &nbsp; &nbsp; ┣ **📂introspect** &nbsp;&nbsp;&nbsp;&nbsp;# introspect json files\
 ┃ &nbsp; &nbsp; &nbsp; ┗ **📂dme** &nbsp;&nbsp;&nbsp;&nbsp;# dme json files\
 ┗ **📂mscoco_imgfeat** &nbsp;&nbsp;&nbsp;&nbsp;# introspect visual features

Optionally, you can follow the following steps to prepare the VQA-Introspect dataset yourself. 

⚠️ **IMPORTANT: If you downloaded the data from the previous links, ignore the next section (Data preparation).**


---

## Training a model

To train a model, run

        python locvqa/train.py --path_config config/config_XX.yaml


This will produce. 

---

## Inference

To run inference ...

        python locvqa/src/tasks/vqa_consistency.py --case XX

after inference... 

---

## Plotting results

To plot results,·..

## Reference

This work was carried out at the [AIMI Lab](https://www.artorg.unibe.ch/research/aimi/index_eng.html) of the [ARTORG Center for Biomedical Engineering Research](https://www.artorg.unibe.ch) of the [University of Bern](https://www.unibe.ch/index_eng.html). Please cite this work as:

> @inproceedings{tascon2023localized,\
  title={Localized Questions in Medical Visual Question Answering},\
  author={Tascon-Morales, Sergio and M{\'a}rquez-Neila, Pablo and Sznitman,Raphael},\
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},\
  pages={--},\
  year={2023}\
  organization={Springer}
}

---

## Acknowledgements

This project was partially funded by the Swiss National Science Foundation through grant 191983.

We thank the authors of [LXMERT](https://github.com/airsplay/lxmert) for the PyTorch implementation of their method.