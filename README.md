# Localized Questions in Medical Visual Question Answering

This is the official repository of the paper "Localized Questions in Medical Visual Question Answering," (MICCAI 2023). We also have a [Project Website](https://sergiotasconmorales.github.io/conferences/miccai2023.html).

**Are you attending MICCAI 2023 in Vancouver? Let's connect! This is my [LinkedIn](https://www.linkedin.com/in/sergio-tascon/) or drop me an email at sergio.tasconmorales@unibe.ch.**


Our paper presents a method to answer questions about regions by using localized attention. In localized attention, a target region can be given to the model so that answers are focused on a user-defined region.  


🔥 Repo updates
- [x] Data download
- [x] Training 
- [x] Inference
- [x] Metrics plotting
- [x] Running the code in this repo to make sure everything works
- [ ] Make sure everything works after updating requirement versions (update done on 06.05.2025)


## Installing requirements
After cloning the repo, create a new environment with Python 3.9, activate it, and then install the required packages by running:

    pip install -r requirements.txt

---

## Data

You can access the datasets [here](https://zenodo.org/record/8192556). After downloading the data, decompress it in the repo folder, and make sure they follow the following structure (i.e. rename the data folder to `data` so that you don't have to change the path to the data in the config files)

**📂data**\
 ┣ **📂STS2017_v1** &nbsp; # RIS dataset\
 ┣ **📂INSEGCAT_v1** &nbsp; # INSEGCAT dataset\
 ┗ **📂DME_v1** &nbsp; # DME dataset\

Each of the above dataset folders should contain two folders: `images` and `qa`. A third folder named `processed` is created during dataset class instantiation when you run the training script. I included this processed data too, so that you can reproduce our results more easily (If you want to generate the processed data again, set `process_qa_again` to `True` in the config files). The DME dataset also contains a folder named `answer_weights` which contains the weights for the answers. The other two datasets do not require this, since they are balanced.


---

## Config files

Please refer to the following table for the names of the config files that lead to the results of the different baselines. Note that in our paper we took the average of 5 models trained with different seeds, so if you train only once, do not expect to obtain the same results reported in the paper.

| **Baseline**   | **Config name**          |
|----------------|--------------------------|
| No mask        | config_nomask.yaml       |
| Region in text | config_regionintext.yaml |
| Crop region    | config_cropregion.yaml   |
| Draw region    | config_drawregion.yaml   |
| Ours           | config_ours.yaml         |


Notice that the files mentioned in the previous table are available for each dataset in the `config` folder.

In the config files, do not forget to configure the paths according to your system.


---

## Training a model

To train a model, run

        python train.py --path_config config/<dataset>/config_XX.yaml

Where `<dataset>` can be one of (`dme`, `insegcat`,`sts2017`) and XX should be changed according to the table above (note that `sts2017` corresponds to the dataset called `RIS` in the paper). The model weights will be stored in the logs folder specified in the config file. Weights and optimizer parameters are saved both for the best and last version of the model. A file named `logbook.json` will contain the config parameters as well as the values of the learning curves. In the folder `answers` the answers are stored for each epoch.

---

## Inference

To run inference, run

        python inference.py --path_config config/<dataset>/config_XX.yaml

after inference, the metrics are printed for the validation and test sets. Also, the folder `answers` will contain the answers files for test and validation (`answers_epoch_val.pt` and `answers_epoch_test.pt` ).

---

## Plotting results

To plot the metrics, run

        python plot_metrics.py --path_config config/<dataset>/config_XX.yaml

This will produce plots of the learning curves, as well as metrics for test and validation in the logs folder specified in the yaml config file.

## Citation

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
