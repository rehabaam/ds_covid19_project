# Datascientest project - Analysis of Covid-19 chest x-rays

## Project description

In order to diagnose patients with Covid-19, the analysis of chest X-rays is a possibility to be explored to more easily detect positive cases. If the classification through deep learning of such data proves effective in detecting positive cases, then this method can be used in hospitals and clinics when traditional testing cannot be done.

## Resources to refer to: 
### Data: 
The data set contains chest x-ray images for covid-19 positive cases but also x-ray images of normal and viral pneumonia. Link to dataset: https://www.kaggle.com/tawsifurrahman/covid19-radiography-database 
(Data size: 1.15 Gb)

### Bibliography:
https://arxiv.org/abs/2003.13865 
https://doi.org/10.1016/j.compbiomed.2021.105002

## Running the project
### GIT
Git commands to use in terminal/console
#### show git log
```bash
git log --oneline --graph --name-status
```
#### show changes by user (blame)
```bash
git blame filename
```
#### show change between commits
```bash
git diff commit1 commit2
```
#### reset change to last commit
```bash
git reset --soft HEAD~1
```
#### create new branch
```bash
git checkout -b new-branch-name
```
#### add stages files
```bash
git add file1
```
#### commit changes with a message
```bash
git commit -m "message"
```
### Python virtual environment
To setup venv for this project, use the script `venv.sh` in the root folder of this project as the following:
```bash
source venv.sh help
```
### Install project dependencies
To install project's dependencies, use Makefile in the root folder of this project as the following:
```bash
make install
```
### Jupyter notebook kernel to the .ds_covid19 env 
To set Jupyter's notebook kernel to .ds_covid19 env, execute this command after the env is activate as the following:
```bash
python3 -m ipykernel install  --user --name=ds_covid19 
```