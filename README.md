# Clustering of Vendor Profiles from German Chats based on Natural Language Processing

## Development setup

The development setup consists of two parts.
The first part is needed for running the project.
The second part is needed for committing changes to the project.

### 1. Create the environment

This project uses `conda` and `poetry` for dependency management and building.
The setup is done as follows:

1. Create a new conda environment

    ```bash
    conda env create -f env.yml
    ```

2. Activate the environment

    ```bash
    conda activate data-2022
    ```

3. Tell poetry to not create a virtualenv

    ```bash
    poetry config --local virtualenvs.create false
    ```

4. Install all dependencies

    ```bash
    poetry install
    ```

5. Extract the data 

    ```bash
    tar -xvf data.tar.gz
    ```
    
6. Run the experiments
   
   ```bash
   dvc repro
   ```
