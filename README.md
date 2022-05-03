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

### 2. Using pre-commit

1. Install pre-commit hooks to your local working copy

    ```bash
    poetry install
    poetry run pre-commit install --overwrite
    poetry run pre-commit install --hook-type commit-msg --overwrite
    poetry run pre-commit install --hook-type pre-push --overwrite
    ```

2. Sometimes an error is thrown on first execution. If so run:

    ```bash
    poetry run pre-commit autoupdate
    ```

3. Work as usual. On commit and push the hooks will automatically be executed

4. (optional) Of course, you can run hooks manually. This can be useful if you want to apply
a hook to all files inside your workspace.
To do this just execute

    ```bash
    poetry run pre-commit run <hook_name> --all-files --hook-stage <stage_when_hook_stage_is_not_pre_commit>
    ```
