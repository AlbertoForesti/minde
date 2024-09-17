# Makefile

# Name of the conda environment
ENV_NAME = minde

# Create the conda environment and install dependencies
create_env:
	conda create --name $(ENV_NAME) pip --yes

# Install the project in edit mode
install:
	pip install -e .

# Remove the conda environment
clean:
	conda remove --name $(ENV_NAME) --all -y

.PHONY: create_env install clean