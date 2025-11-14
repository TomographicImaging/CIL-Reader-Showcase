# CIL-Reader-Showcase
A library of jupyter notebooks to showcase reading open-source CT datasets of a variety of formats. These notebooks have been contributed by users of the [Core Imaging Library](https://github.com/TomographicImaging/CIL).

## Install an environment to run the showcase locally

The easiest way to install an environment to run the showcase is using our maintained environment file which contains the required packages. Running the command below will create a new environment which has specific and tested versions of all CIL dependencies and additional packages required to run the showcase: 

```sh
conda env create -f https://tomographicimaging.github.io/scripts/env/cil_demos.yml
```

## Run the showcase locally

- Activate your environment using: ``conda activate cil-demos``.

- Clone the ``CIL-Reader-Showcase`` repository and move into the ``CIL-Reader-Showcase`` folder.

- Run: ``jupyter-lab`` on the command line.


## Contribution Guidelines

To contribute to the repository, create a fork of the repository and a new branch.

Create a notebook with a descriptive name containing the name of the data type and the sample it's working with.

If the reader your notebook will be using is not already in CIL, please contribute it to [readers](readers)

If you would like to include any images in your demonstration please add them to [images](images). Any image files should begin with your notebook's name.

All contributed files should have the [Apache 2.0 License Header](https://github.com/TomographicImaging/CIL/blob/ab4d56e470eac32959e20fe36ba1bcc6a3eb5361/Wrappers/Python/cil/io/NEXUSDataReader.py#L4-L15) at the top, plus the name of the authors.

Please read and adhere to the [developer guide](https://tomographicimaging.github.io/CIL/nightly/developer_guide/) and local patterns and conventions.

Please also:
    Add a description of your contribution(s) to the top of your file(s)
    Add a link to the publicly available dataset used near the top of the file(s)
    Add the CIL version you ran with near the top of the file(s)
    Mention in your PR if your reader requires any additional dependencies

When your contributions are ready, open a pull request for the CIL developer team to review! When you open the pull request please tick the box saying "Allow edits from maintainers".

The CIL developer team will then review your contribution and may ask for small changes or push some small changes to your fork.

If you wish to contribute a notebook, have any questions or want to find out more then please join the [CIL community on Discord](https://discord.com/invite/9NTWu9MEGq).
