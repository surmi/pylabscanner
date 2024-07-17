# pylabscanner
Set of Python scripts used for automation of scanning jobs with CLI.

# How to run
Following repo uses [pipenv](https://pipenv.pypa.io/) and [setuptools](https://setuptools.pypa.io/). To launch the application in editable mode follow these steps:
- Make sure you have Python 12.
- Install pipenv with pip.
- Install necessary dependencies in editable mode with pipenv `pipenv install -e .` (run in the repo directory).
- Check available commands by running `labscanner`.

# Features
- Implementation of communication with Thorlabs LTS stages based on [thorlabs-apt-device](https://gitlab.com/ptapping/thorlabs-apt-device). Due to long time of execution some functions are provided in asynchronous modes.
- Implementation of communication with Luvitera THz Mini.
- Simple scheduler for scanning tasks. Divides each scan into a set of consecutive actions (move to specific position, obtain value from the detector, home etc.). All actions are stored in a list which can then be iterated over to run each of them. The scheduler also estimates the time of measurement.
- CLI based on the [Click](https://click.palletsprojects.com/) library with [tqdm](https://tqdm.github.io/) progress bar. Provides basic processing of the measured data and plotting functionality.

## TODO
- Implementation of the measurement data storage with corresponding metadata - most probably HDF5 with h5py.
- Better handling for single line scans with CLI.
- Live reading from the detector - useful for identification of the beam position etc.
- Plotting command - selection of the specific data for plotting, plotting of the data from multiple files for easy comparison.

# License
All original work is licensed under the GNU General Public License v3.0. See the [license](LICENSE.txt) for details.

A part of the `thorlabs-apt-device` library (`thorlabs-apt-protocol`) uses an MIT license. For deatils see corresponding [readme](https://gitlab.com/ptapping/thorlabs-apt-device/-/blob/main/README.rst).
