# pylabscanner
Set of Python scripts used for automation of scanning jobs with CLI.

## Features
- Implementation of communication with Thorlabs LTS stages based on [thorlabs-apt-device](https://gitlab.com/ptapping/thorlabs-apt-device). Due to long time of execution some functions are provided in asynchronous modes.
- Implementation of communication with Luvitera THz Mini.
- Simple scheduler for scanning tasks. Divides each scan into a set of consecutive actions (move to specific position, obtain value from the detector, home etc.). All actions are stored in a list which cant then be iterated over to run each of them. The scheduler also estimates the time of measurement.
- CLI based on the [Click](https://click.palletsprojects.com/) library with [tqdm](https://tqdm.github.io/) progress bar. Provides basic processing of the measured data and plotting functionality.

## License
All original work is licensed under the GNU General Public License v3.0. See the LICENSE.txt for details.
