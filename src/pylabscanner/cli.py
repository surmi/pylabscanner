import configparser
import filecmp
import logging
import shutil
from math import floor
from pathlib import Path
from time import time
from typing import Tuple

import click
import pandas as pd
from click.core import ParameterSource
from serial import SerialException, SerialTimeoutException

from .devices import DeviceNotFoundError
from .devices.manager import LiveView
from .scheduler.scheduler import ScanScheduler
from .utils import (
    _parse_detector_frequency,
    parse_detector_settings,
    parse_filepath,
    parse_range,
    parse_scan_parameters,
    plotting,
    postprocessing,
    saving,
    setup_manager,
)


class Config(object):
    def __init__(self):
        # self.verbose = False
        self.debug = False
        self.logger = None
        self.stage_sn = {}


pass_config = click.make_pass_decorator(Config, ensure=True)


def option_mock_devices(function):
    function = click.option(
        "-md",
        "--mock-devices",
        is_flag=True,
        show_default=True,
        default=False,
        help="Use mockups for devices",
    )(function)
    return function


@click.group()
@click.option(
    "--debug",
    is_flag=True,
    help="Run commands with logging outputed to 'log.txt' file. The log file "
    "will be overwritten!",
)
@click.option(
    "--config",
    type=click.Path(
        writable=True,
        path_type=Path,
        resolve_path=True,
    ),
    default=None,
    help="Path to configuration file",
)
@pass_config
def cli(confobj: Config, debug, config: Path):
    """
    CLI for scanning scripts.

    The application uses serial numbers of Thorlabs stages to identify them and
    connect. The serial numbers are stored in configuration file named
    'config.ini' in the default location. If the configuration file is not
    present and '--config' option is not provided, the user is asked for the
    serial numbers. If the '--config' option is used with configuration file
    already exisitng in the default location, files are compared and if
    different, exisitng file is replaced by the one provided by the user.

    \b
    Currently supported devices:
    - THORLABS LTS300 and LTS300/C stages (possible use of 3 stages at the
        same time).
    - LUVITERA THZ MINI 4x1 line of wideband bolometers.
    """
    # logging
    if debug:
        click.echo("Debug mode is on")
        logging.basicConfig(
            filename="log.txt",
            filemode="a",
            format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
            level=logging.DEBUG,
        )
    else:
        logging.basicConfig(level=logging.ERROR)

    confobj.logger = logging.getLogger(__name__)

    config_path = Path(__file__).parent.parent / "config.ini"
    if not config_path.exists() and config is None:
        click.echo("No configuration file")
        click.echo("Setting up configuration file")
        x_serial = click.prompt("Enter serial number of x axis stage", type=str)
        y_serial = click.prompt("Enter serial number of y axis stage", type=str)
        z_serial = click.prompt("Enter serial number of z axis stage", type=str)

        config_content = configparser.ConfigParser()
        config_content["devices"] = {
            "xstageserial": x_serial,
            "ystageserial": y_serial,
            "zstageserial": z_serial,
        }
        with open(config_path, "w") as config_fh:
            config_content.write(config_fh)
        click.echo("New configuration file created successfully")
    elif config is not None:
        if not config.exists():
            raise click.BadOptionUsage(
                "Configuration file does not exist in provided path"
            )
        if not filecmp.cmp(config_path, config, shallow=False):
            click.echo("Copying config file")
            shutil.copyfile(config, config_path)
            click.echo("Configuration file copied successfully")
        else:
            click.echo(
                "Provided configuration file has the same content as currently used"
            )
            click.echo("Skipping file coping")

    config_file = configparser.ConfigParser()
    config_file.read(config_path)
    confobj.stage_sn = {
        "x": config_file["devices"]["xstageserial"],
        "y": config_file["devices"]["ystageserial"],
        "z": config_file["devices"]["zstageserial"],
    }


@cli.command()
@click.argument("stageslist", default="ALL", required=False)
@option_mock_devices
@pass_config
def home(config: Config, stageslist, mock_devices: bool):
    """
    Homes Thorlabs stages.

    Available options: 'X', 'Y', 'Z', 'ALL'. Defaults to 'ALL'.
    """
    click.echo("Initializing devices...")
    try:
        manager = setup_manager(stage_sn=config.stage_sn, is_mockup=mock_devices)
    except SerialException as e:
        config.logger.error(
            "Serial connection error on stage initialization before homing "
            "operation."
        )
        config.logger.error(e)
        click.echo(
            "Serial connection error. Run the app with --debug command to see "
            "details in the log file."
        )
        raise click.Abort
    except RuntimeError as e:
        config.logger.error(
            "Runetime error on stage initialization before homing operation."
        )
        config.logger.error(e)
        click.echo(
            "Runetime error on stage initialization. Verify that stages are "
            "connected and powered on.\nRun the app with --debug command to "
            "see details in the log file."
        )
        raise click.Abort
    click.echo("\tDevices initialized")

    click.echo("Homing...")
    start = time()
    manager.home(stage_label=stageslist)
    # asyncio.run(aso_home_devs(stages), debug=config.debug)
    te_home = time()
    click.echo(f"\tStages homed in: {te_home-start:.2f}s")


@cli.command()
@click.option("-x", type=float, default=None, required=False)
@click.option("-y", type=float, default=None, required=False)
@click.option("-z", type=float, default=None, required=False)
@option_mock_devices
@pass_config
def moveTo(config: Config, x: float, y: float, z: float, mock_devices: bool):
    """
    Moves stages to given position.

    Provide distance for at least one axis ('-x', '-y' or '-z').
    """
    stage_destination = {}
    if x is not None:
        stage_destination["x"] = x
    if y is not None:
        stage_destination["y"] = y
    if z is not None:
        stage_destination["z"] = z

    if len(stage_destination) == 0:
        raise click.UsageError("Provide distance for at least one axis.")

    click.echo("Initializing devices...")
    try:
        manager = setup_manager(stage_sn=config.stage_sn, is_mockup=mock_devices)
    except SerialException as e:
        config.logger.error(
            "Serial connection error on stage initialization before homing "
            "operation."
        )
        config.logger.error(e)
        click.echo(
            "Serial connection error. Run the app with --debug command to see "
            "details in the log file."
        )
        raise click.Abort
    except RuntimeError as e:
        config.logger.error(
            "Runetime error on stage initialization before homing operation."
        )
        config.logger.error(e)
        click.echo(
            "Runetime error on stage initialization. Verify that stages are "
            "connected and powered on.\nRun the app with --debug command to "
            "see details in the log file."
        )
        raise click.Abort
    click.echo("\tDevices initialized")

    click.echo("Moving to designated position(s)...")
    start = time()
    try:
        manager.move_stage(stage_destination=stage_destination)
    except Exception as er:
        config.logger.error("Error while running asynchronous movement to position")
        config.logger.error(er)
        click.echo(
            "Error while running asynchronous movement to position. Run in "
            "debug mode to see more details."
        )
        raise click.Abort
    te_move = time()
    click.echo(f"\tStages moved in: {te_move-start:.2f}s")


@cli.command(context_settings={"show_default": True})
@click.option("-x", type=str, help="Scanning range for X axis.")
@click.option("-y", type=str, help="Scanning range for Y axis")
@click.option("-z", type=str, help="Scanning range for Z axis")
@click.option(
    "-o",
    "outpath",
    type=click.Path(
        writable=True,
        path_type=Path,
        resolve_path=True,
    ),
    default="./out/out.csv",
    help="File name or path to the file in which results will be written.",
)
@click.option(
    "-m",
    "mode",
    type=click.Choice(["flyby", "ptbypt"]),
    default="ptbypt",
    help="Scanning mode",
)
@click.option(
    "-s",
    "linestart",
    type=click.Choice(["snake", "cr"]),
    default="snake",
    help="Where should each line start",
)
@click.option(
    "-dn",
    "det_sens",
    type=click.Choice(["1", "2", "3", "4"]),
    help="Detector sensor",
    default="1",
)
@click.option(
    "-ds",
    "det_samp",
    type=click.Choice(["100", "200", "500", "1000", "2000", "5000"]),
    help="Detector number of samples",
    default="100",
)
@click.option(
    "-df",
    "det_freq",
    type=click.Choice(["1", "2", "5", "10", "20", "40"]),
    help="Detector sampling frequency (in kHz)",
    default="1",
)
@click.option(
    "-nc",
    "noconfirmation",
    is_flag=True,
    help="Run the scan without further confirmation",
)
@click.option(
    "-ts",
    "timestamp",
    is_flag=True,
    help="Wheather to append timestamp to the filename",
    default=False,
)
@click.option(
    "-plt", "plot", is_flag=True, default=False, help="Whether to plot the output data"
)
@click.option(
    "-plts",
    "plot_save",
    is_flag=True,
    default=False,
    help="Whether to save the plot of the output data",
)
@click.option(
    "-mf",
    "modulfreq",
    type=float,
    help="Signal modulation frequency used to calculate FFT results. Required for plotting",
)
@option_mock_devices
@pass_config
def scan(
    config: Config,
    x,
    y,
    z,
    outpath: Path,
    mode,
    noconfirmation,
    linestart,
    det_sens,
    det_samp,
    det_freq,
    timestamp,
    plot,
    plot_save,
    modulfreq,
    mock_devices,
):
    """
    Performs scanning operation.

    CAUTION: single line scans without providing ranges for other axis not
    yet implemented!

    Detector has three settings: sensor selection '-dn', number of samples per
    measurement '-ds', and sampling frequency '-df'. Number of samples and
    sampling frequency influences time of single measurement. Note that this
    influences how much time the stages will wait at given position (for
    'ptbypt' mode) or how large distance will be swept over during single
    measurement (for 'flyby' mode).

    Two scanning modes are available: 'flyby' and 'ptbypt'.
    'flyby' performs measurements during sweeping motion along one of the axis.
    'ptbypt' moves all stages to desired position, performs measurement, and
    moves to the next planned measurement point.

    This command provides also two options ('-s') on which side new line will
    begin. 'snake' will force beginning of next lines to alternate (creating
    snake-like pattern). When 'cr' is selected after a line is finished stages
    will come back to the starting position (like a carriage returning in a
    typewriter to the beginning).

    For scanning ranges (options '-x', '-y' and '-z') provide either 1 or 3
    numbers. Single number is assumed to be a single position to which given
    stage will move. If you with to provide the whole range, you need to pass 3
    numbers separated with ':' (colon). The order in the range:
    beginning:end:number_of-points_to_scan. All values passed as ranges are
    assumed to be in mm.

    Destination for data can be provided via '-o' option. It accepts a file
    name or whole valid path (may be relative) with file name (file name
    here is necessary). Output file is characterized by extension provided
    to the file. The CLI currently handles only CSV ('.csv' or '.txt'
    extensions) and HDF5 ('h5', 'he5', 'hdf5', or 'hdf') files. In case if
    provided file extension is unknown the CLI will default to CSV.
    Providing '-ts' flag appends timestamp to the file name. When provided
    path matches with existing file then timestamp will be added by default.

    The result of measurement is a series of samples per single scanning point
    (number of samples correspond to '-ds' value). Data can be additionally
    postprocessed - calculation of FFT. FFT is calculated only if signal
    modulation frequency is provided with 'modulfreq' option.

    Since after postprocessing each point has single numerical value,
    the program can provide a simple plot (flag '-plt'). With '-plt' flag
    program will generate and display the plot. If '-plts' is provided
    the plot will not be displayed but saved to file instead.

    \b
    Examples of valid ranges:
        '-x 100' - performs measurement with stage X at 100 mm.
        '-y 10:100:3' - performs measurement with stage Y at positions 10, 55,
        and 100.

    \b
    Example of minimal command:
        'labscanner scan -x 10:100:3 -y 100 -z 100' - 3 point scan for y=100,
        z=100, and x=[10,55,100].
    """

    # TODO: add logging to a file for errors (always) and for info (when selected)
    # TODO: identify modulation (chopper) frequency limit
    # TODO: move scan parsing to separate function (return settings dictionary?)
    # TODO: differentiate plot and plot_save behaviour

    # parse input
    ranges = {}
    try:
        measrngx = parse_range(x)
        if measrngx is not None:
            ranges["x"] = measrngx
        measrngy = parse_range(y)
        if measrngy is not None:
            ranges["y"] = measrngy
        measrngz = parse_range(z)
        if measrngz is not None:
            ranges["z"] = measrngz
    except ValueError as e:
        raise click.UsageError(e)
    if measrngx is None or measrngy is None or measrngz is None:
        # TODO: remove when ready
        raise click.UsageError(
            "Single line scans with only one range provided not available yet."
        )
    if measrngx is None and measrngy is None and measrngz is None:
        # return early if not enough information is provided
        raise click.UsageError("Provide scanning range for at least one axis")
    mode, linestart = parse_scan_parameters(mode=mode, linestart=linestart)
    det_sens, det_samp, det_freq = parse_detector_settings(
        detsens=det_sens, detsamp=det_samp, detfreq=det_freq
    )
    try:
        outpath, extension = parse_filepath(filepath=outpath, timestamp=timestamp)
    except ValueError as e:
        raise click.UsageError("Path needs to point to a file")
    if plot and modulfreq is None:
        raise click.UsageError("Can't plot FFT data without modulation frequency")

    # initialize devices
    click.echo("Initializing devices...")
    try:
        manager = setup_manager(
            stage_sn=config.stage_sn,
            is_mockup=mock_devices,
            sensor=det_sens,
            freq=det_freq,
            samples=det_samp,
        )
    except SerialException as e:
        config.logger.error(
            "Serial connection error on stage initialization before homing "
            "operation."
        )
        config.logger.error(e)
        click.echo(
            "Serial connection error. Run the app with --debug command to see "
            "details in the log file."
        )
        raise click.Abort
    except RuntimeError as e:
        config.logger.error(
            "Runetime error on stage initialization before homing operation."
        )
        config.logger.error(e)
        click.echo(
            "Runetime error on stage initialization. Verify that stages are "
            "connected and powered on.\nRun the app with --debug command to "
            "see details in the log file."
        )
        raise click.Abort
    except DeviceNotFoundError as e:
        config.logger.error("Bolometer line not detected. Please check " "connection!")
        config.logger.error(e)
        click.echo("Bolometer line not detected. Please check connection!")
        raise click.Abort
    except SerialTimeoutException as e:
        config.logger.error("Timeout on bolometer line connection")
        config.logger.error(e)
        click.echo(f"Timeout on bolometer line connection: {e}")
        raise click.Abort
    metadata = manager.fill_metadata()
    click.echo("\tDevices initialized")

    # Build the scanning routine
    scheduler = ScanScheduler(
        manager=manager,
        ranges=ranges,
        line_type=mode,
        line_start=linestart,
    )
    scheduler.make_schedule()
    scheduler.fill_metadata(metadata_output=metadata)
    metadata["signal modulation frequency [Hz]"] = modulfreq

    # print the setup parameters and ask for confirmation
    click.echo("---")
    click.echo("Ranges to scan:")
    click.echo(
        f"\tx: {x if x is not None else 0} "
        f"[{len(measrngx) if measrngx is not None else 0} point(s) per line]"
    )
    click.echo(
        f"\ty: {y if y is not None else 0} "
        f"[{len(measrngy) if measrngy is not None else 0} point(s) per line]"
    )
    click.echo(
        f"\tz: {z if z is not None else 0} "
        f"[{len(measrngz) if measrngz is not None else 0} point(s) per line]"
    )
    click.echo(
        "Total number of points in the scan: "
        f"{len(measrngx)*len(measrngy)*len(measrngz)}"
    )
    click.echo(
        f"Estimated time of measurement: {floor(scheduler.ta/60)}m {scheduler.ta % 60:.0f}s"
    )
    click.echo("---\n")

    if noconfirmation or click.confirm("Do you want to continue?"):
        # perform the operation
        click.echo("Measurement started")
        try:
            scheduler.run()
        except SerialTimeoutException as e:
            config.logger.error("Timeout on serial port")
            config.logger.error(e)
            click.echo(f"Timeout on serial port: {e}")
            # home the stages
            manager.home()

            raise click.Abort
        data = scheduler.data
        click.echo(
            f"Actual time of measurement: {floor(scheduler.ta_act/60)}m "
            f"{scheduler.ta_act % 60:.0f}s"
        )
        click.echo("\tMeasurement finished")

        # save raw data to file

        if modulfreq is not None:
            # do the postprocessing
            click.echo(f"Postprocessing - calculating FFT for provided frequency")
            try:
                postprocessing(
                    data=data,
                    modulation_frequency=modulfreq,
                    det_freq=det_freq.freq * 1000,
                )
            except BaseException as exception_any:
                saving(
                    data=data,
                    path=outpath,
                    label="failed_postproc",
                    extension=extension,
                )
                click.echo("Exception while performing postprocessing")
                config.logger.error(exception_any)
                click.echo(exception_any)
                raise exception_any
                # raise click.Abort
            click.echo("\tPostprocessing finished")

        saving(data=data, metadata=metadata, path=outpath, extension=extension)

        if plot or plot_save:
            plotting(data, path=outpath, save=plot_save, show=plot)
    else:
        click.echo("Measurement aborted")


@cli.command()
@click.argument(
    "files", nargs=-1, type=click.Path(readable=True, path_type=Path, resolve_path=True)
)
@click.option(
    "-post",
    "postproc",
    default=None,
    type=click.Choice(["mean", "fft"]),
    help="Postprocessing of obtained measurements",
)
@click.option(
    "-save",
    "plot_save",
    is_flag=True,
    default=False,
    help="Whether to save the plot of the output data",
)
@click.option(
    "-df",
    "det_freq",
    type=click.Choice(["1", "2", "5", "10", "20", "40"]),
    help="Detector sampling frequency (in kHz)",
    default="1",
)
@click.option(
    "-f", "chop_freq", type=float, help="Signal modulation frequency in Hz (if used)"
)
@pass_config
def plot(
    config: Config,
    files: Tuple[Path, ...],
    postproc: None | str,
    plot_save: bool,
    det_freq: click.Choice,
    chop_freq: float | None,
):
    """
    Plots data from selected FILES.

    If single file is provided, plots all processed data from given file unless
    mode is selected with '-post' option.

    For multiple files '-post' option becomes necessary. In this case multiple
    files are displayed one next to another.

    If '-save' option is used and file with corresponding name already exists,
    it will be replaced. When '-save' option is used plot is not displayed.

    The program will attempt to obtain metadata about the measurement from the
    file. If user will provide necessary parameters as one of the options
    those will be used. If necessary parameter will not be present as an option
    or in the metadata user will be prompted for additional information.

    This command is not fully implemented yet.
    """
    current_context = click.get_current_context()
    for f in files:
        if not f.exists():
            raise click.BadArgumentUsage(f"File in {f} path does not exist")

    if len(files) == 1:
        # TODO: metadata detection
        metadata = {}

        data = pd.read_csv(files[0], index_col=0)
        # TODO: correct extension checking
        outpath, extension = parse_filepath(
            filepath=files[0], timestamp=None, extension=".png"
        )

        if postproc is None:
            # plot all processed data
            try:
                print(config.debug)
                plotting(data=data, path=files[0], save=plot_save)
            except ValueError as e:
                config.logger.error(
                    "Requested plotting but no postprocessed data detected."
                )
                config.logger.error(e)
                click.echo(
                    "No postprocessed data read from file. Make sure you "
                    "selected right file or first perform postprocessing on "
                    "the data (-post option)."
                )
                raise click.Abort
        else:
            # check postprocessing parameter availability
            det_freq_source = current_context.get_parameter_source("det_freq")
            if postproc == "fft":
                if det_freq_source == ParameterSource.DEFAULT:
                    if "det_freq" in metadata:
                        # default value -> use metadata
                        det_freq = metadata["det_freq"]
                    else:
                        # prompt user
                        det_freq = click.prompt(
                            "What was the detector's frequency "
                            "in this measurement (in kHz)? (available options: "
                            "1, 2, 5, 10, 20, 40)",
                            type=click.Choice(["1", "2", "5", "10", "20", "40"]),
                        )
                det_freq = _parse_detector_frequency(detfreq=det_freq)

                if chop_freq is None:
                    if "chop_freq" in metadata:
                        # default value -> use metadata
                        chop_freq = metadata["chop_freq"]
                    else:
                        # prompt user
                        chop_freq = click.prompt(
                            "What was the chopper frequency "
                            "in this measurement (in Hz)?",
                            type=float,
                        )

            click.echo(f"Postprocessing - mode {postproc}")
            postprocessing(data, postproc, chop_freq, det_freq.freq * 1000)
            click.echo("\tPostprocessing finished")

            plotting(data=data, path=outpath, save=plot_save)
    elif len(files) > 1:
        raise NotImplementedError("Plotting multiple files not implemented yet")
        if postproc is None:
            raise click.UsageError("No processing mode selected")


@cli.command()
@option_mock_devices
@pass_config
def getPosition(config: Config, mock_devices: bool):
    """
    Print out current position on each of the axes.

    Assumes all three stages are connected.
    """
    click.echo("Initializing devices...")
    try:
        manager = setup_manager(stage_sn=config.stage_sn, is_mockup=mock_devices)
    except SerialException as e:
        config.logger.error(
            "Serial connection error on stage initialization before homing "
            "operation."
        )
        config.logger.error(e)
        click.echo(
            "Serial connection error. Run the app with --debug command to see "
            "details in the log file."
        )
        raise click.Abort
    except RuntimeError as e:
        config.logger.error(
            "Runetime error on stage initialization before homing operation."
        )
        config.logger.error(e)
        click.echo(
            "Runetime error on stage initialization. Verify that stages are "
            "connected and powered on.\nRun the app with --debug command to "
            "see details in the log file."
        )
        raise click.Abort
    click.echo("\tDevices initialized")

    click.echo("Current positions:")
    current_position = manager.current_position
    x = current_position["x"]
    y = current_position["y"]
    z = current_position["z"]
    click.echo(f"\tx: {x}mm\n\ty: {y}mm\n\tz: {z}mm")


@cli.command()
@click.option(
    "-dn",
    "det_sens",
    type=click.Choice(["1", "2", "3", "4"]),
    help="Detector sensor",
    default="1",
)
@click.option(
    "-ds",
    "det_samp",
    type=click.Choice(["100", "200", "500", "1000", "2000", "5000"]),
    help="Detector number of samples",
    default="100",
)
@click.option(
    "-df",
    "det_freq",
    type=click.Choice(["1", "2", "5", "10", "20", "40"]),
    help="Detector sampling frequency (in kHz)",
    default="1",
)
@option_mock_devices
@pass_config
def liveView(
    config: Config, det_sens: str, det_samp: str, det_freq: str, mock_devices: bool
):
    """
    Displays live readout and its FFT from the detector.

    '-dn', '-ds', and '-df' are used to define detector parameters.

    To stop the execution press ENTER while focused on the terminal.
    """
    det_sens, det_samp, det_freq = parse_detector_settings(
        detsens=det_sens, detsamp=det_samp, detfreq=det_freq
    )

    click.echo("Initializing LiveView threads...")
    try:
        manager = setup_manager(
            stage_sn=config.stage_sn,
            is_mockup=mock_devices,
            sensor=det_sens,
            samples=det_samp,
            freq=det_freq,
        )
        lv = LiveView(manager=manager)
    except DeviceNotFoundError as e:
        config.logger.error("Detector not found on initialization.")
        config.logger.error(e)
        click.echo("Bolometer line not detected. Please check connection!")
        raise click.Abort
    click.echo("\tInitialized")

    click.echo("Running LiveView")
    lv.start()
    click.echo("LiveView shut down")
