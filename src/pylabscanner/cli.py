import asyncio
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

from .devices import BoloLine, DeviceNotFoundError
from .devices.LTS import aso_home_devs, aso_move_devs, steps2mm

# from .commands import LineStart, LineType, LiveView, ScanRoutine
from .scheduler.commands import LineStart, LineType
from .utils import (
    _parse_detector_frequency,
    conv_to_steps,
    init_stages,
    parse_detector_settings,
    parse_filepath,
    parse_range,
    plotting,
    postprocessing,
    saving,
)


class Config(object):
    def __init__(self):
        # self.verbose = False
        self.debug = False
        self.logger = None
        self.stage_sn = {}


pass_config = click.make_pass_decorator(Config, ensure=True)


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
            # filemode='w',
            format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
            level=logging.DEBUG,
        )
    else:
        logging.basicConfig(level=logging.ERROR)

    # logging.basicConfig(filename="log.txt",
    # logging.basicConfig(filename=logging_destination,
    #                 # filemode='a',
    #                 filemode='w',
    #                 format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    #                 datefmt='%H:%M:%S',
    #                 level=logging_level)
    # level=logging.DEBUG)
    confobj.logger = logging.getLogger(__name__)
    # if debug:
    #     click.echo("Debug mode is on")
    #     confobj.debug = debug
    # else:
    #     confobj.logger.setLevel(logging.ERROR)

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
@pass_config
def home(config: Config, stageslist):
    """
    Homes Thorlabs stages.

    Available options: 'X', 'Y', 'Z', 'ALL'. Defaults to 'ALL'.
    """
    click.echo("Initializing devices...")
    try:
        stages = init_stages(stageslist=stageslist, stage_no=config.stage_sn)
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
    asyncio.run(aso_home_devs(stages), debug=config.debug)
    te_home1 = time()
    click.echo(f"\tStages homed in: {te_home1-start:.2f}s")


@cli.command()
@click.option("-x", type=float, default=None, required=False)
@click.option("-y", type=float, default=None, required=False)
@click.option("-z", type=float, default=None, required=False)
@pass_config
def moveTo(config: Config, x, y, z):
    """
    Moves stages to given position.

    Provide distance for at least one axis ('-x', '-y' or '-z').
    """
    stagesstr = ""
    pos = []
    if x is not None:
        stagesstr += "X"
        pos.append(x)
    if y is not None:
        stagesstr += "Y"
        pos.append(y)
    if z is not None:
        stagesstr += "Z"
        pos.append(z)

    if len(stagesstr) == 0:
        raise click.UsageError("Provide distance for at least one axis.")

    click.echo("Initializing devices...")
    try:
        stages = init_stages(stageslist=stagesstr, stage_no=config.stage_sn)
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
    pos = conv_to_steps(stages, pos)
    click.echo("\tDevices initialized")

    click.echo("Moving to designated position(s)...")
    start = time()
    try:
        asyncio.run(aso_move_devs(stages, pos), debug=config.debug)
    except Exception as er:
        config.logger.error("Error while running asynchronous movement to position")
        config.logger.error(er)
        click.echo(
            "Error while running asynchronous movement to position. Run in "
            "debug mode to see more details."
        )
        raise click.Abort
    te_home1 = time()
    click.echo(f"\tStages moved in: {te_home1-start:.2f}s")


# @cli.command(context_settings={"show_default": True})
# @click.option("-x", type=str, help="Scanning range for X axis.")
# @click.option("-y", type=str, help="Scanning range for Y axis")
# @click.option("-z", type=str, help="Scanning range for Z axis")
# @click.option(
#     "-o",
#     "outpath",
#     type=click.Path(
#         writable=True,
#         path_type=Path,
#         resolve_path=True,
#     ),
#     default="./out/out.txt",
#     help="File name or path to the file in which results will be written.",
# )
# @click.option(
#     "-m",
#     "mode",
#     type=click.Choice(["flyby", "ptbypt"]),
#     default="ptbypt",
#     help="Scanning mode",
# )
# @click.option(
#     "-s",
#     "linestart",
#     type=click.Choice(["snake", "cr"]),
#     default="snake",
#     help="Where should each line start",
# )
# @click.option(
#     "-dn",
#     "det_sens",
#     type=click.Choice(["1", "2", "3", "4"]),
#     help="Detector sensor",
#     default="1",
# )
# @click.option(
#     "-ds",
#     "det_samp",
#     type=click.Choice(["100", "200", "500", "1000", "2000", "5000"]),
#     help="Detector number of samples",
#     default="100",
# )
# @click.option(
#     "-df",
#     "det_freq",
#     type=click.Choice(["1", "2", "5", "10", "20", "40"]),
#     help="Detector sampling frequency (in kHz)",
#     default="1",
# )
# @click.option(
#     "-nc",
#     "noconfirmation",
#     is_flag=True,
#     help="Run the scan without further confirmation",
# )
# @click.option(
#     "-ts",
#     "timestamp",
#     is_flag=True,
#     help="Wheather to append timestamp to the filename",
#     default=False,
# )
# @click.option("-ext", "extension", type=str, help="Enforce specific extension")
# @click.option(
#     "-plt", "plot", is_flag=True, default=False, help="Whether to plot the output data"
# )
# @click.option(
#     "-plts",
#     "plot_save",
#     is_flag=True,
#     default=False,
#     help="Whether to save the plot of the output data",
# )
# @click.option(
#     "-post",
#     "postproc",
#     default="raw",
#     type=click.Choice(["raw", "mean", "fft"]),
#     help="Postprocessing of obtained measurements",
# )
# @click.option(
#     "-f", "chop_freq", type=float, help="Signal modulation frequency in Hz (if used)"
# )
# @pass_config
# def scan(
#     config: Config,
#     x,
#     y,
#     z,
#     outpath: Path,
#     mode,
#     noconfirmation,
#     linestart,
#     det_sens,
#     det_samp,
#     det_freq,
#     timestamp,
#     extension,
#     plot,
#     plot_save,
#     postproc,
#     chop_freq,
# ):
#     """
#     Performs scanning operation.

#     CAUTION: single line scans without providing ranges for other axis not
#     yet implemented!

#     Detector has three settings: sensor selection '-dn', number of samples per
#     measurement '-ds', and sampling frequency '-df'. Number of samples and
#     sampling frequency influences time of single measurement. Note that this
#     influences how much time the stages will wait at given position (for
#     'ptbypt' mode) or how large distance will be swept over during single
#     measurement (for 'flyby' mode).

#     Two scanning modes are available: 'flyby' and 'ptbypt'.
#     'flyby' performs measurements during sweeping motion along one of the axis.
#     'ptbypt' moves all stages to desired position, performs measurement, and
#     moves to the next planned measurement point.

#     This command provides also two options ('-s') on which side new line will
#     begin. 'snake' will force beginning of next lines to alternate (creating
#     snake-like pattern). When 'cr' is selected after a line is finished stages
#     will come back to the starting position (like a carriage returning in a
#     typewriter to the beginning).

#     For scanning ranges (options '-x', '-y' and '-z') provide either 1 or 3
#     numbers. Single number is assumed to be a single position to which given
#     stage will move. If you with to provide the whole range, you need to pass 3
#     numbers separated with ':' (colon). The order in the range:
#     beginning:end:number_of-points_to_scan. All values passed as ranges are
#     assumed to be in mm.

#     File name or whole valid path (may be relative) with file name can be
#     provided via '-o' option. Note that file is open in w+ mode (it will be
#     overwritten if exists). If '-ext' option is provided the '-ext' value
#     will be appended to the value of '-o' option as extension. Without '-ext'
#     option and with no extension in '-o' value default extension is '.txt'.
#     Providing '-ts' flag appends timestamp to the file name.

#     The result of measurement is a series of samples per single scanning point
#     (number of samples correspond to '-ds' value). Data can be additionally
#     postprocessed. Postprocessing setting is changed with '-post' option.
#     Available options are: 'mean' (average of samples per measured point), or
#     'fft' (data point measured at specified frequency). 'fft' mode requires
#     additional value passed with '-f' option (modulation frequency of the
#     signal).

#     Since after postprocessing each point has single numerical value,
#     the program can provide a simple plot (flag '-plt'). With '-plt' flag
#     program will generate and display the plot. If '-plts' is provided
#     the plot will not be displayed but saved to file. If the scan contains
#     only single line the plot will contain line plot but for more scan lines
#     the plot will display heatmap.

#     \b
#     Examples of valid ranges:
#         '-x 100' - performs measurement with stage X at 100 mm.
#         '-y 10:100:3' - performs measurement with stage Y at positions 10, 55,
#         and 100.

#     \b
#     Example of minimal command:
#         'labscanner scan -x 10:100:3 -y 100 -z 100' - 3 point scan for y=100,
#         z=100, and x=[10,55,100].
#     """

#     # TODO: add other methods for saving data
#     # TODO: implement different file format per extension
#     # TODO: add data from postprocessing to separate file
#     # TODO: identify modulation (chopper) frequency limit
#     # TODO: move scan parsing to separate function (return settings dictionary?)
#     # TODO: differentiate plot and plot_save behaviour
#     # parse input
#     try:
#         measrngx = parse_range(x)
#         measrngy = parse_range(y)
#         measrngz = parse_range(z)
#     except ValueError as e:
#         raise click.UsageError(e)
#     if measrngx is None or measrngy is None or measrngz is None:
#         # TODO: remove when ready
#         raise click.UsageError(
#             "Single line scans with only one range provided not available yet."
#         )
#     if measrngx is None and measrngy is None and measrngz is None:
#         # return early if not enough information is provided
#         raise click.UsageError("Provide scanning range for at least one axis")
#     stagesstr = ""
#     ranges = []
#     if measrngx is not None:
#         stagesstr += "x"
#         ranges.append(measrngx)
#     if measrngy is not None:
#         stagesstr += "y"
#         ranges.append(measrngy)
#     if measrngz is not None:
#         stagesstr += "z"
#         ranges.append(measrngz)
#     if mode == "ptbypt":
#         mode = LineType.PTBYPT
#     elif mode == "flyby":
#         mode = LineType.FLYBY
#     if linestart == "snake":
#         linestart = LineStart.SNAKE
#     elif linestart == "cr":
#         linestart = LineStart.CR
#     det_sens, det_samp, det_freq = parse_detector_settings(
#         detsens=det_sens, detsamp=det_samp, detfreq=det_freq
#     )
#     outpath, extension = parse_filepath(
#         filepath=outpath, timestamp=timestamp, extension=extension
#     )
#     if postproc == "raw" and plot:
#         raise click.UsageError("Can't plot raw data")
#     elif postproc == "fft" and chop_freq is None:
#         raise click.UsageError("Can't plot FFT data without modulation frequency")

#     # initialize devices
#     click.echo("Initializing devices...")
#     try:
#         stages = init_stages(stageslist=stagesstr, stage_no=config.stage_sn)
#     except SerialException as e:
#         config.logger.error(
#             "Serial connection error on stage initialization before homing "
#             "operation."
#         )
#         config.logger.error(e)
#         click.echo(
#             "Serial connection error. Run the app with --debug command to see "
#             "details in the log file."
#         )
#         raise click.Abort
#     except RuntimeError as e:
#         config.logger.error(
#             "Runetime error on stage initialization before homing operation."
#         )
#         config.logger.error(e)
#         click.echo(
#             "Runetime error on stage initialization. Verify that stages are "
#             "connected and powered on.\nRun the app with --debug command to "
#             "see details in the log file."
#         )
#         raise click.Abort
#     try:
#         bl = BoloLine(sensor=det_sens, samples=det_samp, freq=det_freq, cold_start=True)
#     except DeviceNotFoundError as e:
#         config.logger.error("Bolometer line not detected. Please check " "connection!")
#         config.logger.error(e)
#         click.echo("Bolometer line not detected. Please check connection!")
#         raise click.Abort
#     except SerialTimeoutException as e:
#         config.logger.error("Timeout on bolometer line connection")
#         config.logger.error(e)
#         click.echo(f"Timeout on bolometer line connection: {e}")
#         raise click.Abort

#     metadata = {
#         "detector name": "Luvitera THz Mini, 4 sensor bolometer line",
#         "detector sensor number": det_sens.name,
#         "detector sampling": det_samp.nsamp,
#         "detector sampling frequency [kHz]": det_freq.freq,
#         "signal modulation frequency [Hz]": chop_freq,
#         "x axis range [beg:end:no pts|pos]": x,
#         "y axis range [beg:end:no pts|pos]": y,
#         "z axis range [beg:end:no pts|pos]": z,
#         "scanning mode": mode,
#         "scanning line start": linestart,
#     }
#     for stage in stages:
#         if stage.serial_number == config.stage_sn["x"]:
#             metadata["x axis device"] = str(stage)
#         elif stage.serial_number == config.stage_sn["y"]:
#             metadata["y axis device"] = str(stage)
#         elif stage.serial_number == config.stage_sn["z"]:
#             metadata["z axis device"] = str(stage)

#     click.echo("\tDevices initialized")

# # Build the scanning routine
# sc = ScanRoutine(
#     stages=stages,
#     detector=bl,
#     source=None,
#     ranges=ranges,
#     line_start=linestart,
#     line_type=mode,
# )
# sc.build()

# # print the setup parameters and ask for confirmation
# click.echo("---")
# click.echo("Ranges to scan:")
# click.echo(
#     f"\tx: {x if x is not None else 0} "
#     f"[{len(measrngx) if measrngx is not None else 0} point(s) per line]"
# )
# click.echo(
#     f"\ty: {y if y is not None else 0} "
#     f"[{len(measrngy) if measrngy is not None else 0} point(s) per line]"
# )
# click.echo(
#     f"\tz: {z if z is not None else 0} "
#     f"[{len(measrngz) if measrngz is not None else 0} point(s) per line]"
# )
# click.echo(
#     "Total number of points in the scan: "
#     f"{len(measrngx)*len(measrngy)*len(measrngz)}"
# )
# click.echo(f"Estimated time of measurement: {floor(sc.ta/60)}m {sc.ta % 60:.0f}s")
# click.echo("---\n")

# if noconfirmation or click.confirm("Do you want to continue?"):
#     # perform the operation
#     click.echo("Measurement started")
#     try:
#         sc.run()
#     except SerialTimeoutException as e:
#         config.logger.error("Timeout on serial port")
#         config.logger.error(e)
#         click.echo(f"Timeout on serial port: {e}")
#         # home the stages
#         asyncio.run(aso_home_devs(stages), debug=config.debug)

#         raise click.Abort
#     data = sc.data
#     click.echo(
#         f"Actual time of measurement: {floor(sc.ta_act/60)}m "
#         f"{sc.ta_act % 60:.0f}s"
#     )
#     click.echo("\tMeasurement finished")

#     # save raw data to file
#     saving(data, metadata, outpath)

#     if postproc != "raw":
#         # do the postprocessing
#         click.echo(f"Postprocessing - mode {postproc}")
#         try:
#             postprocessing(data, postproc, chop_freq, det_freq.freq * 1000)
#         except BaseException as exception_any:
#             saving(data, outpath, "failed_postproc")
#             click.echo("Exception while performing postprocessing")
#             config.logger.error(exception_any)
#             click.echo(exception_any)
#             raise exception_any
#             # raise click.Abort
#         click.echo("\tPostprocessing finished")

#         # save processed data to file
#         saving(data, metadata=metadata, path=outpath, label="postproc")

#     if plot or plot_save:
#         plotting(data, path=outpath, save=plot_save, show=plot)
# else:
#     click.echo("Measurement aborted")


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
@pass_config
def getPosition(config: Config):
    """
    Print out current position on each of the axes.

    Assumes all three stages are connected.
    """
    click.echo("Initializing devices...")
    try:
        stages = init_stages(stageslist="ALL", stage_no=config.stage_sn)
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
    x, y, z = [steps2mm(s.status["position"], s.convunits["pos"]) for s in stages]
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
@pass_config
def liveView(config: Config, det_sens: str, det_samp: str, det_freq: str):
    """
    Displays live readout and its FFT from the detector.

    '-dn', '-ds', and '-df' are used to define detector parameters.

    To stop the execution press ENTER while focused on the terminal.
    """
    det_sens, det_samp, det_freq = parse_detector_settings(
        detsens=det_sens, detsamp=det_samp, detfreq=det_freq
    )

    click.echo("Initializing LiveView threads...")
    # try:
    #     lv = LiveView(
    #         detector=BoloLine(
    #             sensor=det_sens, samples=det_samp, freq=det_freq, cold_start=True
    #         )
    #     )
    # except DeviceNotFoundError as e:
    #     config.logger.error("Detector not found on initialization.")
    #     config.logger.error(e)
    #     click.echo("Bolometer line not detected. Please check connection!")
    #     raise click.Abort
    click.echo("\tInitialized")

    click.echo("Running LiveView")
    # lv.start()
    click.echo("LiveView shut down")
