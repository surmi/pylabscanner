import numpy as np
from time import sleep
import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Any, Dict
import numpy.typing as npt
from matplotlib.figure import Figure
from serial import SerialException
import logging

from .LTS import LTS, LTSC, error_callback
from .LTS import mm2steps
from .devices import BoloMsgFreq, BoloMsgSamples, BoloMsgSensor


def init_stages(stageslist:str, stage_no:Dict[str, str]) -> List[LTS]:
    """Initialize LTS and LTSC stages.

    Args:
        stageslist (str): which axis to initialize
        stage_no (Dict[str, str]): dictionary with serial numbers of stages

    Returns:
        List[LTS]: list of initialized objects
    """
    stages = []
    logger = logging.getLogger(__name__)
    if stageslist == 'ALL':
            
        try:
            stage_z = LTS(serial_number=stage_no['z'], home=False)
        except SerialException as e:
            logger.error("Exception on serial connection to z axis stage.")
            raise e
        except RuntimeError as e:
            logger.error("Exception on initialization of z axis stage.")
            raise e

        try:
            stage_y = LTS(serial_number=stage_no['y'], home=False)
        except SerialException as e:
            logger.error("Exception on serial connection to y axis stage.")
            raise e
        except RuntimeError as e:
            logger.error("Exception on initialization of y axis stage.")
            raise e

        try:
            stage_x = LTSC(serial_number=stage_no['x'], home=False)
        except SerialException as e:
            logger.error("Exception on serial connection to x axis stage.")
            raise e
        except RuntimeError as e:
            logger.error("Exception on initialization of x axis stage.")
            raise e
        stages.append(stage_x)
        stages.append(stage_y)
        stages.append(stage_z)
    else:
        if 'X' in stageslist or 'x' in stageslist:
            try:
                stage_x = LTSC(serial_number=stage_no['x'], home=False)
            except SerialException as e:
                logger.error("Exception on serial connection to x axis stage.")
                raise e
            except RuntimeError as e:
                logger.error("Exception on initialization of x axis stage.")
                raise e
            stages.append(stage_x)
        if 'Y' in stageslist or 'y' in stageslist:
            try:
                stage_y = LTS(serial_number=stage_no['y'], home=False)
            except SerialException as e:
                logger.error("Exception on serial connection to y axis stage.")
                raise e
            except RuntimeError as e:
                logger.error("Exception on initialization of y axis stage.")
                raise e
            stages.append(stage_y)
        if 'Z' in stageslist or 'z' in stageslist:
            try:
                stage_z = LTS(serial_number=stage_no['z'], home=False)
            except SerialException as e:
                logger.error("Exception on initialization of z axis stage.")
                raise e
            except RuntimeError as e:
                logger.error("Exception on initialization of z axis stage.")
                raise e
            stages.append(stage_z)

    sleep(1)
    for s in stages:
        s.register_error_callback(error_callback)
    return stages


def conv_to_steps(stages:LTS|list[LTS], pos:int|float|list[int]|list[float]) -> int|List[int]:
    """Convert required position to microsteps (used by LTS devices).

    Args:
        stages (LTS | list[LTS]): LTS object representing given stage (or whole list of them)
        pos (int | float | list[int] | list[float]): requested positions for corresponding LTS stages
    """
    if type(stages) is not list: stages = [stages]
    if type(pos) is not list: pos = [pos]
    
    retPos = []
    for s,p in zip(stages, pos):
        retPos.append(mm2steps(p, s.convunits["pos"]))
    if len(retPos) == 1:
        return retPos[0]
    else: return retPos


def parse_range(range:str) -> npt.NDArray:
    """Parse range option. Accepts single float or 3 floats separated with ':'.

    Args:
        range (str): range option

    Raises:
        ValueError: raised if wrong format of the string

    Returns:
        NDArray: array with generated positions to visit
    """
    if range is None:
        return None
    r = [float(i) for i in range.split(':')]
    if len(r) == 1:
        return np.linspace(r[0], r[0], 1)
    elif len(r) == 3:
        return np.linspace(r[0], r[1], int(r[2]))
    else:
        raise ValueError("Wrong number of elements in the range string. Provide 1 or 3 numbers")


def parse_detector_settings(detsens:str, detsamp:str, detfreq:str) -> tuple[BoloMsgSamples, BoloMsgSamples, BoloMsgFreq]:
    """Parse detector settings.

    Args:
        detsens (int): sensor number
        detsamp (int): number of samples
        detfreq (int): sampling frequency

    Returns:
        tuple[BoloMsgSamples, BoloMsgSamples, BoloMsgFreq]: enum objects corresponding to the detector settings
    """
    # for el in BoloMsgFreq:
    #     if str(el.value[1]) == detfreq:
    #         detfreq = el
    detfreq = _parse_detector_frequency(detfreq=detfreq)
    for el in BoloMsgSamples:
        if str(el.value[1]) == detsamp:
            detsamp = el
    if detsens == '1': detsens=BoloMsgSensor.FIRST
    elif detsens == '2': detsens=BoloMsgSensor.SECOND
    elif detsens == '3': detsens=BoloMsgSensor.THIRD
    elif detsens == '4': detsens=BoloMsgSensor.FOURTH
    return (detsens, detsamp, detfreq)


def _parse_detector_frequency(detfreq:str) -> BoloMsgFreq:
    for el in BoloMsgFreq:
        if str(el.value[1]) == detfreq:
            return el


def parse_filepath(filepath:Path, timestamp:bool=True, extension:None|str=None) -> Tuple[Path, str]:
    """Parse file path, timestamp, and extension options.

    Args:
        filepath (Path): output path.
        timestamp (bool): whether to add timestamp to the file name. Defaults to True.
        extension (None | str, optional): force specific extension. Defaults to None.

    Returns:
        Tuple[Path, str]: tuple containing output path and extension of the output file.
    """
    parent = filepath.parent
    stem = filepath.stem
    suffix = filepath.suffix

    if extension is not None:
        if not extension.startswith('.'):
            extension = '.'+extension
        # stem = stem + suffix
        suffix = extension
    elif suffix == '':
        suffix = '.txt'
    
    if timestamp:
        stem += '_{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())
    return (parent/f'{stem}{suffix}', suffix[1:])


def _closest_val(data:npt.ArrayLike, x:float) -> Tuple[int, float]:
    """Find value closest to 'x' in the 'data' and its index.

    Args:
        data (npt.ArrayLike): data to find 'x' in
        x (float): value to find

    Returns:
        Tuple[int, float]: index and value of the closest element
    """
    ind = np.argmin(np.abs(data-x))
    return ind, data[ind]


def postprocessing(
        data:pd.DataFrame,
        mode:str|List[str],
        modulation_frequency:float=None,
        det_freq:float|int=None
):
    """In-place processing of raw data from measurements.
    Data after processing is appended as a new column.
    'modulation_frequency' and 'det_freq' are necessary only for the 'fft' mode.

    Args:
        data (pd.DataFrame): dataframe with raw data
        mode (str|List[str]): what processing method to apply
        modulation_frequency (float, optional): frequency of signal modulation
        to read value from after application of FFT. Defaults to None.
        det_freq (float,int, optional): sampling frequency of the detector

    Raises:
        ValueError: raised if signal frequency or sampling frequency are not provided (only for 'fft' mode)
    """
    # If data is read from file, the type of elements in the 'MEASUREMENT'
    # column will be of string type. Deal with it here.
    if isinstance(data['MEASUREMENT'].at[0], str):
        workdata = data['MEASUREMENT'].str.strip('[]').str.split(',').map(np.array).map(lambda x: x.astype(float))
    else:
        workdata = data['MEASUREMENT']

    # DATA PROCESSING
    if 'mean' in mode:
        val = workdata.map(lambda x: np.mean(x))
        data['MEAN'] = val
        return val

    elif 'fft' in mode:
        if modulation_frequency is None:
            raise ValueError("Signal frequency required")
        if det_freq is None:
            raise ValueError("Signal sample spacing required")

        fft = workdata.map(lambda x: np.fft.rfft(x))

        # calculate closes frequency bin
        sample_spacing = 1/det_freq
        freqs = []
        if isinstance(workdata[0], list):
            freqs = np.fft.rfftfreq(len(workdata[0]), sample_spacing)
        else:
            freqs = np.fft.rfftfreq(workdata[0].size, sample_spacing)
        ind, freq = _closest_val(freqs, modulation_frequency)
        # TODO: check if normalization is correct (1/N)
        val = fft.map(lambda x: np.abs(x[ind])/x.size)
        data['FFT'] = val
        data['FFT_freq'] = freq
        return freq, val


def _predict_plot(data:pd.DataFrame, silent=False) -> Tuple[str, List[str]]:
    """Predict what kind of plot should be made based on the data properties.

    Args:
        data (pd.DataFrame): input data.
        silent (bool, optional): wheter to raise error on fail. Defaults to False.

    Raises:
        ValueError: raised on failed attempt to predict type of the plot. Only raised if 'silent' is equal to False.

    Returns:
        Tuple[str, List[str]]: type of the plot ('2D' or '3D') and order of axis (horizontal and vertical correspondingly).
    """
    ux = data['X'].unique()
    uy = data['Y'].unique()
    uz = data['Z'].unique()
    nx = ux.size
    ny = uy.size
    nz = uz.size

    testar = np.array([nx, ny, nz])
    if np.count_nonzero(np.equal(testar, 1)) == 1:
        # only one axis doesn't change
        # we need a 3D plot (2 axis + val)
        # e.g. imshow(), pcolormesh(), contour(), contourf()
        if nx == 1:
            axorder = ['Y', 'Z']
        elif ny == 1:
            axorder = ['X', 'Z']
        else:
            axorder = ['X', 'Y']
        return '3D', axorder
    elif np.count_nonzero(np.equal(testar, 1)) == 2:
        # two axis doesn't change
        # we need a 2D plot (1 axis + val)
        # e.g. scatter(), plot()
        if nx != 1:
            axorder = ['X']
        elif ny != 1:
            axorder = ['Y']
        else:
            axorder = ['Z']
        return '2D', axorder
    elif not silent:
        raise ValueError("Can't predict type of the plot")
    return None


def plotting(data:pd.DataFrame, path:Path=None, save=False, show=True) -> Tuple[Figure, Any]|None:
    """Plot processed measurements.
    The out path in 'path' argument is assumed to be a full path to the output file
    including the file name. 
    If 'save' argument is set to True, the plot will be saved to the file. If 
    'show' argument is set to True, the plot will be displayed. Both options can
    be used at the same time. For both flags set to False the function skips
    execution and returns early.

    Args:
        data (pd.DataFrame): data.
        path (Path, optional): output path. Defaults to None.
        save (bool, optional): whether to save the plot instead of displaying it. Defaults to False.
        show (bool, optional): whether to save the plot instead of displaying it. Defaults to False.

    Raises:
        ValueError: raised when input data frame does not contain columns with processed data.

    Returns:
        Tuple[Figure, Any]: figure and axes objects with created plot(s).
    """
    # plt.set_loglevel('error')
    # logging.getLogger('PIL').setLevel(logging.ERROR)
    if not (save or show):
        return None
    # if processed data available create two axes to display both
    data.sort_values(by=['X','Y','Z'], inplace=True)
    n = 0
    labels = []
    if 'MEAN' in data.columns:
        n+=1
        labels.append('MEAN')
    if 'FFT' in data.columns:
        n+=1
        labels.append('FFT')
    if n <=0:
        raise ValueError("No data to plot")
    
    collim = 3
    ncols = collim if n >= collim else n
    nrows = np.ceil(n/collim).astype(int)
    fig, axs = plt.subplots(
        ncols=ncols, nrows=nrows, figsize=(3*(ncols+0.5), nrows*3),
        layout='constrained'
    )
    
    # define type of plots
    pltype, axorder = _predict_plot(data)
    if isinstance(axs, plt.Axes):
        axs = np.array([axs])
    for label, ax in zip(labels, axs.flat):
        if pltype == '2D':
            x = data[axorder[0]]
            y = data[label]
            ax.plot(x, y, 'o', ms=4)
            ax.set_title(label)
        elif pltype == '3D':
            ux = data[axorder[0]].unique()
            x = data[axorder[0]]
            uy = data[axorder[1]].unique()
            y = data[axorder[1]]
            val = data[label].to_numpy().reshape((uy.size, ux.size))
            img = ax.imshow(val, cmap='inferno', aspect='equal', origin='lower', extent=[x.min(), x.max(), y.min(), y.max()])
            fig.colorbar(img, ax=ax, orientation='horizontal')
            ax.set_title(label)
    if save:
        path.parent.mkdir(exist_ok=True)
        
        # add label to the file name
        parent = path.parent
        stem = path.stem + f'_plot'
        format = 'png'
        path = parent/f'{stem}.{format}'
        
        plt.savefig(fname=path, format=format, bbox_inches='tight')
    if show:
        plt.show()
    return fig, axs


def saving(data:pd.DataFrame, path:Path, label:str=None):
    """Save data to a file.

    Args:
        data (pd.DataFrame): data frame with measurements.
        path (Path): path to where the data should be saved.
        label (str, optional): if provided, will be attached to the file name. Defaults to None.
    """
    if label is not None:
        # modif filename
        parent = path.parent
        stem = path.stem + f'_{label}'
        suffix = path.suffix
        path = parent/f'{stem}{suffix}'

    path.parent.mkdir(exist_ok=True)

    # TODO: define other saving methods
    with path.open('w+') as f:
        data.to_csv(f)