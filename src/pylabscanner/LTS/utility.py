def steps2mm(steps: int, convunit: int) -> float:
    """Transform value in machine units (microsteps, microsteps/s, microsteps/s^2), to real world values (mm, mm/s, mm/s^2)

    Args:
        steps (int): Number of microsteps.
        convunit (int): Conversion value in microsteps, microsteps/s or microsteps/s^2 (taken directly from the table in "Thorlabs Motion Controllers Host-Controller Communications Protocol", issue 37, 22 May 2023).

    Returns:
        float: Converted value in real world units (mm, mm/s, mm/s^2).
    """
    return steps / convunit


def mm2steps(mm: int | float, convunit: int) -> int:
    """Transform value in real world values (mm, mm/s, mm/s^2) to machine units (microsteps, microsteps/s, microsteps/s^2).

    Args:
        mm (int | float): Distance, velocity or acceleration in mm, mm/s or mm/s^2 accordingly.
        convunit (int): Conversion value in microsteps, microsteps/s or microsteps/s^2 (taken directly from the table in "Thorlabs Motion Controllers Host-Controller Communications Protocol", issue 37, 22 May 2023).

    Returns:
        int: Converted value in machine units (microsteps, microsteps/s or microsteps/s^2)
    """
    return int(mm * convunit)


def error_callback(source, msgid, code, notes):
    print(f"Device {source} reported (error code:{code}, msgid:{msgid}): {notes}")
