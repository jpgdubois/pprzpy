from pathlib import Path
import logging
import matplotlib.pyplot as plt

from pprzpy.tools.parse_data import parse_data_file, parse_schema_xml, deserialize_payload
from pprzpy.plotters.plot_stab_attitude import plot_stab_attitude
from pprzpy.plotters.plot_cyclone_actuators import plot_cyclone_actuators
from pprzpy.plotters.plot_stab_thrust import plot_stab_thrust

def main():
    file_path = Path("/home/jpg/Projects/pprzpy/data/2025-11-15_15-14/22_05_01__01_59_46_SD.data")
    protocol_path = Path("/home/jpg/Projects/pprzpy/data/2025-11-15_15-14/22_05_01__01_59_46_SD.log")

    # Expensive operations, avoid doing them multiple times
    schema = parse_schema_xml(protocol_path)
    df = parse_data_file(file_path)

    # Call the plotting function
    plot_stab_attitude(df, schema)
    plot_stab_thrust(df, schema)

    plot_cyclone_actuators(df, schema)
    plt.show()


if __name__ == "__main__":
    main()