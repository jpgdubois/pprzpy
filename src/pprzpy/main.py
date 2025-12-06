from pathlib import Path
import logging
import matplotlib.pyplot as plt

from pprzpy.tools.parse_data import parse_data_file, parse_schema_xml, deserialize_payload
from pprzpy.tools.select_log import get_latest_log_paths
from pprzpy.plotters.plot_stab_attitude import plot_stab_attitude
from pprzpy.plotters.plot_cyclone_actuators import plot_cyclone_actuators
from pprzpy.plotters.plot_stab_thrust import plot_stab_thrust
from pprzpy.plotters.plot_sensors import plot_imu_gyro, plot_imu_gyro_derivative
from pprzpy.plotters.plot_objective import plot_objective
from pprzpy.plotters.plot_pseudo_command import plot_stab_pseudo_command
from pprzpy.plotters.plot_control_allocation import plot_control_allocation
from pprzpy.plotters.plot_ext_pose_down import plot_ext_pose_down
from pprzpy.plotters.plot_system_monitor import plot_system_monitor
from pprzpy.plotters.plot_ahrs_bias import plot_ahrs_bias

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():

    file_path, protocol_path = get_latest_log_paths(Path("data"), recursive=True)

    # Expensive operations, avoid doing them multiple times
    schema = parse_schema_xml(protocol_path)
    df = parse_data_file(file_path)

    plot_stab_attitude(df, schema)
    plot_stab_thrust(df, schema)

    plot_stab_pseudo_command(df, schema)

    plot_cyclone_actuators(df, schema)
    plot_control_allocation(df, schema)

    plot_ext_pose_down(df, schema)
    plot_ahrs_bias(df, schema)

    # plot_imu_gyro(df, schema)
    # plot_imu_gyro_derivative(df, schema)
    plot_system_monitor(df, schema)

    plt.show()


if __name__ == "__main__":
    main()