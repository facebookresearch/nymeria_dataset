#!/usr/bin/env python3
"""
Export Nymeria dataset IMU sensor poses and biases to CSV files.

For each device (head, observer, lwrist, rwrist) and each IMU sensor
(imu-left, imu-right), exports:
  - T_world_sensor at every native IMU timestamp
  - Gyroscope and accelerometer biases (online or factory calibration)

All timestamps in the output CSV are in TIME_CODE domain (shared across
all devices) so that rows from different recordings are directly comparable.

Output CSV columns:
  timestamp_ns, x, y, z, qw, qx, qy, qz,
  gyro_radsec_x, gyro_radsec_y, gyro_radsec_z,
  accel_msec2_x, accel_msec2_y, accel_msec2_z,
  bias_gyro_x, bias_gyro_y, bias_gyro_z,
  bias_accel_x, bias_accel_y, bias_accel_z

TEMPORAL ALIGNMENT:
  IMU timestamps in VRS represent the raw capture timestamp (DEVICE_TIME).
  However, due to internal signal processing delays in the IMU, the instant
  represented by this timestamp differs from the true measurement instant.

  According to Project Aria documentation:
    â(t_Device) = ã(t_Device + dt_Device_Gyro + 0.5*Δt)

  Where:
    - dt_Device_Gyro is the gyroscope time offset (estimated during calibration)
    - Δt is the IMU sampling period
    - 0.5*Δt accounts for the center of the integration window

  We use ONLY the gyroscope time offset (not accelerometer) because:
    - Gyroscope observes rotational motion which is highly correlated with
      visual motion observed by cameras (used as reference for calibration)
    - The accelerometer time offset estimation is less observable and thus
      less reliable due to the noisier optimization landscape
    - Both sensors share the same timestamp source in the VRS stream,
      so the gyro offset is a better estimate of the true sensor-to-device
      time alignment

  The corrected timestamp for pose query is:
    t_corrected = t_device_raw + dt_gyro_ns + half_period_ns
"""

import csv
import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import click
import numpy as np
from loguru import logger
from tqdm import tqdm


DEVICE_TAGS = ["head", "observer", "lwrist", "rwrist"]
SENSOR_LABELS = ["imu-left", "imu-right"]

CSV_HEADER = [
    "timestamp_ns",

    "x",
    "y",
    "z",
    
    # Hamilton convention for quaternion
    "qw",
    "qx",
    "qy",
    "qz",

    # gyro in rad per sec
    "gyro_radsec_x",
    "gyro_radsec_y",
    "gyro_radsec_z",
    
    # accel in meter per sec²
    "accel_msec2_x",
    "accel_msec2_y",
    "accel_msec2_z",

    # biases
    "bias_gyro_x",
    "bias_gyro_y",
    "bias_gyro_z",
    "bias_accel_x",
    "bias_accel_y",
    "bias_accel_z",
]


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from loguru import logger


@contextmanager
def suppress_native_stderr(show_on_error: bool = True):
    """Redirige le stderr natif (C++) vers un fichier temp.
    En cas d'exception, réaffiche le contenu capturé."""
    stderr_fd = sys.stderr.fileno()
    old_stderr = os.dup(stderr_fd)
    tmp = tempfile.NamedTemporaryFile(mode="w+", suffix=".log", delete=False)

    try:
        os.dup2(tmp.fileno(), stderr_fd)
        yield tmp
    except Exception:
        # Flush et relis le contenu capturé
        tmp.flush()
        tmp.seek(0)
        captured = tmp.read()
        # Restaure stderr avant de logger
        os.dup2(old_stderr, stderr_fd)
        os.close(old_stderr)
        old_stderr = -1
        if show_on_error and captured.strip():
            logger.error(f"Logs natifs capturés avant le crash:\n{captured}")
        raise
    finally:
        if old_stderr != -1:
            os.dup2(old_stderr, stderr_fd)
            os.close(old_stderr)
        tmp.close()
        Path(tmp.name).unlink(missing_ok=True)


def load_report(report_path: Path) -> dict:
    """Load existing JSON report or create empty one."""
    if report_path.is_file():
        with open(report_path) as f:
            return json.load(f)
    return {"created": datetime.now().isoformat(), "sequences": {}}


def save_report(report_path: Path, report: dict) -> None:
    """Save JSON report (atomic-ish write via tmp rename)."""
    report["last_updated"] = datetime.now().isoformat()
    tmp = report_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    tmp.rename(report_path)


# ---------------------------------------------------------------------------
# Sequence discovery
# ---------------------------------------------------------------------------


def discover_sequences(batch_path: Path) -> list[Path]:
    """
    Discover sequence directories from a batch path.

    If batch_path is a .txt file: one sequence directory path per line.
    If batch_path is a directory: each subdirectory is a sequence.
    """
    if batch_path.is_file() and batch_path.suffix == ".txt":
        sequences = []
        with open(batch_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                p = Path(line)
                if p.is_dir():
                    sequences.append(p)
                else:
                    logger.warning(f"Skipping invalid path: {line}")
        return sequences

    if batch_path.is_dir():
        return sorted([p for p in batch_path.iterdir() if p.is_dir()])

    raise ValueError(f"batch_path must be a .txt file or a directory: {batch_path}")


# ---------------------------------------------------------------------------
# IMU time offset helpers
# ---------------------------------------------------------------------------


def get_gyro_time_offset_sec(imu_calib) -> float:
    """
    Get the gyroscope time offset from an ImuCalibration object.

    Falls back to 0.0 if the method is not available (older API versions).
    """

    # Human: Big doubts on the fact that there are multiple ways that were defined
    # to access gyro time offset ... so we try the expected way (currently documented)
    # first and then we let whatever Claude has done related to some doc he's read

    # The originally pinned version 1.5.5, has seemingly no access to time offsets 
    # Seems that the correct version is this (using version 2.1.1)
    if hasattr(imu_calib, "get_time_offset_sec_device_gyro"):
        return imu_calib.get_time_offset_sec_device_gyro()

    # Leaves the others access just in case, but we should inspect
    # other versions of projectaria_tools to understand the history
    # of time offset in calibration

    if hasattr(imu_calib, "get_time_offset"):
        return imu_calib.get_time_offset()

    try:
        # Try the expected API method
        gyro_model = imu_calib.get_gyro_model()
        if hasattr(gyro_model, "get_time_offset"):
            return gyro_model.get_time_offset()
    except Exception:
        pass
    
    # Fallback: no time offset correction
    return 0.0


def retrieve_sampling_period_ns(sensor_calib, timestamps_ns: np.ndarray) -> int:
    """
    Retrieve from calibration or compute the sampling period from timestamps.

    Args:
        timestamps_ns: Array of timestamps in nanoseconds

    Returns:
        Sampling period or computed median sampling period in nanoseconds (eg. 1ms for 1kHz)
    """

    if sensor_calib:
        exit(1)

    if len(timestamps_ns) < 2:
        # Default to 1ms (1kHz) if not enough samples
        return 1_000_000

    diffs = np.diff(timestamps_ns)
    median_period = int(np.median(diffs))
    return median_period


# ---------------------------------------------------------------------------
# Single-sequence export
# ---------------------------------------------------------------------------


def _export_sensor_csv(
    ndp,  # NymeriaDataProvider
    rec,  # RecordingDataProvider
    device_tag: str,
    sensor_label: str,
    csv_path: Path,
    use_online_calib: bool,
    quiet: bool = False,
) -> int:
    """
    Export one CSV for one device/sensor pair.

    Temporal alignment strategy:
      1. Read raw IMU timestamps from VRS (DEVICE_TIME domain)
      2. Apply gyroscope time offset + 0.5*Δt correction to get the true
         measurement instant in DEVICE_TIME
      3. Query pose at the corrected DEVICE_TIME timestamp
      4. Convert corrected timestamp to TIME_CODE for CSV output
         (enables cross-device comparability)

    Returns the number of samples written.
    """
    from projectaria_tools.core.sensor_data import TimeDomain

    # -------------------------------------------------------------------------
    # 1. Get raw IMU timestamps (DEVICE_TIME domain, uncorrected)
    # -------------------------------------------------------------------------
    all_imu_data_raw = rec.get_sensor_data_with_device_time_ns(sensor_label)

    if len(all_imu_data_raw) == 0:
        logger.warning(f"  {device_tag}/{sensor_label}: no data found, skipping")
        return 0

    # -------------------------------------------------------------------------
    # 2. Compute time offset correction
    #
    #    According to Project Aria temporal alignment documentation:
    #      â(t_Device) = ã(t_Device + dt_Device_Gyro + 0.5*Δt)
    #
    #    We use ONLY the gyroscope time offset because:
    #    - Gyro motion is highly correlated with visual motion (better observable)
    #    - Accelerometer time offset estimation is noisier and less reliable
    #    - Both sensors share the same raw timestamp in VRS
    # -------------------------------------------------------------------------

    # Get calibration to extract gyro time offset
    # For factory calib: use static (t_ns=None)
    # For online calib: use first timestamp as reference (offset varies slowly)
    if use_online_calib:
        ref_calib = ndp.get_sensor_calibration(
            device_tag,
            sensor_label,
            t_ns=int(all_imu_data_raw[0].capture_timestamp_ns),
            time_domain=TimeDomain.DEVICE_TIME,
        )
        dt_gyro_sec = get_gyro_time_offset_sec(ref_calib)
        dt_gyro_ns = int(dt_gyro_sec * 1e9)
    else:
        static_calib = ndp.get_sensor_calibration(
            device_tag, sensor_label, t_ns=None
        )
        dt_gyro_ns = 0

    # Compute half sampling period (center of integration window)
    # @todo retrieve perfect sampling period from VRS or Calib information
    sampling_period_ns = retrieve_sampling_period_ns(None, [d.capture_timestamp_ns for d in all_imu_data_raw])
    half_period_ns = sampling_period_ns // 2

    # Total correction to apply
    total_offset_ns = dt_gyro_ns + half_period_ns

    logger.info(
        f"  {device_tag}/{sensor_label}: "
        f"{len(all_imu_data_raw)} samples -> {csv_path.name} "
        f"(dt_gyro={dt_gyro_sec*1000:.3f}ms, Δt/2={half_period_ns/1e6:.3f}ms)"
    )


    # -------------------------------------------------------------------------
    # 3. Export loop
    # -------------------------------------------------------------------------

    vrs_dp = rec.vrs_dp
    n_written = 0
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

        for imu_data in tqdm(all_imu_data_raw, desc=f"{device_tag}/{sensor_label}", disable=quiet):
            t_raw_ns = int(imu_data.capture_timestamp_ns)

            # Apply time offset correction
            t_corrected_ns = t_raw_ns + total_offset_ns

            # -- Pose: T_world_sensor at CORRECTED timestamp --
            T_world_sensor, _ = ndp.get_sensor_pose(
                t_corrected_ns,
                device_tag,
                sensor_label,
                time_domain=TimeDomain.DEVICE_TIME,
            )

            # -- Calibration (biases) --
            if use_online_calib:
                calib = ndp.get_sensor_calibration(
                    device_tag,
                    sensor_label,
                    t_ns=t_corrected_ns,
                    time_domain=TimeDomain.DEVICE_TIME,
                )
            else:
                # use the first online calibration because it seems that
                # we have no other way to have an offline estimate of
                # gyro/accel biases of the sensor (or I missed it)
                calib = ref_calib

            gyro_bias = np.array(calib.get_gyro_model().get_bias())
            accel_bias = np.array(calib.get_accel_model().get_bias())

            # -- Convert CORRECTED timestamp to TIME_CODE for CSV --
            # This ensures cross-device comparability
            t_timecode_ns = vrs_dp.convert_from_device_time_to_timecode_ns(t_corrected_ns)

            # -- Extract pose components --
            xyz = T_world_sensor.translation()[0]
            # to_quat_and_translation() returns [ quat_wxyz translation_xyz ] as a single array
            # prefer rotation() that returns SOE which can then be used to get the quaternion
            # SOE converts to hamilton convetion (wxyz)
            quat_wxyz = T_world_sensor.rotation().to_quat()[0]

            # import pdb
            # breakpoint()

            accel = imu_data.accel_msec2
            gyro  = imu_data.gyro_radsec

            writer.writerow([
                t_timecode_ns,
                f"{xyz[0]:.8f}",
                f"{xyz[1]:.8f}",
                f"{xyz[2]:.8f}",

                f"{quat_wxyz[0]:.8f}",
                f"{quat_wxyz[1]:.8f}",
                f"{quat_wxyz[2]:.8f}",
                f"{quat_wxyz[3]:.8f}",

                f"{accel[0]:.8f}",
                f"{accel[1]:.8f}",
                f"{accel[2]:.8f}",
                f"{gyro[0]:.8f}",
                f"{gyro[1]:.8f}",
                f"{gyro[2]:.8f}",

                f"{gyro_bias[0]:.10f}",
                f"{gyro_bias[1]:.10f}",
                f"{gyro_bias[2]:.10f}",
                f"{accel_bias[0]:.10f}",
                f"{accel_bias[1]:.10f}",
                f"{accel_bias[2]:.10f}",
            ])
            n_written += 1

    return n_written


def export_sequence(
    sequence_dir: Path,
    output_dir: Path,
    use_online_calib: bool,
    hide_loading_logs: bool,
    quiet: bool = False,
) -> dict:
    """
    Export all IMU trajectories for one sequence.

    Returns dict with status and details for the report.
    """
    from nymeria.recording_data_provider import RecordingDataProvider
    from nymeria.data_provider import NymeriaDataProvider

    result = {
        "status": "success",
        "sequence": str(sequence_dir),
        "devices": {},
        "errors": [],
    }

    try:
        # Create output subdirectory for this sequence
        seq_output = output_dir / sequence_dir.name
        seq_output.mkdir(parents=True, exist_ok=True)

        # Initialize provider
        with suppress_native_stderr():
            # If no data were found, we already have a try/except clause
            # to catch the exception and log the error
            ndp = NymeriaDataProvider(
                sequence_rootdir=sequence_dir,
                load_head=True,
                load_observer=True,
                load_wrist=True,
                load_body=False,  # Don't need body motion for IMU export
                load_online_calib=use_online_calib,
            )

        # Export each device/sensor combination
        for device_tag in DEVICE_TAGS:
            try:
                rec = ndp.get_recording(device_tag)
            except ValueError:
                # Recording not loaded (not available or disabled)
                continue

            if not rec.has_pose:
                logger.warning(f"  {device_tag}: no trajectory, skipping")
                continue

            result["devices"][device_tag] = {}

            for sensor_label in SENSOR_LABELS:
                csv_name = f"{device_tag}_{sensor_label}.csv"
                csv_path = seq_output / csv_name

                try:
                    n_samples = _export_sensor_csv(
                        ndp=ndp,
                        rec=rec,
                        device_tag=device_tag,
                        sensor_label=sensor_label,
                        csv_path=csv_path,
                        use_online_calib=use_online_calib,
                        quiet=quiet,
                    )
                    result["devices"][device_tag][sensor_label] = {
                        "samples": n_samples,
                        "file": csv_name,
                    }
                except Exception as e:
                    error_msg = f"{device_tag}/{sensor_label}: {e}\n{traceback.format_exc()}"
                    logger.error(f"  {error_msg}")
                    result["errors"].append(error_msg)
                    result["devices"][device_tag][sensor_label] = {"error": str(e)}

    except Exception as e:
        result["status"] = "failed"
        result["errors"].append(str(e))
        logger.error(f"Sequence {sequence_dir.name} failed: {e}")
        logger.debug(traceback.format_exc())

    if result["errors"] and result["status"] == "success":
        result["status"] = "partial"

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    type=click.Path(exists=True, path_type=Path),
    help="Single sequence directory to process",
)
@click.option(
    "-b",
    "--batch",
    "batch_path",
    type=click.Path(exists=True, path_type=Path),
    help="Batch mode: directory of sequences or .txt file with paths",
)
@click.option(
    "-o",
    "--output",
    "output_dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for CSV files",
)
@click.option(
    "-j",
    "--jobs",
    default=1,
    type=int,
    help="Number of parallel jobs (default: 1)",
)
# Use default True because the it seems that we cannot have access to
# offline calibrated gyro/accel biases, so we need online calibration
@click.option(
    "--use-online-calib",
    is_flag=True,
    default=True,
    help="Use online calibration instead of factory calibration",
)
@click.option(
    "--stop-on-error",
    is_flag=True,
    default=False,
    help="Stop processing on first error (default: continue)",
)
@click.option(
    "--hide-loading-logs",
    is_flag=True,
    default=False,
    help="Hide warnings and errors during dataset loading",
)
def main(
    input_path: Path | None,
    batch_path: Path | None,
    output_dir: Path,
    jobs: int,
    use_online_calib: bool,
    stop_on_error: bool,
    hide_loading_logs: bool,
):
    """
    Export Nymeria IMU trajectories to CSV files.

    Use -i for a single sequence or -b for batch processing.
    """

    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format="<level>{level: <7}</level> <light-blue>{name}.py:</light-blue><black>{function}</black><yellow>:{line}</yellow> {message}",
        level="INFO",
    )

    # Validate inputs
    if input_path is None and batch_path is None:
        raise click.UsageError("Must specify either -i (single) or -b (batch)")
    if input_path is not None and batch_path is not None:
        raise click.UsageError("Cannot specify both -i and -b")

    # Determine sequences to process
    if input_path is not None:
        sequences = [input_path]
    else:
        sequences = discover_sequences(batch_path)

    if not sequences:
        logger.error("No sequences found to process")
        sys.exit(1)

    logger.info(f"Processing {len(sequences)} sequence(s)")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Online calibration: {use_online_calib}")

    # Setup output and report
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    report = load_report(report_path)

    # Process sequences
    failed = 0

    if jobs == 1:
        # Single-threaded
        for seq in tqdm(sequences, desc="Sequences"):
            logger.info(f"Processing: {seq.name}")
            result = export_sequence(seq, output_dir, use_online_calib, hide_loading_logs=hide_loading_logs, quiet=False)
            report["sequences"][seq.name] = result
            save_report(report_path, report)

            if result["status"] == "failed":
                failed += 1
                if stop_on_error:
                    logger.error("Stopping due to --stop-on-error")
                    break
    else:
        # Multi-threaded
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(export_sequence, seq, output_dir, use_online_calib, hide_loading_logs=hide_loading_logs, quiet=True): seq
                for seq in sequences
            }

            with tqdm(total=len(sequences), desc="Sequences") as pbar:
                for future in as_completed(futures):
                    seq = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        result = {
                            "status": "failed",
                            "sequence": str(seq),
                            "errors": [str(e)],
                        }
                        logger.error(f"Sequence {seq.name} raised exception: {e}")

                    report["sequences"][seq.name] = result
                    save_report(report_path, report)

                    if result["status"] == "failed":
                        failed += 1
                        if stop_on_error:
                            logger.error("Stopping due to --stop-on-error")
                            executor.shutdown(wait=False, cancel_futures=True)
                            break

                    pbar.update(1)

    # Summary
    logger.info(f"Completed: {len(sequences) - failed}/{len(sequences)} sequences")
    logger.info(f"Report saved to: {report_path}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
