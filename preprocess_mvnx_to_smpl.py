"""
Batch MVNX to SMPL Preprocessor for Nymeria Dataset
Processes all MVNX files in data_xdata_mvnx and saves to data_smpl_from_xdata_mvnx

IMPORTANT: This script now DEFAULTS to computing local poses using the articulate library.
Local poses are essential for correct SMPL visualization.

Usage:
    # Default (with local poses - RECOMMENDED):
    python preprocess_mvnx_to_smpl.py

    # Without local poses (not recommended):
    python preprocess_mvnx_to_smpl.py --no-use-articulate
"""

import xml.etree.ElementTree as ET
import torch
import numpy as np
from typing import Tuple
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import pickle
torch.set_printoptions(sci_mode=False)

# SMPL joint mapping from Xsens (23 joints) to SMPL (24 joints)
XSENS_TO_SMPL = [0, 19, 15, 1, 20, 16, 3, 21, 17, 4, 22, 18, 
                  5, 11, 7, 6, 12, 8, 13, 9, 13, 9, 13, 9]

class MVNXBatchProcessor:
    """Batch processes MVNX files from Nymeria to SMPL format."""
    
    def __init__(self, input_dir: str, output_dir: str, use_articulate: bool = True):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_articulate = use_articulate

        # Initialize body_model by default for local pose computation
        self.body_model = None
        if use_articulate:
            try:
                import sys
                # Add mvnx_to_smpl to path if not already there
                mvnx_to_smpl_path = Path(__file__).parent / 'mvnx_to_smpl'
                if mvnx_to_smpl_path.exists() and str(mvnx_to_smpl_path) not in sys.path:
                    sys.path.insert(0, str(mvnx_to_smpl_path))

                import core.articulate as art
                from core.paths import Paths
                self.body_model = art.model.ParametricModel(
                    Paths.SMPL_FILE,
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
                self.art = art
                print(f"✓ Loaded articulate library. Local poses will be computed.")
            except ImportError as e:
                print(f"Warning: articulate library not found ({e}). Skipping local pose computation.")
                self.use_articulate = False
    
    def _parse_frame_data(self, frames) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parse orientation and position data from frames."""
        orientations = []
        root_positions = []
        
        for frame in frames:
            if frame.attrib['type'] == 'normal':
                # Parse quaternions (joint orientations)
                quat_data = [float(x) for x in frame[0].text.split(' ')]
                quat_tensor = torch.tensor(quat_data).view(-1, 4)
                orientations.append(quat_tensor)
                
                # Parse positions (joint positions)
                pos_data = [float(x) for x in frame[1].text.split(' ')]
                pos_tensor = torch.tensor(pos_data).view(-1, 3)
                root_positions.append(pos_tensor[0])  # (pelvis)
            
        if orientations:
            orientations = torch.stack(orientations)
            root_positions = torch.stack(root_positions)
        else:
            orientations = torch.empty(0, 23, 4)
            root_positions = torch.empty(0, 3)
        
        return orientations, root_positions
    
    def _convert_coordinate_system(self, orientations: torch.Tensor, root_positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert from Xsens to SMPL coordinate system."""
        # Reorder root position coordinates: [x,y,z] -> [y,z,x]
        root_positions_smpl = root_positions.clone()
        root_positions_smpl = root_positions_smpl[:, [1, 2, 0]]
        
        # Reorder quaternion components: [w,x,y,z] -> [w,z,x,y]
        orientations_smpl = orientations.clone()
        orientations_smpl[:, :, 1] = orientations[:, :, 2]  # x <- z
        orientations_smpl[:, :, 2] = orientations[:, :, 3]  # y <- x  
        orientations_smpl[:, :, 3] = orientations[:, :, 1]  # z <- y    
        
        return orientations_smpl, root_positions_smpl
    
    def _quaternion_to_rotation_matrix(self, quaternions: torch.Tensor) -> torch.Tensor:
        """Convert quaternions to rotation matrices."""
        # quaternions shape: [N, 4] where 4 = [w, x, y, z]
        w = quaternions[:, 0]
        x = quaternions[:, 1]
        y = quaternions[:, 2]
        z = quaternions[:, 3]
        
        # Normalize quaternions
        norm = torch.sqrt(w**2 + x**2 + y**2 + z**2)
        w = w / norm
        x = x / norm
        y = y / norm
        z = z / norm
        
        # Convert to rotation matrix
        rotation_matrices = torch.zeros(quaternions.shape[0], 3, 3)
        
        rotation_matrices[:, 0, 0] = 1 - 2*(y**2 + z**2)
        rotation_matrices[:, 0, 1] = 2*(x*y - w*z)
        rotation_matrices[:, 0, 2] = 2*(x*z + w*y)
        
        rotation_matrices[:, 1, 0] = 2*(x*y + w*z)
        rotation_matrices[:, 1, 1] = 1 - 2*(x**2 + z**2)
        rotation_matrices[:, 1, 2] = 2*(y*z - w*x)
        
        rotation_matrices[:, 2, 0] = 2*(x*z - w*y)
        rotation_matrices[:, 2, 1] = 2*(y*z + w*x)
        rotation_matrices[:, 2, 2] = 1 - 2*(x**2 + y**2)
        
        return rotation_matrices
    
    def _convert_to_smpl_format(self, orientations: torch.Tensor) -> torch.Tensor:
        """Convert Xsens joint orientations to SMPL format."""
        # Reshape orientations for batch processing
        batch_size = orientations.shape[0]
        num_joints = orientations.shape[1]
        
        # Convert quaternions to rotation matrices
        orientations_flat = orientations.view(-1, 4)
        if self.use_articulate and self.body_model is not None:
            rotation_matrices = self.art.math.quaternion_to_rotation_matrix(orientations_flat)
        else:
            rotation_matrices = self._quaternion_to_rotation_matrix(orientations_flat)
        
        glb_full_pose_xsens = rotation_matrices.view(batch_size, num_joints, 3, 3)
        
        # Map from Xsens to SMPL joints
        glb_full_pose_smpl = torch.eye(3).repeat(batch_size, 24, 1, 1)
        for smpl_idx, xsens_idx in enumerate(XSENS_TO_SMPL):
            glb_full_pose_smpl[:, smpl_idx, :] = glb_full_pose_xsens[:, xsens_idx, :]
        
        return glb_full_pose_smpl
    
    def _get_local_poses(self, global_poses: torch.Tensor) -> torch.Tensor:
        """Convert global poses to local poses using inverse kinematics."""
        if self.use_articulate and self.body_model is not None:
            local_poses = self.body_model.inverse_kinematics_R(global_poses)
            return local_poses.view(global_poses.shape[0], 24, 3, 3)
        else:
            # Without articulate, we can't compute local poses
            # This will result in incorrect SMPL visualization
            print("⚠ WARNING: Cannot compute local poses without articulate library!")
            print("⚠ The output SMPL data may not visualize correctly.")
            return None
    
    def process_mvnx_file(self, file_path: Path) -> dict:
        """Process a single MVNX file and return results."""
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            return None
        
        try:
            # Parse XML
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Extract metadata
            mvn_info = root.find('.//{http://www.xsens.com/mvn/mvnx}mvn')
            subject_info = root.find('.//{http://www.xsens.com/mvn/mvnx}subject')
            
            metadata = {
                'mvn_version': mvn_info.get('version') if mvn_info is not None else None,
                'frame_rate': int(subject_info.get('frameRate')) if subject_info is not None else 240,
                'segment_count': int(subject_info.get('segmentCount')) if subject_info is not None else 23,
            }
            
            # Extract joint names
            segments = root[2][1] if len(root) > 2 and len(root[2]) > 1 else None
            joint_names = []
            if segments is not None:
                joint_names = [seg.attrib.get('label', f'joint_{i}') for i, seg in enumerate(segments)]
            
            # Extract frame data
            frames = root[2][-1] if len(root) > 2 and len(root[2]) > 0 else None
            if frames is None:
                print(f"Warning: No frame data found in {file_path}")
                return None
            
            # Parse frame data
            orientations, root_positions = self._parse_frame_data(frames)
            
            if orientations.shape[0] == 0:
                print(f"Warning: No valid frames found in {file_path}")
                return None
            
            # Convert coordinate systems
            orientations, root_positions = self._convert_coordinate_system(orientations, root_positions)
            
            # Convert to SMPL format
            global_poses = self._convert_to_smpl_format(orientations)
            
            # Get local poses if articulate is available
            local_poses = self._get_local_poses(global_poses)
            
            return {
                'global_poses': global_poses,
                'local_poses': local_poses,
                'root_positions': root_positions,
                'joint_names': joint_names,
                'metadata': metadata,
                'num_frames': global_poses.shape[0]
            }
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def save_results(self, results: dict, output_path: Path):
        """Save processing results to files.

        Note: To ensure numpy version compatibility (1.x and 2.x), we only save
        numeric arrays in the npz file. Metadata is stored separately in JSON.
        This avoids pickle-related version incompatibilities.
        """
        if results is None:
            return

        # Save ONLY numeric arrays in npz (no object arrays that require pickle)
        # This ensures compatibility between numpy 1.x and 2.x
        np_results = {
            'global_poses': results['global_poses'].cpu().numpy().astype(np.float32),
            'root_positions': results['root_positions'].cpu().numpy().astype(np.float32),
        }

        if results['local_poses'] is not None:
            np_results['local_poses'] = results['local_poses'].cpu().numpy().astype(np.float32)

        # Save as compressed numpy file (numeric arrays only)
        np.savez_compressed(output_path.with_suffix('.npz'), **np_results)

        # Save ALL metadata in JSON (including joint_names, metadata, num_frames)
        # This is the authoritative source for non-numeric data
        json_metadata = {
            'joint_names': results['joint_names'],
            'metadata': results['metadata'],
            'num_frames': results['num_frames'],
            'global_poses_shape': list(results['global_poses'].shape),
            'root_positions_shape': list(results['root_positions'].shape)
        }
        if results['local_poses'] is not None:
            json_metadata['local_poses_shape'] = list(results['local_poses'].shape)

        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(json_metadata, f, indent=2)
    
    def process_all(self):
        """Process all MVNX files in the input directory."""
        # Find all body_xdata_mvnx files
        mvnx_files = list(self.input_dir.glob('*/body_xdata_mvnx'))
        print(f"Found {len(mvnx_files)} MVNX files to process")
        
        if len(mvnx_files) == 0:
            print("No MVNX files found!")
            return
        
        # Process each file
        success_count = 0
        error_count = 0
        
        for mvnx_file in tqdm(mvnx_files, desc="Processing MVNX files"):
            # Get relative path structure
            relative_dir = mvnx_file.parent.relative_to(self.input_dir)
            output_dir = self.output_dir / relative_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process file
            results = self.process_mvnx_file(mvnx_file)
            
            if results is not None:
                # Save results
                output_path = output_dir / "smpl_data"
                self.save_results(results, output_path)
                success_count += 1
            else:
                error_count += 1
        
        print(f"\nProcessing complete!")
        print(f"Successfully processed: {success_count} files")
        print(f"Errors: {error_count} files")
        print(f"Local poses computed: {'Yes' if self.use_articulate and self.body_model is not None else 'No'}")

        # Save processing summary
        summary = {
            'total_files': len(mvnx_files),
            'success_count': success_count,
            'error_count': error_count,
            'input_dir': str(self.input_dir),
            'output_dir': str(self.output_dir),
            'local_poses_computed': self.use_articulate and self.body_model is not None
        }
        
        with open(self.output_dir / 'processing_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary saved to {self.output_dir / 'processing_summary.json'}")


def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(description="Batch process MVNX files to SMPL format")
    parser.add_argument(
        "--input-dir", 
        type=str, 
        default="/mnt/nas2/naoto/nymeria_dataset/data_xdata_mvnx",
        help="Input directory containing MVNX files"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="/mnt/nas2/naoto/nymeria_dataset/data_smpl_from_xdata_mvnx",
        help="Output directory for SMPL data"
    )
    parser.add_argument(
        "--use-articulate",
        action="store_true",
        default=True,
        help="Use articulate library for local pose computation (default: True, disable with --no-use-articulate)"
    )
    parser.add_argument(
        "--no-use-articulate",
        action="store_false",
        dest="use_articulate",
        help="Disable articulate library (not recommended - will skip local pose computation)"
    )
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = MVNXBatchProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        use_articulate=args.use_articulate
    )
    
    # Process all files
    processor.process_all()


if __name__ == "__main__":
    main()