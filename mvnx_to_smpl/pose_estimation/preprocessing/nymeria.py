"""
MVNX Motion Capture Data Preprocessor
"""

import xml.etree.ElementTree as ET
import torch
torch.set_printoptions(sci_mode=False)
from typing import Tuple, List
from pathlib import Path
from argparse import ArgumentParser

import core.articulate as art
from core.paths import Paths


# SMPL joint mapping from Xsens (23 joints) to SMPL (24 joints)
XSENS_TO_SMPL = [0, 19, 15, 1, 20, 16, 3, 21, 17, 4, 22, 18, 
                            5, 11, 7, 6, 12, 8, 13, 9, 13, 9, 13, 9]

MAX_FRAMES = 2000 # max frames to visualize

class MVNXProcessor:
    """Processes MVNX files from Nymeria to obtain SMPL pose and translation."""
    def __init__(self):
        self.body_model = art.model.ParametricModel(Paths.SMPL_FILE, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_frames = MAX_FRAMES
    
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
            
        orientations = torch.stack(orientations)
        root_positions = torch.stack(root_positions)
        
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
    
    def _convert_to_smpl_format(self, orientations: torch.Tensor) -> torch.Tensor:
        """Convert Xsens joint orientations to SMPL format."""
        glb_full_pose_xsens = art.math.quaternion_to_rotation_matrix(orientations).view(-1, 23, 3, 3)
        glb_full_pose_smpl = torch.eye(3).repeat(glb_full_pose_xsens.shape[0], 24, 1, 1)
        for smpl_idx, xsens_idx in enumerate(XSENS_TO_SMPL):
            glb_full_pose_smpl[:, smpl_idx, :] = glb_full_pose_xsens[:, xsens_idx, :]
        return glb_full_pose_smpl
    
    def _get_local_poses(self, global_poses: torch.Tensor) -> torch.Tensor:
        """Convert global poses to local poses using inverse kinematics."""
        local_poses = self.body_model.inverse_kinematics_R(global_poses)
        return local_poses.view(global_poses.shape[0], 24, 3, 3)
    
    def read_mvnx(self, file_path: str):
        """Read and process MVNX file."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"MVNX file not found: {file_path}")
        
        # Parse XML
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Extract joint names
        segments = root[2][1]
        joint_names = [seg.attrib['label'] for seg in segments]
        
        # Extract frame data
        frames = root[2][-1]
        
        # Parse frame data
        orientations, root_positions = self._parse_frame_data(frames)
        
        # Convert coordinate systems
        orientations, root_positions = self._convert_coordinate_system(orientations, root_positions)
        
        # Convert to SMPL format
        global_poses = self._convert_to_smpl_format(orientations)
        
        return global_poses, joint_names, root_positions
    
    def visualize_motion(self, global_poses: torch.Tensor, root_positions: torch.Tensor, fps: int = 60):
        """Visualize motion using SMPL body model."""
        # Convert to local poses
        local_poses = self._get_local_poses(global_poses)
        root_translation = root_positions

        # Skip frames for visualization
        local_poses = local_poses[:self.max_frames]
        root_translation = root_translation[:self.max_frames]
        
        # Visualize
        self.body_model.view_motion([local_poses], [root_translation], fps=fps)
    

def main():
    """Main processing function."""
    parser = ArgumentParser(description="Process MVNX files from Nymeria")
    parser.add_argument("--file-path", type=str, required=True, help="Path to MVNX file")
    args = parser.parse_args()
    
    # Initialize preprocessor
    processor = MVNXProcessor()
    
    # Process file
    global_poses, joint_names, root_positions = processor.read_mvnx(args.file_path)
    
    # Print results
    print(f"\nProcessing Results:")
    print("-"*20)
    print(f"Global poses shape: {global_poses.shape}")
    print(f"Root positions shape: {root_positions.shape}")
    print(f"Number of joints: {len(joint_names)}")
    
    # Visualize motion
    print("\nStarting motion visualization...")
    processor.visualize_motion(global_poses, root_positions)


if __name__ == "__main__":
    main()