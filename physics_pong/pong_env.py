import mujoco
import mujoco.viewer
import os

def main():
    # Construct the absolute path to the XML model file
    # This ensures the model is found regardless of where the script is run from.
    base_path = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(base_path, "lib/myosuite/envs/myo/assets/arm/myoarm_pong.xml")
    
    if not os.path.exists(xml_path):
        print(f"Error: XML file not found at {xml_path}")
        return

    print(f"Loading MuJoCo model from: {xml_path}")
    
    # Load the model and create a data object for simulation
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        
        # Optional: Reset to the first keyframe ('default' in your XML)
        # mujoco.mj_resetDataKeyframe(model, data, 0)
        
    except Exception as e:
        print(f"Failed to load MuJoCo model: {e}")
        return

    # Launch the interactive passive viewer
    # This provides a full-featured GUI for visualization and interaction.
    # The viewer handles the simulation stepping internally.
    print("Launching interactive viewer...")
    print("Controls:")
    print(" - Space: Pause/Unpause simulation")
    print(" - Left Click & Drag: Rotate view")
    print(" - Right Click & Drag: Translate view")
    print(" - Scroll: Zoom")
    print(" - Drag with right mouse button: Apply forces to bodies")
    
    mujoco.viewer.launch(model, data)

if __name__ == "__main__":
    main()

