import mujoco
import mujoco.viewer
import os
from pathlib import Path

# --- File path setup ---
# Absolute path to the directory where this script lives (e.g., /planning/kinova_gen3/)
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Name of the main XML file to load
MODEL_XML_FILENAME = "scene.xml"

# ---------------------

def main():
    # ⭐ Key step: change the working directory BEFORE loading the XML.
    # MuJoCo resolves relative paths inside the XML (e.g., <include file="gen3.xml"/>,
    # <mesh file="assets/...">) relative to the CURRENT_DIR set here.
    os.chdir(CURRENT_DIR)

    # Build the absolute path to the XML file (or you could pass just the name,
    # since we already changed the CWD above).
    MODEL_XML_PATH = CURRENT_DIR / MODEL_XML_FILENAME

    print(f"Working directory changed to: {CURRENT_DIR}")
    print(f"Attempting to load model from: {MODEL_XML_PATH}")

    # 1) Load MuJoCo model
    try:
        # ⭐ Important: do not pass extra 'assets' or 'file_dir' args.
        # Since the working directory is correct, MuJoCo will find all included files.
        model = mujoco.MjModel.from_xml_path(str(MODEL_XML_PATH))
    except Exception as e:
        print(f"⚠️ Error while loading model: {e}")
        return

    data = mujoco.MjData(model)

    print("✅ Model loaded. Launching MuJoCo viewer...")

    # 2) Launch MuJoCo passive viewer (simulation loop)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
