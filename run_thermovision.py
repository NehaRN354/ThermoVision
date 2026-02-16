"""
ThermoVision - Easy Startup Script
Run this file to start ThermoVision
"""

import sys
import os

# Add src to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

# Import and run main application
from main import main

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║                                                            ║
    ║   ████████╗██╗  ██╗███████╗██████╗ ███╗   ███╗ ██████╗   ║
    ║   ╚══██╔══╝██║  ██║██╔════╝██╔══██╗████╗ ████║██╔═══██╗  ║
    ║      ██║   ███████║█████╗  ██████╔╝██╔████╔██║██║   ██║  ║
    ║      ██║   ██╔══██║██╔══╝  ██╔══██╗██║╚██╔╝██║██║   ██║  ║
    ║      ██║   ██║  ██║███████╗██║  ██║██║ ╚═╝ ██║╚██████╔╝  ║
    ║      ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝ ╚═════╝   ║
    ║                                                            ║
    ║              VISION - Heat & Hazard Awareness              ║
    ║                  For Visually Impaired                     ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝
    """)

    sys.exit(main())