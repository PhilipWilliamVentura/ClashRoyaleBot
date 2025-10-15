# Clash Royale AI Bot 🤖♠️

A Python-based AI bot for **Clash Royale** that learns and improves gameplay through **reinforcement learning**. This project demonstrates **Double DQN**, **ResNet-based neural networks**, and **object detection**, providing a practical showcase of machine learning, game automation, and computer vision.

> ⚠️ **Disclaimer:** This project is not affiliated with Supercell. Use at your own risk—automated gameplay may violate Clash Royale’s Terms of Service.

---

## 🚀 Built With

- **Python 3.12** – Core language for AI and automation  
- **PyTorch** – Deep learning framework for reinforcement learning  
- **Docker** – Containerization for consistent environments  
- **Roboflow** – Object detection for troops and cards  
- **PyAutoGUI** – Game interaction automation  
- **NumPy** – Numerical computation  

---

## 🎯 Features

- **Double DQN agent** that learns optimal strategies in Clash Royale  
- **ResNet-based neural network** for game state analysis  
- **Object detection model** via Roboflow for troop and card recognition  
- Automated gameplay through **BlueStacks emulator**  
- Modular, easy-to-extend codebase for AI experimentation  

---

## 🛠 Getting Started

### Prerequisites

- macOS (only supported OS currently)  
- VSCode or preferred code editor  
- Docker  
- Roboflow Account  
- BlueStacks  
- Python 3.11  

### Installation

1. Install required Python packages:

```bash
pip install -r requirements.txt
```

2. Create a Roboflow account & workspace, then get your API key.

3. Clone the repository:
```bash
git clone https://github.com/PhilipWilliamVentura/ClashRoyaleBot.git
cd ClashRoyaleBot
```

4. Set up environment variables:
```bash
ROBOFLOW_API_KEY=your_actual_api_key_here
WORKSPACE_NAME=your_actual_workspace_name
```

5. Fork Roboflow workflows: Troop Detection, Card Detection & Tower Detection, then update .env with your workspace names.

Go in Docker terminal
6. Install inference-cli
```bash
pip install inference-cli
```

7. Start the Inference Server:
```bash
inference server start
```

### Setup BlueStacks
Open BlueStacks and create a 64-bit Pie instance via Multi-Instance Manager.
Install Clash Royale from Google Play Store.
Resize and position the window on your screen for optimal automation.
Optional: Disable ads for smoother gameplay:
Settings → Preferences → Allow BlueStacks to show Ads → Disabled
