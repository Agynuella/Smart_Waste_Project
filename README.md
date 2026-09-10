# 🌍 Explainable AI-Based Smart Waste Management System

## 📌 Project Overview
This repository contains the software prototype for a final-year academic project titled: **"Development of an Explainable AI-Based Smart Waste Management System."** 

The system modernizes municipal waste collection by integrating IoT sensor data, Deep Learning for waste classification, Explainable AI (XAI) for decision transparency, and Operations Research for dynamic route optimization.

## ✨ Core Features
1. **IoT Edge Simulation (`smart_bin_node.py`):** Simulates edge devices capturing bin fill-levels and triggering camera captures.
2. **AI Waste Classification (`train_waste_model.py`):** A custom-trained MobileNetV2 neural network utilizing the TrashNet dataset to classify waste into 6 categories (Cardboard, Glass, Metal, Paper, Plastic, Trash).
3. **Explainable AI (`explain_model.py`):** Integrates Grad-CAM to generate heatmaps, providing visual proof and transparency of the AI's classification logic.
4. **Dynamic Route Optimization (`route_optimizer.py`):** Utilizes Google OR-Tools to solve the Vehicle Routing Problem (VRP), filtering out empty bins and calculating the most fuel-efficient collection path.
5. **Interactive Web Dashboard (`dashboard.py`):** A Streamlit-based interface for city managers to monitor live bin status, view optimized routes, and audit AI decisions.

## 🛠️ Technology Stack
* **Programming Language:** Python 3.10+
* **Deep Learning Framework:** PyTorch, Torchvision
* **XAI Framework:** pytorch-grad-cam
* **Optimization Engine:** Google OR-Tools
* **Web Dashboard:** Streamlit, Pandas
* **Computer Vision:** OpenCV, Pillow, Matplotlib

## 🚀 How to Run Locally
1. Clone this repository:
   ```bash
   git clone [https://github.com/YourUsername/Smart_Waste_Project.git](https://github.com/YourUsername/Smart_Waste_Project.git)
