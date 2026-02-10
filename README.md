# Simple Vehicle Counting - Nagrak Toll Gate

![Tracking Demo](assets/tracking-demo.gif)

This is a personal project to explore Computer Vision concepts using Python. It demonstrates a simple way to detect, track, and count vehicles (Car, Bus, Truck) from a CCTV video file recorded at **Nagrak Toll Gate Exit (Kota Wisata), Kabupaten Bogor**.

The goal of this project is to create a functional dashboard using **YOLOv5** and **Streamlit** that counts vehicles crossing a virtual line, distinguishing between **Inbound** and **Outbound** traffic. It serves as a basic reference implementation for traffic analysis in a real-world setting.

## How it Works

![Demo Aplikasi](assets/dashboard-demo.gif)

The logic is straightforward:
1.  **Detect:** Uses a pre-trained `yolov5s` model to find vehicles in each frame.
2.  **Track:** Uses a simple centroid tracking method (calculating distance between frames) to give each vehicle an ID.
3.  **Count:** Checks if the vehicle's center point crosses a horizontal line.
    * Moving Down = **Outbound (Exit)**
    * Moving Up = **Inbound (Entry)**

## Project Structure

* `app.py` - The main script containing the logic and Streamlit UI.
* `requirements.txt` - List of libraries needed.
* `data/input/` - Folder to store the video file.

## Installation & Run

1.  **Clone this repo:**
    ```bash
    git clone [https://github.com/adolesans/Nagrak-Vehicle-Detection-Tracking-and-Counting.git](https://github.com/adolesans/Nagrak-Vehicle-Detection-Tracking-and-Counting.git)
    cd Nagrak-Vehicle-Detection-Tracking-and-Counting
    ```

2.  **Install libraries:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app:**
    ```bash
    streamlit run app.py
    ```

## Code Snippet

Here is the simple logic used for tracking the centroids:

```python
# Simple distance-based tracking
for rect in detections:
    cx, cy = calculate_center(rect)
    # Check distance to existing objects...
    if distance < threshold:
        update_id()
    else:
        create_new_id()
